import argparse
import os
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib
# matplotlib.use('TkAgg') # 如有需要可取消注释

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from model import PRPSegmenter


def load_model(model_path: str, device: torch.device) -> PRPSegmenter:
    model = PRPSegmenter(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def generate_gaussian_heatmap(height: int, width: int, center: Tuple[float, float], sigma: float) -> np.ndarray:
    y = np.arange(0, height, 1, float)
    x = np.arange(0, width, 1, float)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    heatmap = np.exp(-((yy - center[1]) ** 2 + (xx - center[0]) ** 2) / (2 * sigma ** 2))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap.astype(np.float32)


def make_disk(radius: int) -> np.ndarray:
    grid = np.arange(-radius, radius + 1)
    gx, gy = np.meshgrid(grid, grid, indexing="xy")
    return (gx ** 2 + gy ** 2) <= radius ** 2


def greedy_circle_centers(
        band_mask: np.ndarray,
        min_center_dist: int,
        rng: np.random.Generator,
        existing: List[Tuple[int, int]] | None = None,
) -> List[Tuple[int, int]]:
    ys, xs = np.where(band_mask)
    if len(xs) == 0:
        return []
    order = rng.permutation(len(xs))
    centers: List[Tuple[int, int]] = []
    existing = existing or []

    # 优化：如果候选点过多，先进行降采样，防止循环过久
    if len(order) > 5000:
        order = order[::10]

    for idx in order:
        y, x = ys[idx], xs[idx]
        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_center_dist ** 2 for cy, cx in centers):
            if all((x - cx) ** 2 + (y - cy) ** 2 >= min_center_dist ** 2 for cy, cx in existing):
                centers.append((y, x))
    return centers


def plan_circle_layout(
        gt1_mask: np.ndarray,
        radius: int,
        spacing: int,
) -> List[Tuple[int, int]]:
    if gt1_mask.sum() == 0:
        return []

    effective_spacing = spacing + 2 * radius
    dist = ndi.distance_transform_edt(~gt1_mask)
    centers: List[Tuple[int, int]] = []
    rng = np.random.default_rng(42)
    band_half_width = max(1, effective_spacing // 3)

    for ring_idx in range(3):
        target = radius + spacing + ring_idx * effective_spacing
        band = (
                (dist >= target - band_half_width)
                & (dist <= target + band_half_width)
                & (~gt1_mask)
        )
        if band.sum() == 0:
            continue
        new_centers = greedy_circle_centers(
            band_mask=band,
            min_center_dist=effective_spacing,
            rng=rng,
            existing=centers,
        )
        centers.extend(new_centers)
    return centers


def draw_circles_on_image(base_image: Image.Image, centers: List[Tuple[int, int]], diameter: int) -> Image.Image:
    canvas = base_image.copy()
    draw = ImageDraw.Draw(canvas)
    radius = diameter // 2
    for y, x in centers:
        bbox = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(bbox, outline="blue", width=2)
    return canvas


def filter_components_by_confidence(
        mask: np.ndarray, logits: np.ndarray, confidence_threshold: float
) -> np.ndarray:
    labeled, num = ndi.label(mask)
    if num == 0:
        return mask
    keep = np.zeros(num + 1, dtype=bool)
    high_conf = logits > confidence_threshold
    for idx in range(1, num + 1):
        if np.any(high_conf & (labeled == idx)):
            keep[idx] = True
    return keep[labeled]


# ================= 核心修复：极速版连接算法 =================
def connect_close_components(mask: np.ndarray, max_distance: float = 100.0) -> np.ndarray:
    def draw_line(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
        y0, x0 = a
        y1, x1 = b
        points = []
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sy = 1 if y0 < y1 else -1
        sx = 1 if x0 < x1 else -1
        err = dx - dy
        while True:
            points.append((y0, x0))
            if y0 == y1 and x0 == x1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    updated = mask.copy()

    # 限制最大迭代次数，防止极其糟糕的情况死循环
    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1
        labeled, num = ndi.label(updated)

        # 如果连通域太多（例如 > 100 个），说明全是噪声，强行停止合并，避免死机
        if num > 100:
            print(f"Warning: 检测到 {num} 个连通区域，噪声过多，停止合并以保护程序。")
            break
        if num <= 1:
            break

        objects = ndi.find_objects(labeled)
        components_coords = []

        # 提取轮廓并稀疏采样的关键步骤
        for idx, slice_obj in enumerate(objects):
            if slice_obj is None:
                components_coords.append(np.array([]))
                continue

            # 1. 获取局部掩膜
            y_slice, x_slice = slice_obj
            local_mask = (labeled[slice_obj] == (idx + 1))

            # 2. 提取轮廓 (Erosion 算法: 实心 - 腐蚀 = 轮廓)
            # 这会将点数从 10000 -> 400
            local_eroded = ndi.binary_erosion(local_mask)
            local_boundary = local_mask ^ local_eroded
            local_coords = np.argwhere(local_boundary)

            # 3. 稀疏采样 (Downsampling)
            # 每隔 10 个点取一个，这会将点数从 400 -> 40
            # 对于 100 像素的距离判断，这种精度完全足够
            if len(local_coords) > 0:
                step = 10
                local_coords = local_coords[::step]

            # 如果轮廓太小没取到点，就退化为取所有点
            if len(local_coords) == 0:
                local_coords = np.argwhere(local_mask)
                if len(local_coords) > 0:
                    local_coords = local_coords[::10]  # 依然降采样

            global_coords = local_coords + np.array([y_slice.start, x_slice.start])
            components_coords.append(global_coords)

        merged = False

        for i in range(num):
            coords_i = components_coords[i]
            if coords_i is None or coords_i.size == 0: continue

            for j in range(i + 1, num):
                coords_j = components_coords[j]
                if coords_j is None or coords_j.size == 0: continue

                # 此时传入 tree 的点非常少，速度极快
                tree = cKDTree(coords_j)
                dists, idxs = tree.query(coords_i)

                min_idx = np.argmin(dists)
                min_dist = dists[min_idx]

                if min_dist < max_distance:
                    p1 = coords_i[min_idx]
                    p2 = coords_j[idxs[min_idx]]

                    start_point = (int(p1[0]), int(p1[1]))
                    target_point = (int(p2[0]), int(p2[1]))

                    for y, x in draw_line(start_point, target_point):
                        updated[y, x] = True

                    merged = True
                    break

            if merged:
                break

        if not merged:
            break

    return updated


# ==========================================================


def infer_with_click(
        model: PRPSegmenter,
        image: Image.Image,
        click_xy: Tuple[float, float] | None,
        sigma: float,
        device: torch.device,
        gt1_threshold: float,
        confidence_threshold: float,
        circle_diameter: int,
        circle_spacing: int,
) -> Tuple[np.ndarray, Image.Image]:
    input_image = image.resize((1280, 1280), Image.BILINEAR)
    if click_xy is None:
        heatmap_np = np.zeros((1280, 1280), dtype=np.float32)
    else:
        click_scaled = (click_xy[0] / 1240 * 1280, click_xy[1] / 1240 * 1280)
        heatmap_np = generate_gaussian_heatmap(1280, 1280, click_scaled, sigma)

    image_tensor = pil_to_tensor(input_image).unsqueeze(0).to(device)
    heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred1 = model(image_tensor, heatmap_tensor)

    pred1_resized = F.interpolate(pred1, size=(1240, 1240), mode="bilinear", align_corners=False)
    logits = pred1_resized.squeeze().cpu().numpy()

    gt1_initial = logits >= gt1_threshold
    gt1_dilated = ndi.binary_dilation(gt1_initial, structure=make_disk(12))
    gt1_filtered = filter_components_by_confidence(gt1_dilated, logits, confidence_threshold)

    # 面积过滤：调大 min_size，进一步减少进入后续计算的噪点
    labeled_temp, num_temp = ndi.label(gt1_filtered)
    if num_temp > 0:
        sizes = ndi.sum(gt1_filtered, labeled_temp, range(1, num_temp + 1))
        min_size = 50  # 建议调至 50 或 100
        remove_indices = np.where(sizes < min_size)[0] + 1
        mask_remove = np.isin(labeled_temp, remove_indices)
        gt1_filtered[mask_remove] = False

    gt1_connected = connect_close_components(gt1_filtered, max_distance=100.0)

    centers = plan_circle_layout(
        gt1_mask=gt1_connected,
        radius=circle_diameter // 2,
        spacing=circle_spacing,
    )
    resized_image = image.resize((1240, 1240), Image.BILINEAR)
    overlay = draw_circles_on_image(resized_image, centers, diameter=circle_diameter)

    return gt1_connected.astype(np.uint8), overlay


def display_intermediate(image: Image.Image) -> Tuple[float, float]:
    print("正在打开图像窗口，请在病灶位置点击鼠标左键...")

    plt.ion()
    fig = plt.figure("Input Image", figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("单击选择提示位置 (Click to select prompt)")

    plt.draw()
    plt.pause(0.1)

    coords = plt.ginput(1, timeout=0, mouse_add=1, mouse_stop=None, mouse_pop=None)

    plt.close(fig)
    plt.ioff()

    if not coords:
        raise RuntimeError("未检测到点击，或者窗口被直接关闭。请重新运行并点击图像。")

    print(f"捕获点击坐标: {coords[0]}")
    return coords[0]


def visualize_results(
        original: Image.Image, gt1: np.ndarray, overlay: Image.Image, fig: plt.Figure | None
) -> plt.Figure:
    if fig is None or not plt.fignum_exists(fig.number):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        axes = fig.axes
        if len(axes) != 3:
            fig.clf()
            axes = fig.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("原图")
    axes[1].imshow(gt1, cmap="gray")
    axes[1].set_title("gt_1 后处理")
    axes[2].imshow(overlay)
    axes[2].set_title("gt_1 蓝色圆形标注")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig


def save_outputs(gt1: np.ndarray, overlay: Image.Image, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    gt1_img = Image.fromarray(gt1 * 255)
    gt1_path = output_dir / f"{stem}_gt_1.png"
    overlay_path = output_dir / f"{stem}_gt_1_overlay.png"
    gt1_img.save(gt1_path)
    overlay.save(overlay_path)
    print(f"保存 gt_1 至 {gt1_path}")
    print(f"保存带圆形标注的 gt_1 至 {overlay_path}")


def main():
    parser = argparse.ArgumentParser(description="交互式点击预测并生成后处理分割结果")
    parser.add_argument("--model-path", default=r"C:\work space\liekong\predict_251230_demo\best_epoch_223_dice_0.6240.pth",
                        help="模型权重路径")
    parser.add_argument("--image-path", default=r"C:\work space\liekong\predict_251230_demo\test", help="待预测图像路径或目录")
    parser.add_argument("--output-dir", default="outputs", help="输出保存目录")
    parser.add_argument("--device", default=None, help="使用的设备，如 cuda:0 或 cpu")
    parser.add_argument("--gt1-threshold", type=float, default=0.5, help="gt_1 阈值")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="保留连通区域所需的最小 logits 阈值",
    )
    parser.add_argument("--sigma", type=float, default=15.0, help="点击生成高斯热图的标准差")
    parser.add_argument("--circle-diameter", type=int, default=15, help="绘制蓝色圆的直径")
    parser.add_argument(
        "--circle-spacing",
        type=int,
        default=5,
        help="蓝色圆之间的最小边界间距（像素）",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.model_path):
        print(f"Error: 模型文件不存在: {args.model_path}")
        return
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: 图像文件不存在: {args.image_path}")
        return

    if image_path.is_dir():
        image_files = sorted(
            [
                p
                for p in image_path.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            ]
        )
    else:
        image_files = [image_path]

    if not image_files:
        print(f"Error: 在目录 {image_path} 中未找到可用图像。")
        return

    model = load_model(args.model_path, device)

    for img_idx, img_path in enumerate(image_files):
        print(f"\n正在处理第 {img_idx + 1}/{len(image_files)} 张图像: {img_path}")
        image = Image.open(img_path).convert("RGB")
        display_image = image.resize((1240, 1240), Image.BILINEAR)

        result_figs: List[plt.Figure] = []
        click_counter = 0

        def save_and_show(gt1: np.ndarray, overlay: Image.Image, suffix: str):
            fig = visualize_results(display_image, gt1, overlay, fig=None)
            result_figs.append(fig)
            stem = f"{img_path.stem}_{suffix}"
            save_outputs(gt1, overlay, Path(args.output_dir), stem)

        def handle_click(event, *, on_infer: Callable[[Tuple[float, float]], Tuple[np.ndarray, Image.Image]]):
            nonlocal click_counter
            if event.inaxes is None or event.button != 1:
                return
            click_xy = (event.xdata, event.ydata)
            print(f"捕获点击坐标: {click_xy}")
            gt1, overlay = on_infer(click_xy)
            suffix = f"click_{click_counter}"
            click_counter += 1
            save_and_show(gt1, overlay, suffix)

        def run_inference(click_xy: Tuple[float, float] | None):
            return infer_with_click(
                model=model,
                image=image,
                click_xy=click_xy,
                sigma=args.sigma,
                device=device,
                gt1_threshold=args.gt1_threshold,
                confidence_threshold=args.confidence_threshold,
                circle_diameter=args.circle_diameter,
                circle_spacing=args.circle_spacing,
            )

        print("执行自动推理（无点击，高斯热图全黑）...")
        auto_gt1, auto_overlay = run_inference(None)
        save_and_show(auto_gt1, auto_overlay, "auto")

        main_fig = plt.figure("Input Image", figsize=(8, 8))
        ax = main_fig.add_subplot(111)
        ax.imshow(display_image)
        ax.axis("off")
        ax.set_title("单击选择提示位置 (Click to select prompt)")

        main_fig.canvas.mpl_connect(
            "button_press_event",
            lambda event: handle_click(event, on_infer=lambda xy: run_inference(xy)),
        )

        def on_main_close(event):
            if event.canvas.figure is main_fig:
                for fig in result_figs:
                    if plt.fignum_exists(fig.number):
                        plt.close(fig)

        main_fig.canvas.mpl_connect("close_event", on_main_close)

        print("图像窗口已打开，监听点击事件...")
        plt.show()
        plt.close("all")


if __name__ == "__main__":
    main()