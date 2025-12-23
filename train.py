import argparse
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PRPDataset
from model import PRPSegmenter
from utils import dice_coefficient, iou_score


def default_device() -> str:
    """Prefer CUDA when available, defaulting to GPU 0 for multi-GPU training."""
    if torch.cuda.is_available():
        return "cuda:1"              ###同步修改
    return "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_image(tensor: torch.Tensor) -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np

    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)


def save_validation_batch(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    masks1: torch.Tensor,
    preds1: torch.Tensor,
    save_root: str,
    epoch: int,
    batch_idx: int,
    original_size: Tuple[int, int] = (1240, 1240),
) -> None:
    import cv2
    import numpy as np

    epoch_dir = os.path.join(save_root, f"epoch_{epoch:03d}")
    ensure_dir(epoch_dir)

    for i in range(images.shape[0]):
        image_np = tensor_to_image(images[i])
        heatmap_np = heatmaps[i, 0].detach().cpu().numpy()
        mask1_np = masks1[i, 0].detach().cpu().numpy()
        pred1_np = preds1[i, 0].detach().cpu().numpy()

        has_click = heatmap_np.max() > 0
        if has_click:
            click_y, click_x = divmod(heatmap_np.argmax(), heatmap_np.shape[1])
            scale_y = original_size[0] / heatmap_np.shape[0]
            scale_x = original_size[1] / heatmap_np.shape[1]
            click_y_resized = int(click_y * scale_y)
            click_x_resized = int(click_x * scale_x)
        else:
            click_y_resized = None
            click_x_resized = None

        image_resized = cv2.resize(image_np, (original_size[1], original_size[0]))
        heatmap_resized = cv2.resize(heatmap_np, (original_size[1], original_size[0]))
        mask1_resized = cv2.resize(mask1_np, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        pred1_resized = cv2.resize(pred1_np, (original_size[1], original_size[0]))

        image_with_click = image_resized.copy()
        if has_click and click_x_resized is not None and click_y_resized is not None:
            cv2.circle(
                image_with_click,
                (int(click_x_resized), int(click_y_resized)),
                8,
                (255, 0, 0),
                thickness=-1,
            )
        else:
            cv2.putText(
                image_with_click,
                "no_click",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
                lineType=cv2.LINE_AA,
            )

        pred1_mask = (pred1_resized > 0.5).astype(np.uint8) * 255
        gt1_mask = (mask1_resized > 0.5).astype(np.uint8) * 255

        basename = f"sample_{batch_idx:03d}_{i:02d}"
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_image.png"), cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_click.png"), cv2.cvtColor(image_with_click, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_heatmap.png"), (heatmap_resized * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_pred1.png"), pred1_mask)
        cv2.imwrite(os.path.join(epoch_dir, f"{basename}_gt1.png"), gt1_mask)


def dice_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(min=1e-6, max=1 - 1e-6)
    dice = dice_coefficient(pred, target).mean()
    bce = nn.functional.binary_cross_entropy(pred, target)
    return (1 - dice) + 0.5 * bce


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_root: Optional[str] = None,
    epoch: Optional[int] = None,
) -> tuple[float, float]:
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for batch_idx, (images, heatmaps, masks1) in enumerate(loader):
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            masks1 = masks1.to(device)
            preds1 = model(images, heatmaps)
            dice_scores.append(dice_coefficient(preds1, masks1).mean().item())
            iou_scores.append(iou_score(preds1, masks1).mean().item())

            if save_root and epoch is not None:
                save_validation_batch(
                    images=images,
                    heatmaps=heatmaps,
                    masks1=masks1,
                    preds1=preds1,
                    save_root=save_root,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )
    model.train()
    return float(sum(dice_scores) / len(dice_scores)), float(sum(iou_scores) / len(iou_scores))


def train(
    train_dir: str,
    val_dir: Optional[str],
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 5e-4,
    num_workers: int = 4,
    device: str = default_device(),
    use_visdom: bool = False,
    visdom_env: str = "prp_segmentation",
    visdom_port: int = 8097,
    output_dir: str = "output",
):
    device = torch.device(device)
    train_dataset = PRPDataset(train_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = PRPDataset(val_dir, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = PRPSegmenter()
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print("Using DataParallel on GPUs: 0 and 1")
        model = nn.DataParallel(model, device_ids=[1])
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    viz = None
    if use_visdom:
        import visdom

        viz = visdom.Visdom(env=visdom_env, port=visdom_port)
        if not viz.check_connection():
            print("[Visdom] Connection failed. Visualizations will be skipped.")
            viz = None

    ensure_dir(output_dir)
    best_val_dice = float("-inf")
    best_epoch: Optional[int] = None

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        def log_to_visdom(
            batch_images: torch.Tensor,
            batch_heatmaps: torch.Tensor,
            batch_masks1: torch.Tensor,
            batch_preds1: torch.Tensor,
        ) -> None:
            """Visualize the current batch on Visdom."""

            if viz is None:
                return

            def _prep_single(t: torch.Tensor) -> torch.Tensor:
                tensor = t.detach().cpu().clamp(0, 1)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)
                if tensor.shape[0] == 1:
                    tensor = tensor.repeat(3, 1, 1)
                return tensor

            img = _prep_single(batch_images[0])
            heatmap = _prep_single(batch_heatmaps[0])
            gt1 = _prep_single(batch_masks1[0])
            pred1 = _prep_single(batch_preds1[0])

            viz.image(img, win="input_image", opts={"title": f"Input Epoch {epoch}"})
            viz.image(heatmap, win="heatmap", opts={"title": f"Heatmap Epoch {epoch}"})
            viz.image(gt1, win="ground_truth_1", opts={"title": f"GT1 Epoch {epoch}"})
            viz.image(pred1, win="prediction_1", opts={"title": f"Pred1 Epoch {epoch}"})

        for images, heatmaps, masks1 in progress:
            images = images.to(device)
            heatmaps = heatmaps.to(device)
            masks1 = masks1.to(device)

            preds1 = model(images, heatmaps)
            loss = dice_bce_loss(preds1, masks1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

            log_to_visdom(images, heatmaps, masks1, preds1)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}")

        # ==============================================================================
        # 核心修改：统一提取单卡模型用于验证和评估
        # 这确保了无论是在验证集还是在训练集上跑 evaluate，都不会因为 Batch 切分问题
        # 导致其中一张卡分到空数据而崩溃。
        # ==============================================================================
        eval_model = model.module if isinstance(model, nn.DataParallel) else model

        if val_loader:
            val_save_dir = os.path.join(output_dir, "val_outputs")

            # 使用 eval_model (单卡) 进行验证
            val_dice, val_iou = evaluate(eval_model, val_loader, device, save_root=val_save_dir, epoch=epoch)
            print(f"Epoch {epoch}: Val Dice={val_dice:.4f} | Val IoU={val_iou:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                # 保存模型时直接用 eval_model (已经是解包过的状态)
                torch.save(eval_model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"New best model saved with Val Dice {val_dice:.4f}")

        # 使用 eval_model (单卡) 计算训练集指标，修复之前的崩溃点
        train_dice, train_iou = evaluate(eval_model, train_loader, device)
        print(f"Epoch {epoch}: Train Dice={train_dice:.4f} | Train IoU={train_iou:.4f}")

    # 最后保存也使用解包后的模型
    final_model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(final_model_to_save.state_dict(), os.path.join(output_dir, "final_model.pth"))

    if best_epoch is not None:
        print(f"Best validation model was achieved at epoch {best_epoch} with Dice {best_val_dice:.4f}")
    else:
        print("Validation was not run; no best epoch to report.")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive PRP area segmentation trainer")
    parser.add_argument("--train_dir", type=str, default="dataset_liekong/train", help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, default="dataset_liekong/val", help="Path to validation dataset")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--use_visdom", action="store_true", help="Enable Visdom visualization")
    parser.add_argument("--visdom_env", type=str, default="prp_segmentation", help="Visdom environment name")
    parser.add_argument("--visdom_port", type=int, default=8097, help="Visdom server port")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save validation outputs and models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        use_visdom=args.use_visdom,
        visdom_env=args.visdom_env,
        visdom_port=args.visdom_port,
        output_dir=args.output_dir,
    )
