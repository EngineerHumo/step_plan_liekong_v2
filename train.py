'''
train:python /data/jiaqi/step_plan_liekong_v2/train.py --use_visdom
'''

import argparse
import logging
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

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def default_device() -> str:
    """Prefer CUDA when available, defaulting to GPU 0 for multi-GPU training."""
    if torch.cuda.is_available():
        return "cuda:0"              ###同步修改
    return "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tensor_to_image(tensor: torch.Tensor) -> "np.ndarray":  # type: ignore[name-defined]
    import numpy as np

    tensor = tensor.detach().cpu()
    if tensor.shape[0] == 3:
        tensor = tensor * IMAGENET_STD + IMAGENET_MEAN
    array = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).astype(np.uint8)


def save_validation_batch(
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    masks1: torch.Tensor,
    preds1: torch.Tensor,
    gt_sdf: Optional[torch.Tensor],
    pred_sdf: Optional[torch.Tensor],
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
        pred_sdf_np = pred_sdf[i, 0].detach().cpu().numpy() if pred_sdf is not None else None
        gt_sdf_np = gt_sdf[i, 0].detach().cpu().numpy() if gt_sdf is not None else None

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
        if pred_sdf_np is not None:
            pred_sdf_resized = cv2.resize(pred_sdf_np, (original_size[1], original_size[0]))
        else:
            pred_sdf_resized = None
        if gt_sdf_np is not None:
            gt_sdf_resized = cv2.resize(gt_sdf_np, (original_size[1], original_size[0]))
        else:
            gt_sdf_resized = None

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
        if gt_sdf_resized is not None:
            gt_sdf_vis = ((gt_sdf_resized + 1.0) * 0.5 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(epoch_dir, f"{basename}_gt_sdf.png"), gt_sdf_vis)
        if pred_sdf_resized is not None:
            pred_sdf_vis = ((pred_sdf_resized + 1.0) * 0.5 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(epoch_dir, f"{basename}_pred_sdf.png"), pred_sdf_vis)


def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dice = dice_coefficient(probs, target).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    return (1 - dice) + 0.5 * bce


def click_point_loss(logits: torch.Tensor, clicks: torch.Tensor) -> torch.Tensor:
    losses = []
    _, _, h, w = logits.shape
    for b in range(clicks.shape[0]):
        y, x = clicks[b].tolist()
        if y < 0 or x < 0:
            continue
        y = int(min(max(y, 0), h - 1))
        x = int(min(max(x, 0), w - 1))
        logit = logits[b, 0, y, x]
        losses.append(nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(1.0, device=logits.device)))

    if not losses:
        return torch.tensor(0.0, device=logits.device)
    return torch.stack(losses).mean()


def _prepare_image_for_visdom(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu()
    if image.shape[0] == 3:
        image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    return image


def _log_images_to_visdom(
    viz: "visdom.Visdom",  # type: ignore[name-defined]
    images: torch.Tensor,
    heatmaps: torch.Tensor,
    masks1: torch.Tensor,
    preds1: torch.Tensor,
    gt_sdf: Optional[torch.Tensor],
    pred_sdf: Optional[torch.Tensor],
    clicks: torch.Tensor,
    epoch: int,
    prefix: str = "val",
):
    def _prep_single(t: torch.Tensor) -> torch.Tensor:
        tensor = t.detach().cpu().clamp(0, 1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor

    img = _prepare_image_for_visdom(images[0])
    img_for_click = img.clone()
    click_y, click_x = clicks[0].tolist()
    if click_y >= 0 and click_x >= 0:
        y = min(click_y, img_for_click.shape[1] - 1)
        x = min(click_x, img_for_click.shape[2] - 1)
        img_for_click[:, y : y + 1, x : x + 1] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    heatmap = _prep_single(heatmaps[0])
    gt1 = _prep_single(masks1[0])
    pred1 = _prep_single(preds1[0])
    sdf_gt_img = _prep_single((gt_sdf[0] + 1.0) * 0.5) if gt_sdf is not None else None
    sdf_pred_img = _prep_single((pred_sdf[0] + 1.0) * 0.5) if pred_sdf is not None else None

    viz.image(img, win=f"{prefix}_input_image", opts={"title": f"{prefix.capitalize()} Input Epoch {epoch}"})
    viz.image(img_for_click, win=f"{prefix}_click", opts={"title": f"{prefix.capitalize()} Click Epoch {epoch}"})
    viz.image(heatmap, win=f"{prefix}_heatmap", opts={"title": f"{prefix.capitalize()} Heatmap Epoch {epoch}"})
    viz.image(gt1, win=f"{prefix}_ground_truth_1", opts={"title": f"{prefix.capitalize()} GT1 Epoch {epoch}"})
    viz.image(pred1, win=f"{prefix}_prediction_1", opts={"title": f"{prefix.capitalize()} Pred1 Epoch {epoch}"})
    if sdf_gt_img is not None:
        viz.image(sdf_gt_img, win=f"{prefix}_gt_sdf", opts={"title": f"{prefix.capitalize()} GT SDF Epoch {epoch}"})
    if sdf_pred_img is not None:
        viz.image(sdf_pred_img, win=f"{prefix}_pred_sdf", opts={"title": f"{prefix.capitalize()} Pred SDF Epoch {epoch}"})


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_root: Optional[str] = None,
    epoch: Optional[int] = None,
    viz: Optional["visdom.Visdom"] = None,  # type: ignore[name-defined]
) -> tuple[float, float]:
    model.eval()
    dice_scores = []
    iou_scores = []
    with torch.no_grad():
        for batch_idx, (images, heatmaps, masks1, clicks, gt_sdf) in enumerate(loader):
            images = images.to(device)
            masks1 = masks1.to(device)
            clicks = clicks.to(device)
            gt_sdf = gt_sdf.to(device)
            logits1, pred_sdf = model(images, clicks)
            preds1 = torch.sigmoid(logits1)
            dice_scores.append(dice_coefficient(preds1, masks1).mean().item())
            iou_scores.append(iou_score(preds1, masks1).mean().item())

            if save_root and epoch is not None:
                save_validation_batch(
                    images=images,
                    heatmaps=heatmaps,
                    masks1=masks1,
                    preds1=preds1,
                    gt_sdf=gt_sdf,
                    pred_sdf=pred_sdf,
                    save_root=save_root,
                    epoch=epoch,
                    batch_idx=batch_idx,
                )
            if viz is not None and batch_idx == 0:
                _log_images_to_visdom(
                    viz, images, heatmaps, masks1, preds1, gt_sdf, pred_sdf, clicks, epoch, prefix="val"
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
    sdf_loss_weight: float = 0.2,
):
    device = torch.device(device)
    ensure_dir(output_dir)
    log_file = os.path.join(output_dir, "train_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )
    logging.info("Using device: %s", device)

    train_dataset = PRPDataset(train_dir, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = PRPDataset(val_dir, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    model = PRPSegmenter()
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print("Using DataParallel on GPUs: 0 and 1")
        model = nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    viz = None
    if use_visdom:
        import visdom

        viz = visdom.Visdom(env=visdom_env, port=visdom_port)
        if not viz.check_connection():
            logging.warning("[Visdom] Connection failed. Visualizations will be skipped.")
            viz = None

    best_val_dice = float("-inf")
    best_epoch: Optional[int] = None
    best_epochs: list[tuple[int, float]] = []
    saved_model_paths: dict[int, str] = {}

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for images, heatmaps, masks1, clicks, gt_sdf in progress:
            images = images.to(device)
            masks1 = masks1.to(device)
            clicks = clicks.to(device)

            gt_sdf = gt_sdf.to(device)

            logits1, pred_sdf = model(images, clicks)
            main_loss = dice_bce_loss(logits1, masks1)
            point_loss = click_point_loss(logits1, clicks)
            sdf_loss = nn.functional.smooth_l1_loss(pred_sdf, gt_sdf)
            loss = main_loss + 0.01 * point_loss + sdf_loss_weight * sdf_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item(), point_loss=point_loss.item(), sdf_loss=sdf_loss.item())

            if viz is not None:
                preds1 = torch.sigmoid(logits1)
                _log_images_to_visdom(
                    viz, images, heatmaps, masks1, preds1, gt_sdf, pred_sdf, clicks, epoch, prefix="train"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logging.info("Epoch %d: Train Loss=%.4f", epoch, avg_loss)

        # ==============================================================================
        # 核心修改：统一提取单卡模型用于验证和评估
        # 这确保了无论是在验证集还是在训练集上跑 evaluate，都不会因为 Batch 切分问题
        # 导致其中一张卡分到空数据而崩溃。
        # ==============================================================================
        eval_model = model.module if isinstance(model, nn.DataParallel) else model

        if val_loader:
            val_save_dir = os.path.join(output_dir, "val_outputs")

            # 使用 eval_model (单卡) 进行验证
            val_dice, val_iou = evaluate(eval_model, val_loader, device, save_root=val_save_dir, epoch=epoch, viz=viz)
            logging.info("Epoch %d: Val Dice=%.4f | Val IoU=%.4f", epoch, val_dice, val_iou)

            best_epochs.append((epoch, val_dice))
            best_epochs = sorted(best_epochs, key=lambda x: x[1], reverse=True)[:3]

            current_top_epochs = {e for e, _ in best_epochs}
            for saved_epoch, path in list(saved_model_paths.items()):
                if saved_epoch not in current_top_epochs and os.path.exists(path):
                    os.remove(path)
                    del saved_model_paths[saved_epoch]

            for e, d in best_epochs:
                if e not in saved_model_paths:
                    model_path = os.path.join(output_dir, f"best_epoch_{e:03d}_dice_{d:.4f}.pth")
                    torch.save(eval_model.state_dict(), model_path)
                    saved_model_paths[e] = model_path

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_epoch = epoch
                logging.info("New best model tracked with Val Dice %.4f", val_dice)

        # 使用 eval_model (单卡) 计算训练集指标，修复之前的崩溃点
        train_dice, train_iou = evaluate(eval_model, train_loader, device)
        logging.info("Epoch %d: Train Dice=%.4f | Train IoU=%.4f", epoch, train_dice, train_iou)

    if best_epochs:
        logging.info("Top 3 validation epochs (epoch, dice): %s", best_epochs)

    if best_epoch is not None:
        logging.info("Best validation model was achieved at epoch %d with Dice %.4f", best_epoch, best_val_dice)
    else:
        logging.info("Validation was not run; no best epoch to report.")


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive PRP area segmentation trainer")
    parser.add_argument("--train_dir", type=str, default="dataset_liekong/train", help="Path to training dataset")
    parser.add_argument("--val_dir", type=str, default="dataset_liekong/val", help="Path to validation dataset")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--use_visdom", action="store_true", help="Enable Visdom visualization")
    parser.add_argument("--visdom_env", type=str, default="prp_segmentation", help="Visdom environment name")
    parser.add_argument("--visdom_port", type=int, default=8097, help="Visdom server port")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save validation outputs and models")
    parser.add_argument("--sdf_loss_weight", type=float, default=0.2, help="Weight for the SDF regression loss")
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
        sdf_loss_weight=args.sdf_loss_weight,
    )
