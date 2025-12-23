from typing import Optional

import matplotlib.pyplot as plt
import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    return (2 * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    return (intersection + eps) / (union + eps)


def plot_sample(image: torch.Tensor, heatmap: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor, save_path: Optional[str] = None):
    """
    Plot input image, heatmap, prediction and ground truth mask for quick inspection.
    Expects tensors in CHW format.
    """
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    heatmap_np = heatmap.detach().cpu().squeeze().numpy()
    pred_np = pred.detach().cpu().squeeze().numpy()
    mask_np = mask.detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[1].imshow(heatmap_np, cmap="hot")
    axes[1].set_title("Simulated Click Heatmap")
    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Prediction")
    axes[3].imshow(mask_np, cmap="gray")
    axes[3].set_title("Ground Truth")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
