import os
from glob import glob
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import BasicTransform


class PRPDataset(torch.utils.data.Dataset):
    """Dataset for semi-automatic PRP segmentation with a single target.

    Each case folder must contain:
        - image.png
        - gt_1.png

    A simulated click heatmap is generated from gt_1 and all images/masks are
    resized to ``target_size`` (default: 1280x1280). Spatial augmentations are
    applied consistently across the image, the mask, and the heatmap to keep
    alignment intact.
    """

    def __init__(
        self,
        root_dir: str,
        image_extensions: Optional[List[str]] = None,
        augment: bool = True,
        target_size: Tuple[int, int] = (1280, 1280),
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.augment = augment
        self.image_extensions = image_extensions or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        self.target_size = target_size
        self._last_click: Optional[Tuple[int, int]] = None

        self.cases = sorted([d for d in glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
        if not self.cases:
            raise ValueError(f"No case folders found in {root_dir}")

        self.samples: List[dict] = []
        for case_dir in self.cases:
            for required in ["image.png", "gt_1.png"]:
                if not os.path.exists(os.path.join(case_dir, required)):
                    raise FileNotFoundError(f"Missing {required} in {case_dir}")

            mask = self._load_mask(os.path.join(case_dir, "gt_1.png"))
            component_count = self._count_components(mask)
            self.samples.append(
                {
                    "case_dir": case_dir,
                    "component_count": component_count,
                    "repeat_idx": 0,
                }
            )

        self.spatial_transform = self._build_spatial_transform()
        self.color_transform = self._build_color_transform() if augment else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def _build_spatial_transform(self) -> A.Compose:
        transforms: list[BasicTransform] = []
        if self.augment:
            transforms.extend(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=10,
                        p=0.7,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=0,
                    ),
                    A.HorizontalFlip(p=0.5),
                ]
            )

        transforms.append(
            A.Resize(
                height=self.target_size[0],
                width=self.target_size[1],
                interpolation=cv2.INTER_LINEAR,
            )
        )

        return A.Compose(
            transforms,
            additional_targets={
                "mask1": "mask",
            },
        )

    def _build_color_transform(self) -> A.Compose:
        return A.Compose(
            [
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=8, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            ]
        )

    def _load_image(self, case_dir: str) -> np.ndarray:
        image_path = os.path.join(case_dir, "image.png")
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                return image

        # Fallback: try other known extensions but strictly prefer files named "image.*"
        for ext in self.image_extensions:
            candidate = os.path.join(case_dir, f"image{ext}")
            if os.path.exists(candidate):
                image = cv2.imread(candidate)
                if image is not None:
                    return image

        raise FileNotFoundError(
            f"No image found in {case_dir}. Expected image.png or image with extensions {self.image_extensions}"
        )

    def _load_mask(self, mask_path: str) -> np.ndarray:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        _, mask_bin = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        return mask_bin.astype(np.uint8)

    @staticmethod
    def _generate_heatmap(height: int, width: int, center: Tuple[int, int], sigma: float = 15.0) -> np.ndarray:
        y = np.arange(0, height, 1, float)
        x = np.arange(0, width, 1, float)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        heatmap = np.exp(-((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma ** 2))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap.astype(np.float32)

    @staticmethod
    def _count_components(mask: np.ndarray) -> int:
        num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        return max(num_labels - 1, 0)

    def _simulate_click(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
        full_mask = mask.copy()
        h, w = full_mask.shape
        if full_mask.max() == 0:
            self._last_click = None
            return full_mask, np.zeros((h, w), dtype=np.float32), None

        num_labels, labels = cv2.connectedComponents(full_mask, connectivity=8)
        valid_labels = [label for label in range(1, num_labels) if np.any(labels == label)]
        if not valid_labels:
            self._last_click = None
            return full_mask, np.zeros((h, w), dtype=np.float32), None

        selected_label = np.random.choice(valid_labels)
        component_mask = (labels == selected_label).astype(np.uint8)

        dist_map = cv2.distanceTransform(component_mask, distanceType=cv2.DIST_L2, maskSize=5)
        _, max_val, _, max_loc = cv2.minMaxLoc(dist_map)
        if max_val <= 0:
            self._last_click = None
            return full_mask, np.zeros((h, w), dtype=np.float32), None

        cy, cx = float(max_loc[1]), float(max_loc[0])
        ys, xs = np.nonzero(component_mask)
        if len(ys) == 0:
            self._last_click = None
            return full_mask, np.zeros((h, w), dtype=np.float32), None

        distances = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        weights = 1.0 / (distances + 1e-3)
        weights_sum = weights.sum()
        if weights_sum <= 0:
            self._last_click = None
            return full_mask, np.zeros((h, w), dtype=np.float32), None

        probabilities = weights / weights_sum
        idx = np.random.choice(len(probabilities), p=probabilities)
        click_y, click_x = int(ys[idx]), int(xs[idx])
        self._last_click = (click_y, click_x)
        heatmap = self._generate_heatmap(h, w, (click_y, click_x))
        return full_mask, heatmap, self._last_click

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        case_dir = sample["case_dir"]
        image = self._load_image(case_dir)
        mask1 = self._load_mask(os.path.join(case_dir, "gt_1.png"))

        augmented = self.spatial_transform(image=image, mask1=mask1)
        image = augmented["image"]
        mask1 = augmented["mask1"]

        if self.color_transform:
            image = self.color_transform(image=image)["image"]

        mask1, heatmap, _ = self._simulate_click(mask1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        tensor_transform = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ],
            additional_targets={
                "mask1": "mask",
                "heatmap": "mask",
            },
        )
        tensors = tensor_transform(image=image, mask1=mask1, heatmap=heatmap)
        image_tensor = tensors["image"]
        mask1_tensor = tensors["mask1"].unsqueeze(0).float()
        heatmap_tensor = tensors["heatmap"].float().unsqueeze(0)

        click_coords = torch.tensor(self._last_click or (-1, -1), dtype=torch.long)

        return image_tensor, heatmap_tensor, mask1_tensor, click_coords
