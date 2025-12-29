from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.checkpoint import checkpoint
from torchvision import models


_heatmap_grid_cache: dict[tuple[int, int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_meshgrid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, device, dtype)
    if key not in _heatmap_grid_cache:
        y = torch.arange(height, device=device, dtype=dtype)
        x = torch.arange(width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        _heatmap_grid_cache[key] = (yy, xx)
    return _heatmap_grid_cache[key]


def build_gaussian_heatmap(
    clicks: torch.Tensor,
    in_hw: tuple[int, int],
    out_hw: tuple[int, int],
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    normalize_peak: bool = True,
) -> torch.Tensor:
    """Generate a per-sample Gaussian heatmap from click coordinates.

    Coordinates are mapped with align_corners=False convention using pixel centers.
    Sigma is defined in the pixel units of the *output* feature map (deep feature space).
    """
    batch = clicks.shape[0]
    h_out, w_out = out_hw
    heatmap = torch.zeros((batch, 1, h_out, w_out), device=device, dtype=dtype)
    yy, xx = _get_meshgrid(h_out, w_out, device, dtype)

    scale_y = h_out / float(in_hw[0])
    scale_x = w_out / float(in_hw[1])

    for b in range(batch):
        y_in, x_in = clicks[b].tolist()
        if y_in < 0 or x_in < 0:
            continue

        y_out = (float(y_in) + 0.5) * scale_y - 0.5
        x_out = (float(x_in) + 0.5) * scale_x - 0.5

        dist = (yy - y_out) ** 2 + (xx - x_out) ** 2
        hm = torch.exp(-dist / (2 * sigma**2))
        if normalize_peak:
            peak = hm.max()
            if peak > 0:
                hm = hm / peak
        heatmap[b, 0] = hm

    return heatmap


def build_heatmap4_5(
    clicks: torch.Tensor,
    in_hw: tuple[int, int],
    x4_hw: tuple[int, int],
    x5_hw: tuple[int, int],
    sigma4: float = 2.0,
    sigma5: float = 1.5,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper to build deep-scale heatmaps for x4 and x5 injection."""
    hm4 = build_gaussian_heatmap(clicks, in_hw, x4_hw, sigma4, dtype=dtype, device=device)
    hm5 = build_gaussian_heatmap(clicks, in_hw, x5_hw, sigma5, dtype=dtype, device=device)
    return hm4, hm5


class ViTFeatureRefiner(nn.Module):
    """Lightweight ViT-style encoder to model long-range dependencies."""

    def __init__(self, channels: int, num_layers: int = 1, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * mlp_ratio),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @staticmethod
    def _build_2d_sincos_position_embedding(height: int, width: int, channels: int, device, dtype):
        if channels % 4 != 0:
            raise ValueError("Channels for positional embedding must be divisible by 4")
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        omega = torch.arange(channels // 4, device=device, dtype=dtype) / (channels // 4)
        omega = 1.0 / (10000**omega)
        out_y = torch.einsum("hw,c->hwc", grid_y, omega)
        out_x = torch.einsum("hw,c->hwc", grid_x, omega)
        pos_emb = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
        pos_emb = pos_emb.reshape(1, height * width, channels)
        return pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        pos_emb = self._build_2d_sincos_position_embedding(h, w, c, x.device, x.dtype)
        tokens = tokens + pos_emb
        encoded = self.encoder(tokens)
        return encoded.transpose(1, 2).reshape(b, c, h, w)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, skip0_channels: int):
        super().__init__()
        self.up4 = UpBlock(encoder_channels[4], encoder_channels[3], encoder_channels[3])
        self.up3 = UpBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.up2 = UpBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.up1 = UpBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])
        self.up0 = UpBlock(encoder_channels[0], skip0_channels, encoder_channels[0])
        self.final_conv = nn.Conv2d(encoder_channels[0], 1, kernel_size=1)

    def forward(self, features, skip0: torch.Tensor):
        c1, c2, c3, c4, c5 = features
        x = self.up4(c5, c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        x = self.up0(x, skip0)
        return self.final_conv(x)


class PRPSegmenter(nn.Module):
    def __init__(self, pretrained: bool = True, prompt_dropout_p: float = 0.2, use_ckpt: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.stem_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Heatmap stems for deep prompt injection
        self.hm4_stem = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.hm5_stem = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Concat + 1x1 mix for x4/x5
        self.mix4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.mix5 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Gating parameters initialized to negative values for conservative blending
        self.g4_raw = nn.Parameter(torch.full((1, 256, 1, 1), -2.0))
        self.g5_raw = nn.Parameter(torch.full((1, 512, 1, 1), -2.0))

        # Prompt-conditioned ViT bottleneck on x5
        self.proj_in = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.vit_refiner_256 = ViTFeatureRefiner(channels=256, num_layers=1, num_heads=4, mlp_ratio=4.0)
        self.proj_out = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.decoder = UNetDecoder(encoder_channels=[64, 64, 128, 256, 512], skip0_channels=64)
        self.prompt_dropout_p = prompt_dropout_p
        self.use_ckpt = use_ckpt

    @staticmethod
    @contextmanager
    def _maybe_restore_bn_buffers(module: nn.Module, enable: bool):
        """
        For non-reentrant checkpointing: keep BN forward/recompute behavior identical
        (to satisfy checkpoint consistency checks), but restore BN running buffers after
        recomputation so running stats are not double-counted.
        """
        if not enable:
            yield
            return

        bn_layers = [
            m
            for m in module.modules()
            if isinstance(m, _BatchNorm) and getattr(m, "track_running_stats", False)
        ]
        if not bn_layers:
            yield
            return

        backups: list[tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]] = []
        for bn in bn_layers:
            rm = getattr(bn, "running_mean", None)
            rv = getattr(bn, "running_var", None)
            nbt = getattr(bn, "num_batches_tracked", None)

            rm_b = None if rm is None else rm.detach().clone()
            rv_b = None if rv is None else rv.detach().clone()
            nbt_b = None if nbt is None else nbt.detach().clone()
            backups.append((rm_b, rv_b, nbt_b))

        try:
            yield
        finally:
            for bn, (rm_b, rv_b, nbt_b) in zip(bn_layers, backups):
                if rm_b is not None and getattr(bn, "running_mean", None) is not None:
                    bn.running_mean.data.copy_(rm_b)
                if rv_b is not None and getattr(bn, "running_var", None) is not None:
                    bn.running_var.data.copy_(rv_b)
                if nbt_b is not None and getattr(bn, "num_batches_tracked", None) is not None:
                    bn.num_batches_tracked.data.copy_(nbt_b)

    def _maybe_ckpt_module(self, module: nn.Module, *args):
        """
        Checkpoint a module. If it contains BatchNorm with running-stat tracking,
        prevent double-counting by restoring BN buffers after recomputation.
        """
        if self.training and self.use_ckpt:
            has_bn = any(
                isinstance(m, _BatchNorm) and getattr(m, "track_running_stats", False)
                for m in module.modules()
            )

            if has_bn:
                def context_fn():
                    # forward: normal
                    # recompute: normal, but restore BN buffers afterward (no net update)
                    return nullcontext(), self._maybe_restore_bn_buffers(module, True)

                return checkpoint(module, *args, use_reentrant=False, context_fn=context_fn)

            return checkpoint(module, *args, use_reentrant=False)

        return module(*args)

    def _apply_prompt_dropout_pair(self, hm4: torch.Tensor, hm5: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.prompt_dropout_p > 0:
            keep_mask = (
                torch.rand((hm4.shape[0], 1, 1, 1), device=hm4.device, dtype=hm4.dtype)
                > self.prompt_dropout_p
            ).to(hm4.dtype)
            hm4 = hm4 * keep_mask
            hm5 = hm5 * keep_mask
        return hm4, hm5

    def forward(self, image: torch.Tensor, clicks: torch.Tensor) -> torch.Tensor:
        x1_img = self.relu(self.bn1(self.conv1(image)))  # (B, 64, H/2, W/2)
        x2 = self._maybe_ckpt_module(self.layer1, self.maxpool(x1_img))  # (B, 64, H/4, W/4)
        x3 = self._maybe_ckpt_module(self.layer2, x2)  # (B, 128, H/8, W/8)
        x4_img = self._maybe_ckpt_module(self.layer3, x3)  # (B, 256, H/16, W/16)
        x5_img = self._maybe_ckpt_module(self.layer4, x4_img)  # (B, 512, H/32, W/32)

        hm4, hm5 = build_heatmap4_5(
            clicks=clicks,
            in_hw=(image.shape[2], image.shape[3]),
            x4_hw=(x4_img.shape[2], x4_img.shape[3]),
            x5_hw=(x5_img.shape[2], x5_img.shape[3]),
            dtype=x4_img.dtype,
            device=x4_img.device,
        )

        hm4, hm5 = self._apply_prompt_dropout_pair(hm4, hm5)

        hm4_feat = self.hm4_stem(hm4)
        hm5_feat = self.hm5_stem(hm5)

        x4_cond = self.mix4(torch.cat([x4_img, hm4_feat], dim=1))
        g4 = torch.sigmoid(self.g4_raw)
        x4_out = x4_img + g4 * (x4_cond - x4_img)

        x5_cond = self.mix5(torch.cat([x5_img, hm5_feat], dim=1))
        g5 = torch.sigmoid(self.g5_raw)
        x5_blend = x5_img + g5 * (x5_cond - x5_img)

        x5_256 = self._maybe_ckpt_module(self.proj_in, x5_blend)
        x5_256 = self._maybe_ckpt_module(self.vit_refiner_256, x5_256)
        x5_out = self._maybe_ckpt_module(self.proj_out, x5_256)

        x0_img = self.stem_conv(image)  # (B, 64, H, W)
        logits = self.decoder([x1_img, x2, x3, x4_out, x5_out], skip0=x0_img)
        return logits
