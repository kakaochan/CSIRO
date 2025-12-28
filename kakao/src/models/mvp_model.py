"""MVP Model: TiledFiLMDINO from top90 ensemble notebook.

This model uses:
- DINO backbone (vit_base_patch14_reg4_dinov2 or similar)
- 2x2 grid tiling for multi-scale features
- FiLM (Feature-wise Linear Modulation) for context-aware feature modulation
- Two-stream processing (left/right image halves)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer.

    Generates gamma and beta parameters from context to modulate features.
    """

    def __init__(self, feat_dim):
        super().__init__()
        hidden = max(32, feat_dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim * 2)
        )

    def forward(self, context):
        """
        Args:
            context: (B, feat_dim) context vector
        Returns:
            gamma: (B, feat_dim) scaling parameters
            beta: (B, feat_dim) shift parameters
        """
        gamma_beta = self.mlp(context)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return gamma, beta


class BaseDINO(nn.Module):
    """Base class for DINO-based models."""

    def __init__(self, backbone_name):
        super().__init__()
        self.dropout = 0.30
        self.hidden_ratio = 0.25
        self.grid = (2, 2)  # 2x2 tiling
        self.backbone_name = backbone_name

        # Create DINO backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0
        )

        self.feat_dim = self.backbone.num_features
        self.input_size = self._get_input_size(self.backbone)
        self.combined_dim = self.feat_dim * 2  # left + right features

        hidden_size = max(8, int(self.combined_dim * self.hidden_ratio))

        def make_head():
            return nn.Sequential(
                nn.Linear(self.combined_dim, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_size, 1),
            )

        # Three prediction heads
        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
        self.softplus = nn.Softplus(beta=1.0)

    def _get_input_size(self, model):
        """Infer input size from model config."""
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
            size = model.patch_embed.img_size
            return int(size if isinstance(size, (int, float)) else size[0])

        if hasattr(model, "img_size"):
            size = model.img_size
            return int(size if isinstance(size, (int, float)) else size[0])

        cfg = getattr(model, "default_cfg", {}) or {}
        input_size = cfg.get("input_size", None)

        if input_size:
            if isinstance(input_size, (tuple, list)) and len(input_size) >= 2:
                return int(input_size[1])
            return int(input_size if isinstance(input_size, (int, float)) else 224)

        # Default for DINOv2/v3
        arch = cfg.get("architecture", "") or str(type(model))
        return 518 if "dinov2" in arch.lower() or "dinov3" in arch.lower() else 224

    def merge_features(self, left_feat, right_feat):
        """Merge left and right features and predict targets.

        Args:
            left_feat: (B, feat_dim) features from left stream
            right_feat: (B, feat_dim) features from right stream

        Returns:
            total: (B, 1) Dry_Total_g prediction
            gdm: (B, 1) GDM_g prediction
            green: (B, 1) Dry_Green_g prediction
        """
        combined = torch.cat([left_feat, right_feat], dim=1)
        green = self.softplus(self.head_green(combined))
        clover = self.softplus(self.head_clover(combined))
        dead = self.softplus(self.head_dead(combined))
        gdm = green + clover
        total = gdm + dead
        return total, gdm, green


class TiledFiLMDINO(BaseDINO):
    """Tiled DINO with FiLM modulation for two-stream biomass prediction.

    Architecture:
    1. Split input into 2x2 grid of tiles
    2. Extract features from each tile with DINO backbone
    3. Apply FiLM modulation based on global context
    4. Merge left/right stream features
    5. Predict 3 targets (green, clover, dead) and calculate 2 derived targets
    """

    def __init__(self, backbone_name="vit_base_patch14_reg4_dinov2"):
        super().__init__(backbone_name)
        # FiLM layers for left and right streams
        self.film_left = FiLM(self.feat_dim)
        self.film_right = FiLM(self.feat_dim)

    def _split_dimension(self, length, parts):
        """Split a dimension into equal parts.

        Args:
            length: Total length to split
            parts: Number of parts

        Returns:
            List of (start, end) tuples
        """
        step = length // parts
        segments = []
        start = 0

        for _ in range(parts - 1):
            segments.append((start, start + step))
            start += step

        segments.append((start, length))
        return segments

    def _extract_tile_features(self, x):
        """Extract features from each tile in the grid.

        Args:
            x: (B, C, H, W) input image

        Returns:
            features: (B, num_tiles, feat_dim) tile features
        """
        B, C, H, W = x.shape
        rows, cols = self.grid
        row_segments = self._split_dimension(H, rows)
        col_segments = self._split_dimension(W, cols)
        features = []

        for (rs, re) in row_segments:
            for (cs, ce) in col_segments:
                tile = x[:, :, rs:re, cs:ce]
                # Resize tile to backbone input size
                if tile.shape[-2:] != (self.input_size, self.input_size):
                    tile = F.interpolate(
                        tile,
                        size=(self.input_size, self.input_size),
                        mode="bilinear",
                        align_corners=False
                    )
                feat = self.backbone(tile)
                features.append(feat)

        # Stack: (num_tiles, B, feat_dim) -> (B, num_tiles, feat_dim)
        return torch.stack(features, dim=0).permute(1, 0, 2)

    def _process_stream(self, x, film_layer):
        """Process one stream (left or right) with tiling and FiLM.

        Args:
            x: (B, C, H, W) input image
            film_layer: FiLM layer for this stream

        Returns:
            features: (B, feat_dim) aggregated features
        """
        # Extract tile features: (B, num_tiles, feat_dim)
        tiles = self._extract_tile_features(x)

        # Global context from average of all tiles
        context = tiles.mean(dim=1)  # (B, feat_dim)

        # FiLM modulation
        gamma, beta = film_layer(context)
        modulated = tiles * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # Aggregate modulated tile features
        return modulated.mean(dim=1)

    def forward(self, left_img, right_img):
        """Forward pass for two-stream input.

        Args:
            left_img: (B, C, H, W) left half of image
            right_img: (B, C, H, W) right half of image

        Returns:
            Dict with keys: 'logits', 'total', 'gdm', 'green'
            Each value is (B, 1) or (B, 5) tensor
        """
        left_feat = self._process_stream(left_img, self.film_left)
        right_feat = self._process_stream(right_img, self.film_right)
        total, gdm, green = self.merge_features(left_feat, right_feat)

        # Calculate derived targets
        clover = torch.clamp(gdm - green, min=0.0)
        dead = torch.clamp(total - gdm, min=0.0)

        # Pack into 5 targets in standard order:
        # [Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g]
        logits = torch.cat([green, dead, clover, gdm, total], dim=1)

        return {
            'logits': logits,
            'total': total,
            'gdm': gdm,
            'green': green,
        }
