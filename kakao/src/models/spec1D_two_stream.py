from typing import Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn


class Spec1DTwoStream(nn.Module):
    """Two-Stream architecture following REFERENCE implementation.

    Processes left and right image patches separately through a shared backbone,
    then combines features before passing to decoder.
    """

    def __init__(
        self,
        cfg: DictConfig,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder

    def forward(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with two image streams.

        Args:
            img_left: Left image patch (B, 3, 768, 768)
            img_right: Right image patch (B, 3, 768, 768)
            labels: Optional labels
            masks: Optional masks
            do_mixup: Unused (for API compatibility)
            do_cutmix: Unused (for API compatibility)

        Returns:
            dict with 'logits': (B, n_classes)
        """
        # Process left and right patches through shared feature extractor
        features_left = self.feature_extractor(img_left)    # (B, 1, n_features, 1)
        features_right = self.feature_extractor(img_right)  # (B, 1, n_features, 1)

        # Combine features by concatenation (doubling the channel dimension)
        # features_left/right: (B, 1, n_features, 1)
        # Squeeze and concatenate: (B, n_features) each
        features_left = features_left.squeeze(1).squeeze(-1)   # (B, n_features)
        features_right = features_right.squeeze(1).squeeze(-1) # (B, n_features)
        combined_features = torch.cat([features_left, features_right], dim=1)  # (B, n_features*2)

        # Decoder processes combined features
        # ThreeTargetDecoder expects (B, n_channels)
        outputs = self.decoder(combined_features)  # returns {'logits': (B, 5)}
        logits = outputs['logits']  # (B, 5)

        return {"logits": logits}
