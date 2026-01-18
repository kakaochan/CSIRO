"""Lightning-free model for Kaggle inference (no Lightning dependency)."""

import torch
import torch.nn as nn
from src.models.common import get_model


class CSIROInferenceModel(nn.Module):
    """Inference-only model without Lightning dependency.

    This is a lightweight wrapper around the core model architecture
    for use in Kaggle environments where Lightning cannot be installed.
    """

    def __init__(self, cfg, target_mean=None, target_std=None):
        """Initialize inference model.

        Args:
            cfg: Configuration object with model architecture settings
            target_mean: Mean values for target normalization (from scaler)
            target_std: Std values for target normalization (from scaler)
        """
        super().__init__()

        # Create the core model (Spec1D with feature_extractor + decoder)
        self.net = get_model(
            cfg,
            feature_dim=3,
            target_mean=target_mean,
            target_std=target_std,
        )

    def forward(self, *args, **kwargs):
        """Forward pass supporting both Original and Two-Stream.

        Args:
            *args: Either (x,) for Original or (img_left, img_right) for Two-Stream
            **kwargs: Optional keyword arguments

        Returns:
            dict with 'logits': (B, 5) predictions
        """
        return self.net(*args, **kwargs)
