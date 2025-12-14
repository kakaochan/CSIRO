"""Lightning-free model for Kaggle inference (no Lightning dependency)."""

import torch
import torch.nn as nn
from src.models.common import get_model


class CSIROInferenceModel(nn.Module):
    """Inference-only model without Lightning dependency.

    This is a lightweight wrapper around the core model architecture
    for use in Kaggle environments where Lightning cannot be installed.
    """

    def __init__(self, cfg):
        """Initialize inference model.

        Args:
            cfg: Configuration object with model architecture settings
        """
        super().__init__()

        # Create the core model (Spec1D with feature_extractor + decoder)
        self.net = get_model(
            cfg,
            feature_dim=3,
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

    def load_state_dict_from_checkpoint(self, checkpoint_path, device='cpu'):
        """Load weights from a checkpoint file.

        Args:
            checkpoint_path: Path to .pth file
            device: Device to load weights to

        Returns:
            self (for chaining)
        """
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Remove 'net.' prefix if present (from Lightning checkpoints)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.'):
                cleaned_state_dict[key[4:]] = value  # Remove 'net.' prefix
            else:
                cleaned_state_dict[key] = value

        self.net.load_state_dict(cleaned_state_dict, strict=False)
        return self
