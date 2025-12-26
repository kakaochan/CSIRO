"""Three-Target Decoder with formula-based calculation for remaining targets.

This decoder predicts only 3 key targets (Dry_Total_g, GDM_g, Dry_Green_g)
and calculates the remaining 2 targets (Dry_Clover_g, Dry_Dead_g) using formulas:
- Dry_Clover_g = max(0, GDM_g - Dry_Green_g)
- Dry_Dead_g = max(0, Dry_Total_g - GDM_g)

This approach:
1. Reduces redundancy (targets have linear dependencies)
2. Ensures mathematical consistency (no contradictions)
3. Focuses learning on the most important targets (80% of score weight)
"""

import torch
import torch.nn as nn


class ThreeTargetDecoder(nn.Module):
    """Decoder that predicts 3 targets and calculates 2 remaining targets.

    Predicted targets:
    - Dry_Total_g (weight: 0.5 = 50% of score)
    - GDM_g (weight: 0.2 = 20% of score)
    - Dry_Green_g (weight: 0.1 = 10% of score)

    Calculated targets:
    - Dry_Clover_g = max(0, GDM_g - Dry_Green_g)
    - Dry_Dead_g = max(0, Dry_Total_g - GDM_g)
    """

    def __init__(self, cfg, n_channels, n_pairs=None, n_actions=None, num_timesteps=None):
        """Initialize Three-Target Decoder.

        Args:
            cfg: Configuration object
            n_channels: Number of input channels from feature extractor
            n_pairs: Unused (for compatibility with get_decoder)
            n_actions: Unused (for compatibility with get_decoder)
            num_timesteps: Unused (for compatibility with get_decoder)
        """
        super().__init__()

        dropout = cfg.decoder.dropout
        hidden_dim = n_channels // 2

        # Head for Dry_Total_g (most important: 50% weight)
        self.head_total = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Head for GDM_g (20% weight)
        self.head_gdm = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # Head for Dry_Green_g (10% weight)
        self.head_green = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input features (B, n_channels) from feature extractor

        Returns:
            (B, 5) tensor containing all 5 target predictions
                Order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g]
                (matches train.csv order for compatibility with existing dataset)
        """
        # Predict 3 key targets
        pred_total = self.head_total(x)   # (B, 1)
        pred_gdm = self.head_gdm(x)       # (B, 1)
        pred_green = self.head_green(x)   # (B, 1)

        # Calculate remaining 2 targets using formulas
        # Clamp to 0 to avoid negative predictions
        pred_clover = torch.clamp(pred_gdm - pred_green, min=0.0)  # (B, 1)
        pred_dead = torch.clamp(pred_total - pred_gdm, min=0.0)    # (B, 1)

        # Concatenate in train.csv order: [Clover, Dead, Green, Total, GDM]
        logits = torch.cat([
            pred_clover,  # Dry_Clover_g (calculated)
            pred_dead,    # Dry_Dead_g (calculated)
            pred_green,   # Dry_Green_g (predicted)
            pred_total,   # Dry_Total_g (predicted)
            pred_gdm      # GDM_g (predicted)
        ], dim=1)  # (B, 5)

        return logits  # (B, 5)
