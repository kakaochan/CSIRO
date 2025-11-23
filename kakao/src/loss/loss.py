"""Loss function implementations for CSIRO pasture biomass prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLossWrapper(nn.Module):
    """Wrapper for standard BCE loss with pos_weight support."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes, n_timesteps)
            targets: (batch, n_classes, n_timesteps)
            pos_weight: (1, n_classes, 1) or None

        Returns:
            loss: scalar
        """
        if pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        else:
            loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

        return loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples.
                   Higher alpha gives more weight to positive examples.
            gamma: Focusing parameter >= 0. Higher gamma focuses more on hard examples.
                   gamma=0 is equivalent to standard BCE loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes, n_timesteps)
            targets: (batch, n_classes, n_timesteps)
            pos_weight: (1, n_classes, 1) or None - applied to alpha weighting

        Returns:
            loss: scalar
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute BCE loss (without reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute p_t (probability of the true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight
        if pos_weight is not None:
            # Use pos_weight to modulate alpha for positive examples
            # pos_weight shape: (1, n_classes, 1)
            alpha_t = pos_weight * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class CombinedLoss(nn.Module):
    """Combination of multiple losses with weighting."""

    def __init__(self, loss_types=['bce'], loss_weights=[1.0], focal_alpha=0.25, focal_gamma=2.0):
        """
        Args:
            loss_types: List of loss types to combine (e.g., ['bce', 'focal'])
            loss_weights: List of weights for each loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        self.loss_types = loss_types
        self.loss_weights = loss_weights

        self.losses = nn.ModuleDict()
        for loss_type in loss_types:
            if loss_type == 'bce':
                self.losses[loss_type] = BCEWithLogitsLossWrapper()
            elif loss_type == 'focal':
                self.losses[loss_type] = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes, n_timesteps)
            targets: (batch, n_classes, n_timesteps)
            pos_weight: (1, n_classes, 1) or None

        Returns:
            loss: scalar
        """
        total_loss = 0.0
        for loss_type, weight in zip(self.loss_types, self.loss_weights):
            loss = self.losses[loss_type](logits, targets, pos_weight)
            total_loss += weight * loss

        return total_loss


def get_loss_function(cfg):
    """Factory function to get loss function based on config.

    Args:
        cfg: Configuration object with model.loss settings

    Returns:
        loss_fn: Loss function module
    """
    loss_name = cfg.model.loss

    if loss_name == 'bce':
        return BCEWithLogitsLossWrapper()

    elif loss_name == 'focal':
        alpha = getattr(cfg.model, 'focal_alpha', 0.25)
        gamma = getattr(cfg.model, 'focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == 'combined':
        loss_types = getattr(cfg.model, 'loss_types', ['bce', 'focal'])
        loss_weights = getattr(cfg.model, 'loss_weights', [0.5, 0.5])
        focal_alpha = getattr(cfg.model, 'focal_alpha', 0.25)
        focal_gamma = getattr(cfg.model, 'focal_gamma', 2.0)
        return CombinedLoss(
            loss_types=loss_types,
            loss_weights=loss_weights,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
