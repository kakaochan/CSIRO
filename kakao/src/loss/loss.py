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


class MSELossWrapper(nn.Module):
    """Standard Mean Squared Error loss for regression tasks."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes) - predicted values
            targets: (batch, n_classes) - ground truth values
            pos_weight: Ignored (for API compatibility)

        Returns:
            loss: scalar
        """
        return self.loss_fn(logits, targets)


class L1LossWrapper(nn.Module):
    """Mean Absolute Error (L1) loss for regression tasks."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='mean')

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes) - predicted values
            targets: (batch, n_classes) - ground truth values
            pos_weight: Ignored (for API compatibility)

        Returns:
            loss: scalar
        """
        return self.loss_fn(logits, targets)


class SmoothL1LossWrapper(nn.Module):
    """Huber loss (SmoothL1Loss) for regression tasks.

    Less sensitive to outliers than MSE.
    Used in the reference implementation (lb-0-57).
    """

    def __init__(self, beta=1.0):
        """
        Args:
            beta: The threshold at which to change between L1 and L2 loss.
                  Default is 1.0 (standard Huber loss).
        """
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=beta)

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes) - predicted values
            targets: (batch, n_classes) - ground truth values
            pos_weight: Ignored (for API compatibility)

        Returns:
            loss: scalar
        """
        return self.loss_fn(logits, targets)


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss matching competition evaluation metric.

    Applies per-target weights matching the competition's R² weights:
    [0.1, 0.1, 0.1, 0.2, 0.5] for [Dry_Green, Dry_Dead, Dry_Clover, GDM, Dry_Total]
    """

    def __init__(self, target_weights=None):
        """
        Args:
            target_weights: List or tensor of weights for each target.
                           Default: [0.1, 0.1, 0.1, 0.2, 0.5] (competition weights)
        """
        super().__init__()
        if target_weights is None:
            # Competition weights: [Dry_Green, Dry_Dead, Dry_Clover, GDM, Dry_Total]
            target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
        self.register_buffer('target_weights', torch.tensor(target_weights, dtype=torch.float32))

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes) - predicted values
            targets: (batch, n_classes) - ground truth values
            pos_weight: Ignored (for API compatibility)

        Returns:
            loss: scalar
        """
        # Compute squared errors: (batch, n_classes)
        squared_errors = (logits - targets) ** 2

        # Apply per-target weights: (n_classes,)
        weighted_errors = squared_errors * self.target_weights.unsqueeze(0)

        # Return mean across all elements
        return weighted_errors.mean()


class WeightedSmoothL1Loss(nn.Module):
    """Weighted SmoothL1 (Huber) loss matching competition evaluation metric.

    Combines the robustness of Huber loss with target weighting.
    Based on reference implementation strategy.
    """

    def __init__(self, target_weights=None, beta=1.0):
        """
        Args:
            target_weights: List or tensor of weights for each target.
                           Default: [0.1, 0.1, 0.1, 0.2, 0.5] (competition weights)
            beta: The threshold at which to change between L1 and L2 loss.
        """
        super().__init__()
        if target_weights is None:
            # Competition weights: [Dry_Green, Dry_Dead, Dry_Clover, GDM, Dry_Total]
            target_weights = [0.1, 0.1, 0.1, 0.2, 0.5]
        self.register_buffer('target_weights', torch.tensor(target_weights, dtype=torch.float32))
        self.beta = beta

    def forward(self, logits, targets, pos_weight=None):
        """
        Args:
            logits: (batch, n_classes) - predicted values
            targets: (batch, n_classes) - ground truth values
            pos_weight: Ignored (for API compatibility)

        Returns:
            loss: scalar
        """
        # Compute smooth L1 loss per element (no reduction): (batch, n_classes)
        diff = torch.abs(logits - targets)
        loss_per_element = torch.where(
            diff < self.beta,
            0.5 * (diff ** 2) / self.beta,
            diff - 0.5 * self.beta
        )

        # Apply per-target weights: (n_classes,)
        weighted_loss = loss_per_element * self.target_weights.unsqueeze(0)

        # Return mean across all elements
        return weighted_loss.mean()


def get_loss_function(cfg):
    """Factory function to get loss function based on config.

    Args:
        cfg: Configuration object with model.loss settings

    Returns:
        loss_fn: Loss function module
    """
    loss_name = cfg.model.loss

    # --- Regression Losses ---
    if loss_name == 'mse':
        return MSELossWrapper()

    elif loss_name == 'l1' or loss_name == 'mae':
        return L1LossWrapper()

    elif loss_name == 'smoothl1' or loss_name == 'huber':
        beta = getattr(cfg.model, 'smoothl1_beta', 1.0)
        return SmoothL1LossWrapper(beta=beta)

    elif loss_name == 'weighted_mse':
        target_weights = getattr(cfg.model, 'target_weights', None)
        return WeightedMSELoss(target_weights=target_weights)

    elif loss_name == 'weighted_smoothl1':
        target_weights = getattr(cfg.model, 'target_weights', None)
        beta = getattr(cfg.model, 'smoothl1_beta', 1.0)
        return WeightedSmoothL1Loss(target_weights=target_weights, beta=beta)

    # --- Classification Losses (legacy) ---
    elif loss_name == 'bce':
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
