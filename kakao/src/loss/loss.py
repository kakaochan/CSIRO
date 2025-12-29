"""Loss function implementations for CSIRO pasture biomass prediction."""

import torch
import torch.nn as nn


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
    [0.1, 0.1, 0.1, 0.5, 0.2] for [Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM]
    (train.csv alphabetical order)
    """

    def __init__(self, target_weights=None):
        """
        Args:
            target_weights: List or tensor of weights for each target.
                           Default: [0.1, 0.1, 0.1, 0.5, 0.2] (competition weights)
        """
        super().__init__()
        if target_weights is None:
            # Competition weights: [Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM]
            target_weights = [0.1, 0.1, 0.1, 0.5, 0.2]
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
    Weights: [0.1, 0.1, 0.1, 0.5, 0.2] for [Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM]
    Based on reference implementation strategy.
    """

    def __init__(self, target_weights=None, beta=1.0):
        """
        Args:
            target_weights: List or tensor of weights for each target.
                           Default: [0.1, 0.1, 0.1, 0.5, 0.2] (competition weights)
            beta: The threshold at which to change between L1 and L2 loss.
        """
        super().__init__()
        if target_weights is None:
            # Competition weights: [Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM]
            target_weights = [0.1, 0.1, 0.1, 0.5, 0.2]
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

    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
