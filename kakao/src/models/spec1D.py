from typing import Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn


class Spec1D(nn.Module):
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

        # ★★★ channels_fc を削除（1→1は害しかないため） ★★★
        self.use_channels_fc = False

        # loss params (unused now)
        self.loss_weight = torch.tensor(cfg.loss.loss_weight) if "loss_weight" in cfg.loss else None
        self.label_weight = torch.tensor(cfg.label_weight) if "label_weight" in cfg else None
        self.pos_weight = torch.tensor(cfg.pos_weight) if "pos_weight" in cfg else None

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> dict[str, torch.Tensor]:

        # feature_extractor 出力: (B, 1, n_features=768(モデル固有), 1)
        x = self.feature_extractor(x)

        # decoder へ入力：(B, 1, n_features=768(モデル固有), 1)
        logits = self.decoder(x)    # (B, 1, n_classes=5)

        logits = logits.squeeze(1)  # (B, n_classes=5)

        output = {"logits": logits}

        if labels is not None:
            loss = self.loss_fn(logits, labels, masks)
            output["loss"] = loss

        return output
