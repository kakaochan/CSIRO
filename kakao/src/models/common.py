from typing import Union

import torch.nn as nn
from omegaconf import DictConfig


from src.models.decoder.multihead_decoder import MultiHeadDecoder
from src.models.decoder.three_target_decoder import ThreeTargetDecoder

from src.models.feature_extractor.timm_backbone import TimmBackboneExtractor
from src.models.spec1D import Spec1D
from src.models.spec1D_two_stream import Spec1DTwoStream

MODELS = Union[Spec1D, Spec1DTwoStream]


def get_feature_extractor(cfg: DictConfig, feature_dim: int):

    if cfg.feature_extractor.name == "TimmBackboneExtractor":
        feature_extractor = TimmBackboneExtractor(
            model_name=cfg.feature_extractor.model_name,
            pretrained=cfg.feature_extractor.pretrained,
            in_channels=feature_dim,
        )
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.feature_extractor.name}")

    return feature_extractor


def get_decoder(cfg: DictConfig, n_channels: int, n_classes):


    if cfg.decoder.name == "MultiHeadDecoder":
        decoder = MultiHeadDecoder(
            input_size=n_channels,
            n_classes=n_classes,
            dropout=cfg.decoder.dropout,
        )
    elif cfg.decoder.name == "ThreeTargetDecoder":
        decoder = ThreeTargetDecoder(
            cfg=cfg,
            n_channels=n_channels,
        )
    else:
        raise ValueError(f"Invalid decoder name: {cfg.decoder.name}")

    return decoder


def get_model(cfg: DictConfig, feature_dim: int, ) -> MODELS:
    model: MODELS

    if cfg.model.name == "Spec1D":
        feature_extractor = get_feature_extractor(cfg, feature_dim)
        decoder = get_decoder(cfg, n_channels=feature_extractor.out_channels, n_classes=5)
        model = Spec1D(
            cfg=cfg,
            feature_extractor=feature_extractor,
            decoder=decoder,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )

    elif cfg.model.name == "Spec1DTwoStream":
        feature_extractor = get_feature_extractor(cfg, feature_dim)
        # Two-Stream: features are concatenated, so decoder receives 2x channels
        decoder = get_decoder(cfg, n_channels=feature_extractor.out_channels * 2, n_classes=5)
        model = Spec1DTwoStream(
            cfg=cfg,
            feature_extractor=feature_extractor,
            decoder=decoder,
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )

    else:
        raise NotImplementedError

    return model