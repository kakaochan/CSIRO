import torch
import torch.nn as nn
import timm


class TimmBackboneExtractor(nn.Module):
    """timm backboneで特徴抽出"""
    def __init__(self, model_name='convnext_tiny', pretrained=True, in_channels=3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
            in_chans=in_channels
        )
        # self.out_chans = 1
        self.out_channels = self.backbone.num_features

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x)  # (B, num_features)
        return features.unsqueeze(1).unsqueeze(-1)  # (B, 1, num_features, 1)
