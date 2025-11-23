import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR, CosineAnnealingLR
from lightning.pytorch import LightningModule
from src.models.common import get_model
from src.loss import get_loss_function
from torch import nn
import numpy as np
from pathlib import Path


class CSIROModel(LightningModule):
    def __init__(self, cfg, val_fold):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.val_fold = val_fold
        self.validation_step_outputs = []

        self.__best_loss = np.inf

        self.loss_function = get_loss_function(cfg)

        self.net = get_model(
            cfg,
            feature_dim=3,
            n_pairs=1,
            n_actions=5,
            num_timesteps=1,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x = batch['sample_img']
        y = batch['target']

        outputs = self.net(x)
        logits = outputs['logits']

        loss = self.loss_function(logits, y.float())

        self.log(f"train_loss_fold{self.val_fold}", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['sample_img']
        y = batch['target']

        outputs = self.net(x)
        logits = outputs['logits']

        val_loss = self.loss_function(logits, y.float())

        output_dict = {
            "val_loss": val_loss.detach(),
            "logits": logits.detach(),
            "target": y.detach()
        }

        self.validation_step_outputs.append(output_dict)
        return val_loss

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        self.log(f"val_loss_fold{self.val_fold}", avg_loss, on_epoch=True, prog_bar=True)

        if avg_loss < self.__best_loss:
            self.__best_loss = avg_loss
            torch.save(self.state_dict(), Path(self.cfg.dir.model_dir) / f"{self.cfg.exp_name}_best_loss_fold{self.val_fold}.pth")
            print(f"Saved best loss model: {avg_loss:.4f}")

        self.validation_step_outputs = []

    def configure_optimizers(self):
        if self.cfg.trainer.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.cfg.trainer.lr,
                weight_decay=self.cfg.trainer.weight_decay,
            )
        elif self.cfg.trainer.optimizer == "adam":
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.cfg.trainer.lr,
                weight_decay=self.cfg.trainer.weight_decay,
            )
        else:
            raise NotImplementedError(f"{self.cfg.trainer.optimizer} is not valid.")

        if getattr(self.cfg.trainer, "use_warmup", False):
            warmup_epochs = getattr(self.cfg.trainer, "warmup_epochs", 0)
            total_epochs = self.cfg.trainer.epochs

            warmup_scheduler = LinearLR(
                optimizer=optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            main_scheduler = CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=total_epochs - warmup_epochs,
                T_mult=1,
                eta_min=1e-6,
                last_epoch=-1,
            )
            lr_scheduler = SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            lr_scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.cfg.trainer.epochs,
                eta_min=1e-6,
                last_epoch=-1,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            },
        }


def load_model(cfg, val_fold, stage='train', train=True):
    """Load CSIROModel with optional checkpoint loading"""
    if train:
        model_ckpt = getattr(cfg.model, 'model_ckpt', None)
    else:
        model_ckpt = getattr(cfg.model, 'final_model_path', None)

    if model_ckpt is not None:
        state_dict = torch.load(model_ckpt, map_location=cfg.device)["state_dict"]
        print("loading model from checkpoint")
    else:
        state_dict = None

    if state_dict is not None:
        keys = list(state_dict.keys())
        for k in keys:
            if "ema" in k:
                state_dict.pop(k)

    if train:
        model = CSIROModel(cfg=cfg, val_fold=val_fold)
        if state_dict is not None:
            if stage == 'train_finetune':
                keys_to_remove = [k for k in state_dict.keys() if 'decoder' in k]
                for k in keys_to_remove:
                    state_dict.pop(k)
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
    else:
        model = CSIROModel(cfg=cfg, val_fold=val_fold)
        if state_dict is not None:
            model.load_state_dict(state_dict)

    if not train:
        model.eval()

    if cfg.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Auto device selection: {device}")
    elif cfg.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    else:
        device = cfg.device

    model.to(device)

    return model
