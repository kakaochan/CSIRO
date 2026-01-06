import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR, CosineAnnealingLR
from lightning.pytorch import LightningModule
from src.models.common import get_model
from src.loss import get_loss_function
from src.utils.metrics import weighted_r2_score
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path


class CSIROModel(LightningModule):
    def __init__(self, cfg, val_fold):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.val_fold = val_fold
        self.validation_step_outputs = []

        self.__best_r2 = -np.inf

        self.loss_function = get_loss_function(cfg)

        self.net = get_model(
            cfg,
            feature_dim=3,
        )

        # ========= FREEXZE BACKBONE ===========
        if self.cfg.model.freeze_backbone:
            if self.cfg.model.name == "MVPModel":
                for p in self.net.backbone.parameters():
                    p.requires_grad = False
            else:
                pass

    def forward(self, *args, **kwargs):
        """Forward pass supporting both Original and Two-Stream.

        Args:
            *args: Either (x,) for Original or (img_left, img_right) for Two-Stream
            **kwargs: Optional keyword arguments

        Returns:
            Model outputs
        """
        return self.net(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        y = batch['target']

        # Check if Two-Stream or Original
        if 'img_left' in batch and 'img_right' in batch:
            # Two-Stream
            outputs = self(batch['img_left'], batch['img_right'])
        else:
            # Original
            outputs = self(batch['sample_img'])

        logits = outputs['logits'] #(B, 5)

        loss = self.loss_function(logits, y.float())

        self.log(f"train_loss_fold{self.val_fold}", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['target']
        image_ids = batch['image_id']

        # Check if Two-Stream or Original
        if 'img_left' in batch and 'img_right' in batch:
            # Two-Stream
            outputs = self.net(batch['img_left'], batch['img_right'])
        else:
            # Original
            outputs = self.net(batch['sample_img'])

        logits = outputs['logits']

        val_loss = self.loss_function(logits, y.float())

        output_dict = {
            "val_loss": val_loss.detach(),
            "logits": logits.detach(),
            "target": y.detach(),
            "image_id": image_ids
        }

        self.validation_step_outputs.append(output_dict)
        return val_loss
    
    def freeze_backbone(self):
        for p in self.net.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.net.backbone.parameters():
            p.requires_grad = True

    def reset_optimizer(self):
        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )
        self.trainer.optimizers = [optimizer]
        self.trainer.lr_schedulers = [] 

    # def on_train_epoch_start(self):
    #     return super().on_train_epoch_start()

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        # Compute R² score
        all_logits = torch.cat([x["logits"] for x in outputs], dim=0).cpu().numpy()
        all_targets = torch.cat([x["target"] for x in outputs], dim=0).cpu().numpy()
        all_image_ids = [img_id for batch in outputs for img_id in batch["image_id"]]

        weighted_r2, individual_r2s = weighted_r2_score(all_targets, all_logits)

        self.log(f"val_loss_fold{self.val_fold}", avg_loss, on_epoch=True, prog_bar=True)
        self.log(f"val_r2_fold{self.val_fold}", weighted_r2, on_epoch=True, prog_bar=True)

        # Log individual R² scores
        target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
        for i, (name, r2) in enumerate(zip(target_names, individual_r2s)):
            self.log(f"val_r2_{name}_fold{self.val_fold}", r2, on_epoch=True, prog_bar=False)

        # Save submission CSV if enabled
        if self.cfg.save_submission_df:
            self._save_submission_csv(all_image_ids, all_logits, all_targets)

        if weighted_r2 > self.__best_r2:
            self.__best_r2 = weighted_r2
            model_save_path = Path(self.cfg.dir.model_dir) / self.cfg.exp_name / f"best_r2_fold{self.val_fold}.pth"
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), model_save_path)
            print(f"Saved best R² model: {model_save_path}, R²: {weighted_r2:.4f}, Loss: {avg_loss:.4f}")

        self.validation_step_outputs = []

    def _save_submission_csv(self, image_ids, predictions, targets):
        """Create submission CSV from validation predictions."""
        # Target column names in order (matches train.csv alphabetical order)
        target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

        # Create rows for submission format
        rows = []
        for img_id, pred_vals, true_vals in zip(image_ids, predictions, targets):
            for col_name, pred_val, true_val in zip(target_cols, pred_vals, true_vals):
                sample_id = f"{img_id}__{col_name}"
                rows.append({
                    'sample_id': sample_id,
                    'target': pred_val,
                    'true_target': true_val  # For debugging
                })

        # Create DataFrame
        submission_df = pd.DataFrame(rows)

        # Save to outputs directory
        output_dir = Path(self.cfg.dir.outputs_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"submission_fold{self.val_fold}_epoch{self.current_epoch}.csv"
        submission_df.to_csv(csv_path, index=False)

        print(f"Saved submission CSV: {csv_path}")

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
