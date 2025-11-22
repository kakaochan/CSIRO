import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR, CosineAnnealingLR
from lightning.pytorch import LightningModule
from src.models.common import get_model
from src.loss import get_loss_function
from torch import nn
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import json


class MouseBehaviorModel(LightningModule):
    def __init__(self, cfg, val_fold):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.val_fold = val_fold
        self.validation_step_outputs = []

        self.__best_loss = np.inf
        self.__best_metric = -np.inf

        # pos_weight統計用
        self.pos_weight_stats = []

        # 損失関数
        self.loss_function = get_loss_function(cfg)

        # Model
        self.n_pairs = 4  # 2 mice: 2x2 = 4 pairs
        self.n_actions = 37
        self.net = get_model(
            cfg,
            feature_dim=8,  # 2 mice × 2 bodyparts × 2 coords
            n_pairs=self.n_pairs,
            n_actions=self.n_actions,
            num_timesteps=10000,
        )

        # Load pair_mapping and actions for F-beta calculation
        pair_mapping_path = Path(cfg.dir.processed_dir) / 'pair_mapping.json'
        if pair_mapping_path.exists():
            with open(pair_mapping_path, 'r') as f:
                mapping_data = json.load(f)
            # Convert pair_mapping to expected format: {0: "1_1", 1: "1_2", ...}
            raw_pair_mapping = mapping_data['pair_mapping']
            self.pair_mapping = {
                int(k): f"{v['agent_id']}_{v['target_id']}"
                for k, v in raw_pair_mapping.items()
            }
            self.actions = mapping_data['actions']
            self.use_fbeta = True
        else:
            self.pair_mapping = None
            self.actions = None
            self.use_fbeta = False
            print("Warning: pair_mapping.json not found. F-beta calculation will be disabled.")

    def forward(self, x):
        return self.net(x)

    def _compute_pos_weight(self, y_reshaped, n_classes):
        """Compute pos_weight based on config settings.

        Args:
            y_reshaped: (batch, n_classes, n_timesteps)
            n_classes: number of classes

        Returns:
            pos_weight: (n_classes,) tensor
        """
        pos_weight_cfg = self.cfg.model.pos_weight

        if pos_weight_cfg.mode == 'fixed':
            # Fixed pos_weight for all classes
            pos_weight = torch.full((n_classes,), pos_weight_cfg.value,
                                   dtype=torch.float32, device=y_reshaped.device)
        elif pos_weight_cfg.mode == 'dynamic':
            # Dynamic pos_weight based on batch statistics
            pos_count = y_reshaped.sum(dim=(0, 2))  # (n_classes,)
            neg_count = y_reshaped.size(0) * y_reshaped.size(2) - pos_count
            pos_weight = neg_count / (pos_count + 1e-6)  # (n_classes,)
            pos_weight = torch.clamp(pos_weight, min=pos_weight_cfg.min, max=pos_weight_cfg.max)
        else:
            raise ValueError(f"Invalid pos_weight mode: {pos_weight_cfg.mode}")

        return pos_weight

    def training_step(self, batch, batch_idx):
        x = batch['x']  # (batch, 8, 10000) - 2 mice × 2 bodyparts × 2 coords
        y = batch['y']  # (batch, 4, 37, 10000) - 4 pairs × 37 actions

        outputs = self.net(x)
        logits = outputs['logits']  # (batch, n_timesteps, n_classes)
        logits = logits.transpose(1, 2)  # (batch, n_classes, n_timesteps)

        # Reshape for loss calculation
        y_reshaped = y.view(y.size(0), -1, y.size(-1))  # (batch, 4*37=148, 10000)
        n_classes = y_reshaped.size(1)

        # Compute pos_weight
        pos_weight = self._compute_pos_weight(y_reshaped, n_classes)

        # Record pos_weight for statistics
        self.pos_weight_stats.append(pos_weight.detach().cpu())

        # Reshape pos_weight for broadcasting: (n_classes,) -> (1, n_classes, 1)
        pos_weight_reshaped = pos_weight.view(1, -1, 1)

        # Compute loss with pos_weight
        loss = self.loss_function(logits, y_reshaped.float(), pos_weight=pos_weight_reshaped)

        self.log(f"train_loss_fold{self.val_fold}", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['x']  # (batch, 8, 10000) - 2 mice × 2 bodyparts × 2 coords
        y = batch['y']  # (batch, 4, 37, 10000) - 4 pairs × 37 actions

        outputs = self.net(x)
        logits = outputs['logits']  # (batch, n_timesteps, n_classes)
        logits = logits.transpose(1, 2)  # (batch, n_classes, n_timesteps)

        # Reshape for loss calculation
        y_reshaped = y.view(y.size(0), -1, y.size(-1))  # (batch, 4*37=148, 10000)
        n_classes = y_reshaped.size(1)

        # Compute pos_weight (same as training)
        pos_weight = self._compute_pos_weight(y_reshaped, n_classes)

        # Reshape pos_weight for broadcasting: (n_classes,) -> (1, n_classes, 1)
        pos_weight_reshaped = pos_weight.view(1, -1, 1)

        # Compute loss with pos_weight
        val_loss = self.loss_function(logits, y_reshaped.float(), pos_weight=pos_weight_reshaped)

        # Store outputs with metadata for F-beta calculation
        output_dict = {
            "val_loss": val_loss.detach(),
            "logits": logits.detach(),
            "target": y_reshaped.detach()
        }

        # Add metadata for F-beta calculation if enabled
        if self.use_fbeta:
            output_dict["target_original"] = y.detach()  # Keep original shape (batch, 4, 37, 10000)
            output_dict["video_ids"] = batch['video_id']
            output_dict["lab_ids"] = batch['lab_id']
            output_dict["behaviors_labeled"] = batch['behaviors_labeled']

        self.validation_step_outputs.append(output_dict)
        return val_loss

    def on_train_epoch_end(self):
        """Output pos_weight statistics at the end of training epoch."""
        if len(self.pos_weight_stats) > 0:
            all_weights = torch.stack(self.pos_weight_stats).mean(dim=0)  # (n_classes,)
            print(f"\n[Train Epoch {self.current_epoch}] Pos_weight stats - "
                  f"min: {all_weights.min():.2f}, max: {all_weights.max():.2f}, "
                  f"mean: {all_weights.mean():.2f}, median: {all_weights.median():.2f}")
            self.pos_weight_stats = []  # Clear for next epoch

    def _save_validation_logits(self, outputs):
        """Save probabilities (sigmoid of logits) per video for analysis.

        Directory structure:
        processed_data/logits/fold{val_fold}/epoch{epoch}/{video_id}.csv

        CSV format (rows=timesteps, columns=pair_action, values=probabilities [0-1]):
        timestep,1_1_allogroom,1_1_approach,...,2_2_tussle
        0,0.089,0.226,...,0.187
        1,0.226,0.059,...,0.368
        ...
        9999,0.031,0.226,...,0.143
        """
        # Create directory structure
        logits_base_dir = Path(self.cfg.dir.processed_dir) / 'logits'
        fold_dir = logits_base_dir / f'fold{self.val_fold}'
        epoch_dir = fold_dir / f'epoch{self.current_epoch}'
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Process each batch
        batch_offset = 0
        for output in outputs:
            batch_logits = output["logits"]  # (batch, n_classes, n_timesteps)
            batch_video_ids = output["video_ids"]

            batch_size = batch_logits.shape[0]

            for i in range(batch_size):
                video_id = batch_video_ids[i]
                if isinstance(video_id, torch.Tensor):
                    video_id = int(video_id.item())

                # Get logits for this video: (n_classes, n_timesteps)
                video_logits = batch_logits[i].cpu().numpy()

                # Convert to probabilities (sigmoid)
                video_probs = 1 / (1 + np.exp(-video_logits))  # (n_classes, n_timesteps)

                # Transpose to (n_timesteps, n_classes) for easier reading
                # Each row = timestep, Each column = pair_action
                import pandas as pd
                video_probs_transposed = video_probs.T  # (n_timesteps, n_classes)

                # Create meaningful column names: pair_action (e.g., "1_1_allogroom")
                columns = []
                for pair_idx in range(self.n_pairs):
                    pair_name = self.pair_mapping[pair_idx]  # e.g., "1_1"
                    for action in self.actions:
                        columns.append(f'{pair_name}_{action}')

                # Create DataFrame with timestep as index
                df = pd.DataFrame(video_probs_transposed, columns=columns)
                df.index.name = 'timestep'

                # Save as CSV
                save_path = epoch_dir / f'{video_id}.csv'
                df.to_csv(save_path)

            batch_offset += batch_size

        print(f"Saved logits for {batch_offset} videos to: {epoch_dir}")

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        all_logits  = torch.cat([x["logits"] for x in outputs], dim=0)
        all_targets = torch.cat([x["target"] for x in outputs], dim=0).float()

        # Save logits per video if enabled
        if getattr(self.cfg, 'save_logits', False):
            self._save_validation_logits(outputs)

        # Analyze ground truth label distribution
        all_targets_flat = all_targets.sum(dim=(0, 2))  # (n_classes,) - sum across batch and time
        classes_with_labels = (all_targets_flat > 0).sum().item()
        print(f"\n[Validation Epoch {self.current_epoch}] Ground truth distribution:")
        print(f"  Classes with labels > 0: {classes_with_labels} / {all_targets_flat.shape[0]}")
        if classes_with_labels > 0:
            top_k = min(10, classes_with_labels)
            top_values, top_indices = torch.topk(all_targets_flat, top_k)
            print(f"  Top {top_k} classes by frequency:")
            for i in range(top_k):
                class_idx = top_indices[i].item()
                count = top_values[i].item()
                pair_idx = class_idx // self.n_actions
                action_idx = class_idx % self.n_actions
                print(f"    Class {class_idx} (pair={pair_idx}, action={action_idx}): {count:.0f} frames")

        # # Old binary F1 (removed - was meaningless for multi-label classification)
        # preds = (torch.sigmoid(all_logits) > 0.5).int().cpu().numpy()
        # tgts  = all_targets.int().cpu().numpy()
        # f1 = f1_score(tgts.ravel(), preds.ravel(), average="binary", zero_division=0)
        # self.log(f"binary_f1_fold{self.val_fold}", f1, on_epoch=True, prog_bar=True)

        self.log(f"val_loss_fold{self.val_fold}", avg_loss, on_epoch=True, prog_bar=True)

        # Calculate F-beta score if enabled
        fbeta_score = 0.0
        if self.use_fbeta:
            from src.utils.submission import logits_to_submission, labels_to_solution
            from src.utils.loss.f_beta import mouse_fbeta

            # Collect all metadata
            # Convert tensors to Python types if needed
            all_video_ids = []
            for x in outputs:
                for vid in x["video_ids"]:
                    if isinstance(vid, torch.Tensor):
                        all_video_ids.append(int(vid.item()))
                    else:
                        all_video_ids.append(vid)

            all_lab_ids = []
            for x in outputs:
                for lab in x["lab_ids"]:
                    if isinstance(lab, torch.Tensor):
                        all_lab_ids.append(str(lab.item()) if lab.dtype == torch.float32 else lab.item())
                    else:
                        all_lab_ids.append(lab)

            all_behaviors_labeled = []
            for x in outputs:
                for behavior in x["behaviors_labeled"]:
                    all_behaviors_labeled.append(behavior)

            all_targets_original = torch.cat([x["target_original"] for x in outputs], dim=0)

            # Convert to DataFrames
            # Get threshold from config, default to 0.5 if not specified
            if hasattr(self.cfg, 'fbeta') and hasattr(self.cfg.fbeta, 'threshold'):
                threshold = self.cfg.fbeta.threshold
            else:
                threshold = 0.5
            print(f"Using threshold: {threshold} for F-beta calculation")

            # Debug: Check logits and probabilities statistics
            probs = torch.sigmoid(all_logits).cpu().numpy()
            print(f"Logits - min: {all_logits.min():.4f}, max: {all_logits.max():.4f}, mean: {all_logits.mean():.4f}")
            print(f"Probs - min: {probs.min():.4f}, max: {probs.max():.4f}, mean: {probs.mean():.4f}")
            print(f"Probs > {threshold}: {(probs > threshold).sum()} / {probs.size} = {(probs > threshold).mean():.4f}")

            submission_df = logits_to_submission(
                all_logits,
                all_video_ids,
                self.pair_mapping,
                self.actions,
                threshold=threshold
            )

            solution_df = labels_to_solution(
                all_targets_original,
                all_video_ids,
                all_lab_ids,
                self.pair_mapping,
                self.actions,
                behaviors_labeled=all_behaviors_labeled
            )

            # Debug information
            print(f"submission_df shape: {len(submission_df)} rows")
            print(f"solution_df shape: {len(solution_df)} rows")
            if len(submission_df) > 0:
                print(f"submission_df sample:\n{submission_df.head()}")
            if len(solution_df) > 0:
                print(f"solution_df sample:\n{solution_df.head()}")

            # Calculate F-beta score with Precision/Recall
            if len(submission_df) > 0 and len(solution_df) > 0:
                fbeta_score, precision, recall = mouse_fbeta(solution_df, submission_df, beta=1.0, return_details=True)
                self.log(f"fbeta_fold{self.val_fold}", fbeta_score, on_epoch=True, prog_bar=True)
                self.log(f"precision_fold{self.val_fold}", precision, on_epoch=True, prog_bar=False)
                self.log(f"recall_fold{self.val_fold}", recall, on_epoch=True, prog_bar=False)
                print(f"F-beta score: {fbeta_score:.4f}")
                print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

                # Optionally save submission DataFrame for inspection
                if getattr(self.cfg, 'save_submission_df', False):
                    save_dir = Path(self.cfg.dir.outputs_dir) / "submissions"
                    save_dir.mkdir(exist_ok=True)
                    submission_path = save_dir / f"val_submission_fold{self.val_fold}_epoch{self.current_epoch}.csv"
                    submission_df.to_csv(submission_path, index=False)
                    print(f"Saved submission to: {submission_path}")
            else:
                print("Warning: Empty submission or solution DataFrame. F-beta score not calculated.")

        # Use F-beta for model selection
        if self.use_fbeta and fbeta_score > self.__best_metric:
            self.__best_metric = fbeta_score
            torch.save(self.state_dict(), Path(self.cfg.dir.model_dir) / f"{self.cfg.exp_name}_best_score_fold{self.val_fold}.pth")
            print(f"Saved best score model: F-beta={fbeta_score:.4f}")

        if avg_loss < self.__best_loss:
            self.__best_loss = avg_loss
            torch.save(self.state_dict(), Path(self.cfg.dir.model_dir) / f"{self.cfg.exp_name}_best_loss_fold{self.val_fold}.pth")
            print(f"Saved best loss model: {avg_loss:.4f}")

        self.validation_step_outputs = []

    def configure_optimizers(self):
        """ Optimizer & Scheduler setup (CMI参考) """
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

        # warmup + cosine annealing
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
                "monitor": f"binary_f1_fold{self.val_fold}",
                "mode": "max",
            },
        }


def load_model(cfg, val_fold, stage='train', train=True):
    """Load MouseBehaviorModel with optional checkpoint loading"""
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
        model = MouseBehaviorModel(cfg=cfg, val_fold=val_fold)
        if state_dict is not None:
            # pretrain to train - remove decoder weights if needed
            if stage == 'train_finetune':
                keys_to_remove = [k for k in state_dict.keys() if 'decoder' in k]
                for k in keys_to_remove:
                    state_dict.pop(k)
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
    else:
        model = MouseBehaviorModel(cfg=cfg, val_fold=val_fold)
        if state_dict is not None:
            model.load_state_dict(state_dict)

    if not train:
        model.eval()
    
    # デバイス自動判定
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
