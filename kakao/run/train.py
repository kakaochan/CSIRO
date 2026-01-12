import numpy as np 
import gc
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from src.datamodule import CSIRODataModule
from src.modelmodule import load_model


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def main(cfg):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(cfg.seed, workers=True)
    model_save_dir: Path = Path(cfg.dir.model_dir) / cfg.exp_name
    os.makedirs(model_save_dir, exist_ok=True)

    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project="CSIRO",
        entity="gaiji",
        offline=cfg.offline,
        save_dir=cfg.dir.outputs_dir,
        # checkpoint_name=cfg.exp_name,  # ← Lightning/WandbLoggerには存在しないのでコメントアウト
    )
    pl_logger.log_hyperparams(cfg)

    for val_fold in range(cfg.n_splits):
        LOGGER.info(f'start training val_fold {val_fold}')
        datamodule = CSIRODataModule(cfg=cfg, val_fold=val_fold)
        datamodule.setup(stage='fit')

        # scalerを取得してmodelに渡す
        scaler = datamodule.scaler
        model = load_model(cfg=cfg, val_fold=val_fold, scaler=scaler)

        lr_monitor = LearningRateMonitor("epoch")
        progress_bar = TQDMProgressBar()
        model_summary = RichModelSummary(max_depth=2)

        # state_onlyの場合はState精度を監視
        if cfg.model.get('state_only', False):
            monitor_metric = f"val_StateAcc_fold{val_fold}"
            monitor_mode = "max"
        else:
            monitor_metric = f"val_r2_fold{val_fold}"
            monitor_mode = "max"

        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=cfg.trainer.patience,
        )

        trainer = Trainer(
            default_root_dir=cfg.dir.outputs_dir,
            accelerator=cfg.trainer.accelerator,
            precision="16-mixed" if cfg.trainer.use_amp else 32,
            # training
            max_epochs=cfg.trainer.epochs,
            gradient_clip_val=cfg.trainer.gradient_clip_val,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
            callbacks=[lr_monitor, progress_bar, model_summary, early_stopping],
            logger=pl_logger,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
            enable_checkpointing=False,
            log_every_n_steps=10,
        )

        trainer.fit(model=model, datamodule=datamodule)

        del model, datamodule, trainer
        torch.cuda.empty_cache()
        gc.collect()

    # WandBは自動的に終了されるため、手動でfinish()を呼ぶ必要なし
    # (Hydra + WandB + Colabの組み合わせで finish() を手動呼び出しすると
    #  logging callbackとの競合で OSError が発生する可能性がある)
    # pl_logger.experiment.finish()


if __name__ == "__main__":
    main()
