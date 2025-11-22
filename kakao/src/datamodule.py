from pathlib import Path
import numpy as np
import polars as pl
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
from src.dataset.common import get_datasets
from src.dataset.collate import custom_collate_fn

class MabeMiceDatamodule(LightningDataModule):
    def __init__(self, cfg, val_fold):
        super().__init__()    
        self.cfg = cfg
        self.val_fold = val_fold

        self.train_ds = None
        self.valid_ds = None
        #multi phaseモデルを作成する際に、使用する用
        self.test_df = None

    def setup(self, stage):
        print('datamodule_setup')
        train_df = pd.read_csv(Path(self.cfg.dir.input_dir)/"train_with_split.csv")
        dont_use_labs = [
            'MABe22_keypoints', 'MABe22_movies',  # アノテーションなし
            'BoisterousParrot',  # 密度0.6%
            'PleasantMeerkat',   # 密度4.5%
            'AdaptableSnail',    # 3-4匹マウス含む
        ]
        train_df = train_df.loc[~train_df['lab_id'].isin(dont_use_labs)]

        self.train_ds, self.valid_ds = get_datasets(cfg=self.cfg, df=train_df, val_fold=self.val_fold)

    def train_dataloader(self):
        return DataLoader(self.train_ds, collate_fn=custom_collate_fn, **self.cfg.train_loader)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, collate_fn=custom_collate_fn, **self.cfg.val_loader)