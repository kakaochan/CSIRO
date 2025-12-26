from pathlib import Path
import numpy as np
import polars as pl
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
from src.dataset.common import get_datasets
# from src.dataset.collate import custom_collate_fn

class CSIRODataModule(LightningDataModule):
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
        meta_df = pd.read_csv(Path(self.cfg.dir.input_dir)/"train.csv")
        self.train_ds, self.valid_ds = get_datasets(cfg=self.cfg, df=train_df, meta_df=meta_df, val_fold=self.val_fold)

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.cfg.train_loader)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, **self.cfg.val_loader)