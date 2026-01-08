from pathlib import Path
import numpy as np
import polars as pl
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
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
        self.scaler = None  # StandardScalerを保存

    def setup(self, stage):
        print('datamodule_setup')
        df = pd.read_csv(Path(self.cfg.dir.input_dir)/"integrated_train.csv")

        # Configで正規化の有効/無効を制御
        use_normalization = self.cfg.get('use_target_normalization', True)

        if use_normalization:
            print("Target normalization: ENABLED")

            # 既存scalerのパスがあるかチェック
            scaler_path_config = self.cfg.get('scaler_path', None)

            if scaler_path_config is not None:
                # Stage 2: 既存scalerをロード
                # {fold}プレースホルダーを現在のfold番号で置換
                scaler_path_config = scaler_path_config.replace('{fold}', str(self.val_fold))
                scaler_load_path = Path(scaler_path_config)
                self.scaler = joblib.load(scaler_load_path)
                print(f"Loaded existing scaler from: {scaler_load_path}")
                print(f"  Mean: {self.scaler.mean_}")
                print(f"  Std:  {self.scaler.scale_}")
            else:
                # Stage 1: 新規にfit
                # Trainデータから統計量を計算（Valは含めない）
                train_df = df[df['fold'] != self.val_fold]
                target_cols = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g',
                               'Dry_Total_g', 'GDM_g']
                train_targets = train_df[target_cols].values  # (N_train, 5)

                # StandardScalerでfit
                self.scaler = StandardScaler()
                self.scaler.fit(train_targets)

                # scalerを保存（推論時に使用）
                scaler_dir = Path(self.cfg.dir.model_dir) / self.cfg.exp_name
                scaler_dir.mkdir(parents=True, exist_ok=True)
                scaler_path = scaler_dir / f"scaler_fold{self.val_fold}.pkl"
                joblib.dump(self.scaler, scaler_path)
                print(f"Saved new scaler to: {scaler_path}")
                print(f"  Mean: {self.scaler.mean_}")
                print(f"  Std:  {self.scaler.scale_}")
        else:
            print("Target normalization: DISABLED (using raw scale)")
            self.scaler = None

        # データセット作成（scalerを渡す）
        self.train_ds, self.valid_ds = get_datasets(
            cfg=self.cfg,
            df=df,
            val_fold=self.val_fold,
            scaler=self.scaler
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.cfg.train_loader)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, **self.cfg.val_loader)