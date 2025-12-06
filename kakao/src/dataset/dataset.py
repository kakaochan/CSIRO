from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
from tqdm import tqdm
import gc

class CSIRODataset(Dataset):
    def __init__(self, cfg, val_fold, train_df, meta_df, mode='train'):
        self.cfg = cfg
        self.val_fold = val_fold
        self.train_df = train_df
        self.meta_df = meta_df
        self.mode = mode
        self.image_paths = Path(self.cfg.dir.data_dir) / 'train'

    def __len__(self):
        return len(self.train_df)


    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        image_id = row['image_id']
        jpg_path = self.image_paths / f"{image_id}.jpg"
        df_subset = self.meta_df.loc[self.meta_df['sample_id'].str.startswith(image_id)]
        target = df_subset['target'].values
        sample_image = cv2.imread(str(jpg_path))

        if sample_image is None:
            print(f"Warning: 画像が読み込めません: {jpg_path}. 黒画像を返します.")
            sample_image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        sample_image = torch.from_numpy(sample_image).permute(2, 0, 1).float() / 255.0
        target = torch.from_numpy(target).float()

        sample = {
            'sample_img': sample_image,  # (3, H, W)
            'target': target,  # (5,)
            'image_id': image_id  # For submission CSV creation
        }
        return sample


class CSIROTestDataset(Dataset):
    """Test dataset for Kaggle submission (no targets)"""

    def __init__(self, cfg, test_df):
        self.cfg = cfg
        self.test_df = test_df
        # Extract unique images from test.csv (1 image = 5 rows)
        self.unique_images = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
        self.image_dir = Path(cfg.kaggle.test_images)

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        row = self.unique_images.iloc[idx]
        # Extract image_id from 'test/ID1001187975.jpg' -> 'ID1001187975'
        image_path = row['image_path']
        image_id = Path(image_path).stem  # 'ID1001187975'

        jpg_path = self.image_dir / f"{image_id}.jpg"
        sample_image = cv2.imread(str(jpg_path))

        if sample_image is None:
            print(f"Warning: 画像が読み込めません: {jpg_path}. 黒画像を返します.")
            sample_image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        # Same preprocessing as training
        sample_image = torch.from_numpy(sample_image).permute(2, 0, 1).float() / 255.0

        return {
            'sample_img': sample_image,  # (3, H, W)
            'image_id': image_id  # For submission CSV
        }