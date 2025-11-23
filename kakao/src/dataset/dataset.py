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

        sample = {
            'sample_img': sample_image,
            'target': target
        }
        return sample