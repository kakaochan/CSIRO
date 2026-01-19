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
    def __init__(self, cfg, val_fold, train_df, mode='train', scaler=None):
        self.cfg = cfg
        self.val_fold = val_fold
        self.train_df = train_df
        self.mode = mode
        self.image_paths = Path(self.cfg.dir.data_dir) / 'train'
        self.scaler = scaler

    def __len__(self):
        return len(self.train_df)


    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        image_id = row['image_id']
        jpg_path = self.image_paths / f"{image_id}.jpg"
        target = np.array([
            row['Dry_Clover_g'],
            row['Dry_Dead_g'],
            row['Dry_Green_g'],
            row['Dry_Total_g'],
            row['GDM_g']
        ], dtype=np.float32)
        sample_image = cv2.imread(str(jpg_path))


        if sample_image is None:
            print(f"Warning: 画像が読み込めません: {jpg_path}. 黒画像を返します.")
            sample_image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        sample_image = torch.from_numpy(sample_image).permute(2, 0, 1).float() / 255.0

        # StandardScalerで正規化
        if self.scaler is not None:
            target_norm = self.scaler.transform(target.reshape(1, -1))[0]  # (5,)
            target = torch.from_numpy(target_norm).float()
            if self.cfg.model.target_ratio:
                ratio_target = np.array([
                    target_norm[3],
                    row["Dead_per_Total"],
                    row["Green_per_Total"]
                ])
                ratio_target = torch.from_numpy(ratio_target).float()
        else:
            if self.cfg.model.target_ratio:
                ratio_target = np.array([
                    target[3],
                    row["Dead_per_Total"],
                    row["Green_per_Total"]
                ])
                ratio_target = torch.from_numpy(ratio_target).float()
            target = torch.from_numpy(target).float()

        sample = {
            'sample_img': sample_image,  # (3, H, W)
            'target': target,  # (5,) 正規化済み
            'image_id': image_id  # For submission CSV creation
        }
        if self.cfg.model.target_ratio:
            sample['ratio_target'] = ratio_target

        return sample


class CSIROTwoStreamDataset(Dataset):
    """Two-Stream Dataset following REFERENCE implementation.

    Splits 2000x1000 images into left and right 1000x1000 patches,
    resizes each to 768x768 to preserve fine-grained details.
    """

    def __init__(self, cfg, val_fold, train_df, mode='train', scaler=None):
        self.cfg = cfg
        self.val_fold = val_fold
        self.train_df = train_df
        self.mode = mode
        self.image_paths = Path(self.cfg.dir.data_dir) / 'train'
        self.img_size = cfg.dataset.img_size
        self.scaler = scaler

        # Augmentation (training時のみ)
        if self.mode == 'train':
            self.transform = self.build_train_transform()
        else:
            self.transform = None

    def build_train_transform(self):
        """学習用のAugmentationを構築"""
        aug_cfg = self.cfg.augmentation
        transforms = []

        # 幾何変換
        if aug_cfg.get('hflip_prob', 0) > 0:
            transforms.append(A.HorizontalFlip(p=aug_cfg.hflip_prob))
        if aug_cfg.get('vflip_prob', 0) > 0:
            transforms.append(A.VerticalFlip(p=aug_cfg.vflip_prob))

        # 色調変換（個別にON/OFF可能）
        if aug_cfg.get('brightness_prob', 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=0,
                val_shift_limit=aug_cfg.get('brightness_limit', 20),
                p=aug_cfg.brightness_prob
            ))
        if aug_cfg.get('saturation_prob', 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=aug_cfg.get('saturation_limit', 20),
                val_shift_limit=0,
                p=aug_cfg.saturation_prob
            ))
        if aug_cfg.get('hue_prob', 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=aug_cfg.get('hue_limit', 20),
                sat_shift_limit=0,
                val_shift_limit=0,
                p=aug_cfg.hue_prob
            ))

        # RandomErasing (CoarseDropout in Albumentations)
        if aug_cfg.get('random_erasing_prob', 0) > 0:
            transforms.append(A.CoarseDropout(
                max_holes=1,
                max_height=int(self.img_size * aug_cfg.get('random_erasing_scale_max', 0.2)),
                max_width=int(self.img_size * aug_cfg.get('random_erasing_scale_max', 0.2)),
                min_holes=1,
                min_height=int(self.img_size * aug_cfg.get('random_erasing_scale_min', 0.02)),
                min_width=int(self.img_size * aug_cfg.get('random_erasing_scale_min', 0.02)),
                fill_value=0,
                p=aug_cfg.random_erasing_prob
            ))

        if transforms:
            return A.ReplayCompose(transforms)  # ReplayComposeで同じ変換を再生可能に
        return None

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]
        image_id = row['image_id']
        jpg_path = self.image_paths / f"{image_id}.jpg"

        target = np.array([
            row['Dry_Clover_g'],
            row['Dry_Dead_g'],
            row['Dry_Green_g'],
            row['Dry_Total_g'],
            row['GDM_g']
        ], dtype=np.float32)

        # Load image (2000x1000)
        sample_image = cv2.imread(str(jpg_path))
        if sample_image is None:
            print(f"Warning: 画像が読み込めません: {jpg_path}. 黒画像を返します.")
            sample_image = np.zeros((1000, 2000, 3), dtype=np.uint8)

        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        # Split into left and right patches (1000x1000 each)
        height, width, _ = sample_image.shape
        mid_point = width // 2
        img_left = sample_image[:, :mid_point, :]      # (1000, 1000, 3)
        img_right = sample_image[:, mid_point:, :]     # (1000, 1000, 3)

        # Resize each patch to 768x768 (preserves more detail than resizing full 2000x1000)
        img_left = cv2.resize(img_left, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Apply augmentation (training時のみ、両パッチに同じ変換を適用)
        if self.transform is not None:
            result_left = self.transform(image=img_left)
            img_left = result_left['image']
            # 同じ変換をright patchに再生
            img_right = A.ReplayCompose.replay(result_left['replay'], image=img_right)['image']

        # Convert to torch tensors and normalize
        img_left = torch.from_numpy(img_left).permute(2, 0, 1).float() / 255.0   # (3, 768, 768)
        img_right = torch.from_numpy(img_right).permute(2, 0, 1).float() / 255.0 # (3, 768, 768)

        # StandardScalerで正規化
        if self.scaler is not None:
            target_norm = self.scaler.transform(target.reshape(1, -1))[0]  # (5,)
            target = torch.from_numpy(target_norm).float()
            if self.cfg.model.target_ratio:
                ratio_target = np.array([
                    target_norm[3],
                    row["Dead_per_Total"],
                    row["Green_per_Total"]
                ])
                ratio_target = torch.from_numpy(ratio_target).float()
        else:
            if self.cfg.model.target_ratio:
                ratio_target = np.array([
                    target[3],
                    row["Dead_per_Total"],
                    row["Green_per_Total"]
                ])
                ratio_target = torch.from_numpy(ratio_target).float()
            target = torch.from_numpy(target).float()

        sample = {
            'img_left': img_left,      # (3, 768, 768)
            'img_right': img_right,    # (3, 768, 768)
            'target': target,          # (5,) 正規化済み
            'image_id': image_id
        }
        if self.cfg.model.target_ratio:
            sample['ratio_target'] = ratio_target

        if self.cfg.model.target_state:
            sample['state_target'] = torch.tensor(row['state_target'], dtype=torch.long)

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


class CSIROTwoStreamTestDataset(Dataset):
    """Two-Stream test dataset for Kaggle submission (no targets).

    Splits 2000x1000 images into left and right 1000x1000 patches,
    resizes each to 768x768 to preserve fine-grained details.
    """

    def __init__(self, cfg, test_df):
        self.cfg = cfg
        self.test_df = test_df
        # Extract unique images from test.csv (1 image = 5 rows)
        self.unique_images = test_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
        self.image_dir = Path(cfg.kaggle.test_images)
        self.img_size = 768  # Target resize dimension

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

        # Split into left and right patches (1000x1000 each)
        height, width, _ = sample_image.shape
        mid_point = width // 2
        img_left = sample_image[:, :mid_point, :]      # (1000, 1000, 3)
        img_right = sample_image[:, mid_point:, :]     # (1000, 1000, 3)

        # Resize each patch to 768x768
        img_left = cv2.resize(img_left, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Convert to torch tensors and normalize
        img_left = torch.from_numpy(img_left).permute(2, 0, 1).float() / 255.0   # (3, 768, 768)
        img_right = torch.from_numpy(img_right).permute(2, 0, 1).float() / 255.0 # (3, 768, 768)

        return {
            'img_left': img_left,      # (3, 768, 768)
            'img_right': img_right,    # (3, 768, 768)
            'image_id': image_id
        }