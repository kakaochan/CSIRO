from src.dataset.dataset import MouseDataset


_DATASET_MAP = {
    "original":   MouseDataset
    #増えたらこれ以降に書きましょうね。
}

def get_datasets(cfg, df, val_fold):
    key = cfg.dataset.name.lower()
    if key not in _DATASET_MAP:
        raise ValueError(f"Unsupported dataset: {key}")
    train_df = df.loc[df['fold'] != val_fold].drop("fold", axis=1)
    valid_df = df.loc[df['fold'] == val_fold].drop("fold", axis=1)

    DatasetCls = _DATASET_MAP[key]

    train_ds = DatasetCls(cfg, val_fold, train_df, mode='train')
    valid_ds = DatasetCls(cfg, val_fold, valid_df, mode='valid')
    return train_ds, valid_ds