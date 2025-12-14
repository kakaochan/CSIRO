from src.dataset.dataset import CSIRODataset, CSIROTwoStreamDataset


_DATASET_MAP = {
    "original": CSIRODataset,
    "two_stream": CSIROTwoStreamDataset
    #増えたらこれ以降に書きましょうね。
}

def get_datasets(cfg, df, meta_df, val_fold):
    key = cfg.dataset.name.lower()
    if key not in _DATASET_MAP:
        raise ValueError(f"Unsupported dataset: {key}")
    train_df = df.loc[df['fold'] != val_fold].drop("fold", axis=1)
    valid_df = df.loc[df['fold'] == val_fold].drop("fold", axis=1)

    DatasetCls = _DATASET_MAP[key]

    train_ds = DatasetCls(cfg, val_fold, train_df, meta_df= meta_df, mode='train')
    valid_ds = DatasetCls(cfg, val_fold, valid_df, meta_df= meta_df, mode='valid')
    return train_ds, valid_ds