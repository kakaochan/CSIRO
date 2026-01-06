from src.dataset.dataset import (
    CSIRODataset,
    CSIROTwoStreamDataset,
    CSIROTestDataset,
    CSIROTwoStreamTestDataset,
)


_DATASET_MAP = {
    "original": CSIRODataset,
    "two_stream": CSIROTwoStreamDataset
    #増えたらこれ以降に書きましょうね。
}

_TEST_DATASET_MAP = {
    "original": CSIROTestDataset,
    "two_stream": CSIROTwoStreamTestDataset
    #増えたらこれ以降に書きましょうね。
}

def get_datasets(cfg, df, val_fold, scaler=None):
    key = cfg.dataset.name.lower()
    if key not in _DATASET_MAP:
        raise ValueError(f"Unsupported dataset: {key}")
    train_df = df.loc[df['fold'] != val_fold].drop("fold", axis=1)
    valid_df = df.loc[df['fold'] == val_fold].drop("fold", axis=1)

    DatasetCls = _DATASET_MAP[key]

    train_ds = DatasetCls(cfg, val_fold, train_df, mode='train', scaler=scaler)
    valid_ds = DatasetCls(cfg, val_fold, valid_df, mode='valid', scaler=scaler)
    return train_ds, valid_ds


def get_test_dataset(cfg, test_df):
    """Factory function to get test dataset based on config.

    Args:
        cfg: Configuration object with dataset.name
        test_df: Test dataframe

    Returns:
        Test dataset instance
    """
    key = cfg.dataset.name.lower()
    if key not in _TEST_DATASET_MAP:
        raise ValueError(f"Unsupported test dataset: {key}")

    TestDatasetCls = _TEST_DATASET_MAP[key]
    return TestDatasetCls(cfg, test_df)