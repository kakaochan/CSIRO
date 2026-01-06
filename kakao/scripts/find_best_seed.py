import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# パス設定
data_dir = Path(__file__).parent.parent / "data"
train_csv_path = data_dir / "train.csv"

train_df = pd.read_csv(train_csv_path)

unique_id = train_df["sample_id"].str.split("__").str[0].unique()

list_of_subdf = []
for id in unique_id:
    sub_df = train_df[train_df["sample_id"].str.startswith(id)]
    sub_df_dict = {
        "image_id"      : id,
        "Sampling_Date" : sub_df.iloc[0]["Sampling_Date"],
        "State"         : sub_df.iloc[0]["State"],
        "Species"       : sub_df.iloc[0]["Species"],
        "Pre_GSHH_NDVI" : sub_df.iloc[0]["Pre_GSHH_NDVI"],
        "Height_Ave_cm" : sub_df.iloc[0]["Height_Ave_cm"],
        "Dry_Clover_g"  : sub_df.iloc[0]["target"],
        "Dry_Dead_g"    : sub_df.iloc[1]["target"],
        "Dry_Green_g"   : sub_df.iloc[2]["target"],
        "Dry_Total_g"   : sub_df.iloc[3]["target"],
        "GDM_g"         : sub_df.iloc[4]["target"],
        "Dead_per_Total": sub_df.iloc[1]["target"] / sub_df.iloc[3]["target"]
    }
    list_of_subdf.append(sub_df_dict)
new_train_df = pd.DataFrame(list_of_subdf)

# 層別化変数の準備
new_train_df['Sampling_Date_dt'] = pd.to_datetime(new_train_df['Sampling_Date'])
new_train_df['month'] = new_train_df['Sampling_Date_dt'].dt.month
new_train_df['quarter'] = ((new_train_df['month'] - 1) // 3 + 1).astype(str)

# 月別と四半期別の両方を試す
stratification_types = {
    'monthly': new_train_df['month'].astype(str) + '_' + new_train_df['State'],
    'quarterly': 'Q' + new_train_df['quarter'] + '_' + new_train_df['State']
}

for strat_name, strata_values in stratification_types.items():
    print(f"\n{'='*60}")
    print(f"Testing {strat_name.upper()} stratification (seeds 0-999)...")
    print(f"{'='*60}")

    best_seed = None
    best_std = float('inf')
    results = []

    for seed in tqdm(range(0, 1000)):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_assignment = np.full(len(new_train_df), -1)

        groups = new_train_df['Sampling_Date']
        strata = strata_values

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(new_train_df, y=strata, groups=groups)):
            fold_assignment[val_idx] = fold

        # Fold分布の標準偏差を計算
        fold_counts = pd.Series(fold_assignment).value_counts().sort_index().values
        fold_std = np.std(fold_counts)

        results.append({'seed': seed, 'std': fold_std, 'counts': list(fold_counts)})

        if fold_std < best_std:
            best_std = fold_std
            best_seed = seed

    # 上位10個を表示
    results_df = pd.DataFrame(results)
    print(f"\n=== Top 10 most balanced seeds ({strat_name}) ===")
    for idx, row in results_df.nsmallest(10, 'std').iterrows():
        counts_str = ', '.join([f'{c:2d}' for c in row['counts']])
        print(f"seed={row['seed']:3d}, std={row['std']:5.2f}, counts=[{counts_str}]")

    print(f"\n=== Best seed ({strat_name}) ===")
    print(f"seed={best_seed}, std={best_std:.2f}")
