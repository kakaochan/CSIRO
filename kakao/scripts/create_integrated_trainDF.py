import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
# パス設定
data_dir = Path(__file__).parent.parent / "data"
train_csv_path = data_dir / "train.csv"
output_csv_path = data_dir / "integrated_train.csv"

train_df = pd.read_csv(train_csv_path)

unique_id = train_df["sample_id"].str.split("__").str[0].unique()

list_of_subdf = []
for id in tqdm(unique_id):
    sub_df = train_df[train_df["sample_id"].str.startswith(id)]
    sub_df_dict = {
        "image_id"      : id,  # image_id追加
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
        "Dead_per_Total": sub_df.iloc[1]["target"] / sub_df.iloc[3]["target"],
        "Green_per_Total": sub_df.iloc[2]['target'] / sub_df.iloc[3]['target'],
        'Clover_per_Total': sub_df.iloc[0]['target'] / sub_df.iloc[3]['target'],
        'state_target': 0 if sub_df.iloc[0]["State"] == 'NSW' else (1 if sub_df.iloc[0]["State"] == 'WA' else 2),
    }
    list_of_subdf.append(sub_df_dict)
new_train_df = pd.DataFrame(list_of_subdf)

# 月とStateで層別化するための前処理
new_train_df['Sampling_Date_dt'] = pd.to_datetime(new_train_df['Sampling_Date'])
new_train_df['month'] = new_train_df['Sampling_Date_dt'].dt.month
new_train_df['strata'] = new_train_df['month'].astype(str) + '_' + new_train_df['State']

seed = 335
# 5-Fold分割追加（StratifiedGroupKFold: Sampling_Dateでグループ化、月×Stateで層別化）
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
new_train_df['fold'] = -1

groups = new_train_df['Sampling_Date']
strata = new_train_df['strata']
for fold, (train_idx, val_idx) in enumerate(sgkf.split(new_train_df, y=strata, groups=groups)):
    new_train_df.loc[val_idx, 'fold'] = fold

# State分類用の80/20分割（state_fold列）
state_seed = 131
sgkf_state = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=state_seed)
new_train_df['state_fold'] = -1  # 全てTrainで初期化

groups_state = new_train_df['Sampling_Date']
strata_state = new_train_df['State']  # Stateのみで層別化
for fold, (train_idx, val_idx) in enumerate(sgkf_state.split(new_train_df, y=strata_state, groups=groups_state)):
    if fold == 4:  # 最後の1つをValに
        new_train_df.loc[val_idx, 'state_fold'] = 0
    # 他は-1（Train）のまま

new_train_df.to_csv(output_csv_path, index=False)
print(f"\nFold distribution (バイオマス用):")
print(f'seed={seed}')
print(new_train_df['fold'].value_counts().sort_index())

print(f"\nState_fold distribution (State分類用):")
print(f'state_seed={state_seed}')
print(new_train_df['state_fold'].value_counts().sort_index())
print(f"\nState distribution in state_fold=0 (Val):")
print(new_train_df[new_train_df['state_fold']==0]['State'].value_counts())
