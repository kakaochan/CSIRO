import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
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
        "Dead_per_Total": sub_df.iloc[1]["target"] / sub_df.iloc[3]["target"]
    }
    list_of_subdf.append(sub_df_dict)
new_train_df = pd.DataFrame(list_of_subdf)

# 5-Fold分割追加（GroupKFold: Sampling_Dateでグループ化）
gkf = GroupKFold(n_splits=5)
new_train_df['fold'] = -1

groups = new_train_df['Sampling_Date']
for fold, (train_idx, val_idx) in enumerate(gkf.split(new_train_df, groups=groups)):
    new_train_df.loc[val_idx, 'fold'] = fold

new_train_df.to_csv(output_csv_path, index=False)
print(f"\nFold distribution:")
print(new_train_df['fold'].value_counts().sort_index())
