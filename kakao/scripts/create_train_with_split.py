import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

# パス設定
data_dir = Path(__file__).parent.parent / "data"
train_csv_path = data_dir / "train.csv"
output_csv_path = data_dir / "train_with_split.csv"

# train.csvを読み込み
df = pd.read_csv(train_csv_path)

# sample_idから画像IDを抽出
df['image_id'] = df['sample_id'].str.split('__').str[0]

# 必要な列のみ抽出してユニーク化（Sampling_Dateも追加）
df_unique = df[['image_id', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']].drop_duplicates(subset=['image_id']).reset_index(drop=True)

# 5-Fold作成（GroupKFold: Sampling_Dateでグループ化）
gkf = GroupKFold(n_splits=5)
df_unique['fold'] = -1

groups = df_unique['Sampling_Date']
for fold, (train_idx, val_idx) in enumerate(gkf.split(df_unique, groups=groups)):
    df_unique.loc[val_idx, 'fold'] = fold

# 保存
df_unique.to_csv(output_csv_path, index=False)
print(f"Saved to {output_csv_path}")
print(f"Total samples: {len(df_unique)}")
print(f"\nFold distribution:")
print(df_unique['fold'].value_counts().sort_index())
