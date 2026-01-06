import pandas as pd
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
df = pd.read_csv(data_dir / "integrated_train.csv")

print("=== Sampling_Dateごとのサンプル数分布 ===")
sampling_date_counts = df.groupby('Sampling_Date').size().sort_values(ascending=False)
print(sampling_date_counts.describe())
print("\nTop 10 largest groups:")
print(sampling_date_counts.head(10))
print(f"\nTotal unique Sampling_Dates: {len(sampling_date_counts)}")

print("\n=== strataの分布 ===")
strata_counts = df['strata'].value_counts().sort_index()
print(strata_counts)

print("\n=== Foldごとの統計 ===")
for fold in range(5):
    fold_df = df[df['fold'] == fold]
    print(f"\nFold {fold} (n={len(fold_df)}):")
    print(f"  Unique Sampling_Dates: {fold_df['Sampling_Date'].nunique()}")
    print(f"  strata distribution:")
    for strata, count in fold_df['strata'].value_counts().sort_index().items():
        print(f"    {strata}: {count}")
