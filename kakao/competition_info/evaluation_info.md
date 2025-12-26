# 評価指標と提出形式まとめ

## 1. 評価指標：重み付き R²（Weighted R²）

このコンペでは、ターゲットごとに個別の R² を計算して平均するのではなく、全サンプル（画像 × 5ターゲット）をまとめて、一度に重み付き R² を計算する。各行にはターゲットに応じた重みが付与される。

---

## 2. ターゲットごとの重み

| ターゲット名 | 重み |
|--------------|------|
| Dry_Green_g  | 0.1  |
| Dry_Dead_g   | 0.1  |
| Dry_Clover_g | 0.1  |
| GDM_g        | 0.2  |
| Dry_Total_g  | 0.5  |

---

## 3. 重み付き R² の計算式

### R²

\[
R^2 = 1 - \frac{SSR}{SST}
\]

### SSR（Residual Sum of Squares）

\[
SSR = \sum_i w_i (y_i - \hat{y}_i)^2
\]

### SST（Total Sum of Squares）

\[
SST = \sum_i w_i (y_i - \bar{y}_w)^2
\]

- \(y_i\)：正解値  
- \(\hat{y}_i\)：予測値  
- \(w_i\)：ターゲット別の重み  
- \(\bar{y}_w\)：重み付き平均値

---

## 4. 提出ファイル形式（Submission Format）

- CSV形式  
- 必須カラム  
  - `sample_id`  
  - `target`

### sample_id の形式

