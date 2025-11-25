# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSIRO Kaggle competition codebase for pasture biomass prediction (image regression task). Originally adapted from MABe mouse behavior classification competition, hence some naming/structure remnants.

**Task**: Predict 5 biomass targets (Dry_Clover_g, Dry_Dead_g, Dry_Green_g, Dry_Total_g, GDM_g) from pasture images
**Evaluation**: Weighted R² (weights: 0.1, 0.1, 0.1, 0.2, 0.5)

## Essential Commands

### Training
```bash
# Run training with default config (local environment)
python kakao/run/train.py

# Override config environment
python kakao/run/train.py dir=colab  # or mac

# Override specific parameters
python kakao/run/train.py trainer.epochs=20 model.name=Spec2DCNN
```

### Data Preparation
```bash
# Generate train_with_split.csv with 5-fold split
python kakao/scripts/create_train_with_split.py
```

### Package Management
```bash
# Sync dependencies
uv sync
```

## Architecture: Modular 3-Layer Design

The codebase uses a **pluggable composition pattern** for experimentation flexibility:

```
Input Image → Feature Extractor → Decoder → Output Logits
```

### Component Locations

- **Feature Extractors**: `kakao/src/models/feature_extractor/`
  - CNNSpectrogram (multi-kernel Conv1D)
  - LSTMFeatureExtractor (LSTM-based)
  - SpecFeatureExtractor (torchaudio spectrogram)
  - PANNsFeatureExtractor (audio features)

- **Decoders**: `kakao/src/models/decoder/`
  - LSTMDecoder (temporal modeling)
  - UNet1DDecoder (with SE blocks)
  - TransformerDecoder (attention-based)
  - MLPDecoder (simple feedforward)

- **Container Models**: `kakao/src/models/`
  - `Spec1D`: Simple pipeline (feature_extractor → decoder)
  - `Spec2DCNN`: Complex pipeline with segmentation_models UNet encoder

### How Components Connect

All assembled via factory function in `kakao/src/models/common.py`:
```python
get_model(cfg, feature_dim, n_pairs, n_actions, num_timesteps)
  ↓ calls
get_feature_extractor(cfg, feature_dim, num_timesteps)
get_decoder(cfg, n_channels, n_pairs, n_actions, num_timesteps)
```

Configuration-driven via Hydra YAML files.

## Data Pipeline Flow

```
train.csv (raw data)
  ↓ create_train_with_split.py
train_with_split.csv (with fold column)
  ↓ CSIRODataModule.setup()
CSIRODataset (loads images + targets from train.csv metadata)
  ↓ DataLoader batching
{'sample_img': (B,H,W,3), 'target': (B,n_classes)}
  ↓ CSIROModel.training_step()
Loss computation + backprop
```

**Key insight**: `train_with_split.csv` contains fold assignments for 5-fold CV. Original `train.csv` contains actual target values matched via `sample_id` prefix.

## Configuration System (Hydra)

Main config: `kakao/run/conf/train.yaml`
Environment configs: `kakao/run/conf/dir/{local,mac,colab}.yaml`

**Key config sections**:
- `model.name`: Selects Spec1D vs Spec2DCNN
- `feature_extractor`: Swappable component config
- `decoder`: Swappable component config
- `model.loss`: Loss function selection (bce/focal/combined)
- `trainer`: Optimizer, LR, epochs, etc.

**Example config modification**:
```yaml
feature_extractor:
  name: LSTMFeatureExtractor
  hidden_size: 128
  bidirectional: true

decoder:
  name: TransformerDecoder
  hidden_size: 256
  nhead: 8
```

## Training Loop (PyTorch Lightning)

Handled by `CSIROModel` (LightningModule) in `kakao/src/modelmodule.py`:

1. `__init__`: Creates model via `get_model()`, instantiates loss function
2. `training_step`: Forward pass + loss computation (Lightning auto-backprops)
3. `validation_step`: Collects outputs for epoch-end aggregation
4. `on_validation_epoch_end`: Computes avg val loss, saves best checkpoint
5. `configure_optimizers`: AdamW/Adam + optional warmup + CosineAnnealing schedulers

**5-fold CV**: Training script runs 5 separate training runs (`val_fold` 0-4), each with different train/val split.

## Loss Functions

Located in `kakao/src/loss/loss.py`:

- **BCEWithLogitsLossWrapper**: Standard BCE with pos_weight support
- **FocalLoss**: Focuses on hard examples (Lin et al. 2017)
- **CombinedLoss**: Weighted sum of multiple losses

**pos_weight modes**:
- `dynamic`: Computed from batch statistics (neg_count / pos_count)
- `fixed`: Static weight value

## Evaluation Metrics

Reference implementation in `kakao/src/utils/metrics.py`:
```python
weighted_r2_score(y_true, y_pred)  # Returns (weighted_r2, individual_r2s)
```

Weights: [0.1, 0.1, 0.1, 0.2, 0.5] for [Green, Dead, Clover, GDM, Total]

## Adding New Components

**New Feature Extractor**:
1. Create class in `kakao/src/models/feature_extractor/`
2. Must output: `(B, out_chans, height, T)` shaped tensor
3. Register in `get_feature_extractor()` in `common.py`
4. Update `train.yaml` with new config

**New Decoder**:
1. Create class in `kakao/src/models/decoder/`
2. Input: `(B, channels, T)` → Output: `(B, T, n_classes)`
3. Register in `get_decoder()` in `common.py`
4. Update `train.yaml`

**New Loss Function**:
1. Add class to `loss.py` (must accept `logits, targets, pos_weight`)
2. Register in `get_loss_function()`
3. Set `model.loss` in config

## Important Quirks

- **MABe Heritage**: Code structure optimized for temporal/sequential data. Current CSIRO task is image regression, so feature_extractor/decoder may need adjustment for spatial (not temporal) processing.

- **Model Shape Mismatch**: Current `get_model()` expects time-series inputs. For image regression, may need to bypass or modify Spec1D/Spec2DCNN.

- **Dataset dual-dataframe**: `CSIRODataset` receives both `train_df` (from train_with_split.csv) for fold info and `meta_df` (from train.csv) for target values. Images matched via `image_id` prefix in `sample_id`.

- **WandB Logging**: Project name set to "CSIRO" in `train.py`. Logs sent to entity "gaiji".

- **Reference Implementation**: Check `REFERENCE/lb-0-57-infer-model-code.ipynb` for alternative approach (two-stream ConvNeXt with multi-head regression).
