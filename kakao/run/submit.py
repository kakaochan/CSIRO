"""Kaggle submission script for CSIRO pasture biomass prediction."""

import sys
from pathlib import Path
import hydra
import pandas as pd
import torch
from torch.utils.data import DataLoader




@hydra.main(config_path="conf", config_name="submit", version_base="1.3")
def main(cfg):
    # submit.py の親ディレクトリ（kakao/）を Python パスに追加
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))  # kakao/ をパスに追加
    from src.dataset.common import get_test_dataset
    from src.inference import load_test_models, run_inference, create_submission, load_state_model
    """Main submission pipeline.

    Steps:
    1. Load test data
    2. Load trained models
    3. Run inference
    4. Create submission.csv
    """

    print("=" * 60)
    print("CSIRO Kaggle Submission Pipeline")
    print("=" * 60)

    # ========================================
    # 1. Load test data
    # ========================================
    print("\n[1/4] Loading test data...")
    test_csv_path = Path(cfg.kaggle.test_csv)

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    test_df = pd.read_csv(test_csv_path)
    print(f"   Loaded {len(test_df)} rows from test.csv")

    # Create dataset
    test_dataset = get_test_dataset(cfg, test_df)
    print(f"   Created dataset ({cfg.dataset.name}) with {len(test_dataset)} unique images")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.inference.batch_size,
        shuffle=False,
        num_workers=cfg.inference.num_workers,
        pin_memory=True
    )
    print(f"   DataLoader ready (batch_size={cfg.inference.batch_size})")

    # ========================================
    # 2. Load trained models
    # ========================================
    print("\n[2/4] Loading trained models...")

    if cfg.inference.ensemble_folds:
        models, scalers = load_test_models(cfg)
        print(f"   Loaded {len(models)} models for ensemble")
    else:
        # Load only first model
        raise NotImplementedError("Single model inference not implemented yet")
    
    if cfg.postprocess.enabled:
        state_model = load_state_model(cfg)
    else:
        state_model = None

    # ========================================
    # 3. Run inference
    # ========================================
    print("\n[3/4] Running inference...")

    device = cfg.inference.device
    print(f"  Using device: {device}")

    predictions, image_ids = run_inference(
        cfg=cfg,
        models=models,
        test_loader=test_loader,
        device=device,
        scalers=scalers,
        ensemble=cfg.inference.ensemble_folds,
        state_model=state_model,
    )

    print(f"   Predictions shape: {predictions.shape}")

    # ========================================
    # 4. Create submission
    # ========================================
    print("\n[4/4] Creating submission CSV...")

    submission_df = create_submission(
        predictions=predictions,
        image_ids=image_ids,
        output_path=cfg.kaggle.submission_csv
    )

    print("\n" + "=" * 60)
    print(" Submission pipeline completed successfully!")
    print("=" * 60)
    print(f"\nSubmission file: {cfg.kaggle.submission_csv}")
    print(f"Ready for Kaggle submission!")


if __name__ == "__main__":
    main()
