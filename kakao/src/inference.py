"""Inference utilities for Kaggle submission."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_test_models(cfg):
    """Load all trained models for ensemble prediction.

    Args:
        cfg: Configuration object with model paths

    Returns:
        List of loaded models in eval mode
    """
    from src.inference_model import CSIROInferenceModel
    from omegaconf import OmegaConf

    models = []
    model_dir = Path(cfg.kaggle.model_dir)

    # Determine device
    if cfg.inference.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.inference.device

    print(f"Using device: {device}")

    for i, model_file in enumerate(cfg.kaggle.model_files):
        model_path = model_dir / model_file

        # Create config for model architecture
        model_cfg = OmegaConf.create({
            'model': cfg.model,
            'feature_extractor': cfg.feature_extractor,
            'decoder': cfg.decoder,
            'augmentation': {'mixup_alpha': 0.0, 'cutmix_alpha': 0.0}
        })

        # Load model (Lightning-free)
        print(f"Loading model {i}: {model_path}")
        model = CSIROInferenceModel(model_cfg)

        # Load state dict from checkpoint
        state_dict = torch.load(model_path, map_location=device)

        # Remove 'net.' prefix from keys (Lightning saves with this prefix)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.'):
                cleaned_state_dict[key[4:]] = value  # Remove 'net.' prefix
            else:
                cleaned_state_dict[key] = value

        model.net.load_state_dict(cleaned_state_dict, strict=False)

        # Set to eval mode and move to device
        model.eval()
        model = model.to(device)
        models.append(model)

        print(f"  ✓ Loaded successfully on {device}")

    return models


def run_inference(models, test_loader, device, ensemble=True):
    """Run inference on test data.

    Args:
        models: List of models (for ensemble) or single model
        test_loader: DataLoader for test data
        device: Device to run inference on
        ensemble: Whether to ensemble multiple models

    Returns:
        predictions: numpy array of shape (N, 5) - predictions for all images
        image_ids: list of image IDs corresponding to predictions
    """
    if not isinstance(models, list):
        models = [models]

    all_predictions = []
    all_image_ids = []

    print(f"\nRunning inference on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch['sample_img'].to(device)
            image_ids = batch['image_id']

            batch_preds = []

            # Get predictions from each model
            for model in models:
                outputs = model(images)
                logits = outputs['logits']  # (B, 5)
                batch_preds.append(logits.cpu().numpy())

            # Ensemble (average) if multiple models
            if ensemble and len(models) > 1:
                batch_pred = np.mean(batch_preds, axis=0)  # (B, 5)
            else:
                batch_pred = batch_preds[0]

            all_predictions.append(batch_pred)
            all_image_ids.extend(image_ids)

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)  # (N, 5)

    print(f"✓ Inference complete: {len(predictions)} images predicted")

    return predictions, all_image_ids


def create_submission(predictions, image_ids, output_path):
    """Create submission CSV in Kaggle format.

    Args:
        predictions: numpy array of shape (N, 5) - predictions for all images
        image_ids: list of image IDs
        output_path: path to save submission.csv

    Returns:
        submission_df: pandas DataFrame with submission
    """
    target_names = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']

    rows = []
    for img_id, pred_vals in zip(image_ids, predictions):
        for target_name, pred_val in zip(target_names, pred_vals):
            sample_id = f"{img_id}__{target_name}"
            rows.append({
                'sample_id': sample_id,
                'target': pred_val
            })

    submission_df = pd.DataFrame(rows)

    # Save to CSV
    submission_df.to_csv(output_path, index=False)

    print(f"\n✓ Submission saved to: {output_path}")
    print(f"  Total rows: {len(submission_df)}")
    print(f"\nFirst 10 rows:")
    print(submission_df.head(10))

    return submission_df
