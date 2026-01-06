"""Inference utilities for Kaggle submission."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib


def load_test_models(cfg):
    """Load all trained models for ensemble prediction.

    Args:
        cfg: Configuration object with model paths

    Returns:
        tuple: (models, scalers) where models is list of loaded models and scalers is list of scalers
    """
    from src.inference_model import CSIROInferenceModel
    from omegaconf import OmegaConf

    models = []
    scalers = []
    model_dir = Path(cfg.kaggle.model_dir)

    # Determine device
    if cfg.inference.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.inference.device

    print(f"Using device: {device}")

    # Check if target normalization is enabled
    use_normalization = cfg.get('use_target_normalization', False)
    if use_normalization:
        print("Target normalization: ENABLED")
    else:
        print("Target normalization: DISABLED (using raw scale)")

    for i, model_file in enumerate(cfg.kaggle.model_files):
        model_path = model_dir / model_file

        # Load scaler if normalization is enabled
        target_mean = None
        target_std = None
        scaler = None

        if use_normalization:
            scaler_file = cfg.kaggle.scaler_files[i]
            scaler_path = model_dir / scaler_file
            print(f"Loading scaler {i}: {scaler_path}")

            scaler = joblib.load(scaler_path)
            target_mean = scaler.mean_
            target_std = scaler.scale_

            print(f"  Mean: {target_mean}")
            print(f"  Std:  {target_std}")

        # Create config for model architecture
        model_cfg = OmegaConf.create({
            'model': cfg.model,
            'feature_extractor': cfg.feature_extractor,
            'decoder': cfg.decoder,
            'augmentation': {'mixup_alpha': 0.0, 'cutmix_alpha': 0.0}
        })

        # Load model (Lightning-free)
        print(f"Loading model {i}: {model_path}")
        model = CSIROInferenceModel(model_cfg, target_mean=target_mean, target_std=target_std)

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
        scalers.append(scaler)

        print(f"  ✓ Loaded successfully on {device}")

    return models, scalers


def run_inference(models, test_loader, device, scalers=None, ensemble=True):
    """Run inference on test data.

    Args:
        models: List of models (for ensemble) or single model
        test_loader: DataLoader for test data
        device: Device to run inference on
        scalers: List of scalers (for denormalization) or None
        ensemble: Whether to ensemble multiple models

    Returns:
        predictions: numpy array of shape (N, 5) - predictions for all images (real scale)
        image_ids: list of image IDs corresponding to predictions
    """
    if not isinstance(models, list):
        models = [models]

    if scalers is not None and not isinstance(scalers, list):
        scalers = [scalers]

    all_predictions = []
    all_image_ids = []

    print(f"\nRunning inference on {len(test_loader)} batches...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            image_ids = batch['image_id']

            # Check if Two-Stream or Original
            if 'img_left' in batch and 'img_right' in batch:
                # Two-Stream
                img_left = batch['img_left'].to(device)
                img_right = batch['img_right'].to(device)

                batch_preds = []
                for i, model in enumerate(models):
                    outputs = model(img_left, img_right)
                    logits = outputs['logits']  # (B, 5) - normalized space if scaler exists
                    logits_np = logits.cpu().numpy()

                    # Denormalize if scaler exists
                    if scalers is not None and scalers[i] is not None:
                        logits_np = scalers[i].inverse_transform(logits_np)  # (B, 5) - real scale

                    batch_preds.append(logits_np)
            else:
                # Original
                images = batch['sample_img'].to(device)

                batch_preds = []
                for i, model in enumerate(models):
                    outputs = model(images)
                    logits = outputs['logits']  # (B, 5) - normalized space if scaler exists
                    logits_np = logits.cpu().numpy()

                    # Denormalize if scaler exists
                    if scalers is not None and scalers[i] is not None:
                        logits_np = scalers[i].inverse_transform(logits_np)  # (B, 5) - real scale

                    batch_preds.append(logits_np)

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
