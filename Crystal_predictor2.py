import os
import torch
from pathlib import Path
import pandas as pd
from crystals_experiment.crystal_graph import crystal_graph_dataset
from polynet.app.services.model_training import load_gnn_model
import numpy as np
import shutil


def load_ensemble(model_dir, device="cpu"):
    models = {}
    for filename in sorted(os.listdir(model_dir)):
        if filename.endswith(".pt"):
            model_name = filename.replace(".pt", "")
            path = os.path.join(model_dir, filename)
            model = load_gnn_model(path, map_location=device)
            model.eval()
            models[model_name] = model
    return models


def make_inference_dataset(cif_path, dataset_args):
    """
    Creates a temporary dataset containing ONLY the new CIF file.
    """
    temp_dir = Path("temp_inference")
    raw_dir = temp_dir / "raw"
    processed_dir = temp_dir / "processed"

    shutil.rmtree(temp_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True)

    # Copy CIF into dataset raw/
    shutil.copy(cif_path, raw_dir / Path(cif_path).name)

    # Dummy targets file (not used)
    dummy_targets = pd.DataFrame([[Path(cif_path).stem, 0]])

    dataset = crystal_graph_dataset(
        root=temp_dir,
        targets=dummy_targets,
        **dataset_args
    )
    return dataset


def predict_cif(models, dataset, device="cpu"):
    data = dataset[0].to(device)
    predictions = {}

    for name, model in models.items():
        with torch.no_grad():
            y = model(data).item()
            predictions[name] = y

    preds = np.array(list(predictions.values()))
    predictions["ensemble_mean"] = preds.mean()
    predictions["ensemble_std"] = preds.std()

    return predictions


def main():
    # 1. PATHS
    print("Test")
    cif_path = "crystals_test/data/raw/c1.cif"
    model_dir = "crystals_experiment/results/models"

    print(model_dir)

    # 2. Dataset parameters (must match training!)
    dataset_args = dict(
        max_d=3.5,
        step=0.5,
        vor_cut_off=3.5,
    )

    # 3. Load models
    print("Loading models...")
    models = load_ensemble(model_dir)

    # 4. Prepare dataset for new CIF
    print("Processing CIF...")
    dataset = make_inference_dataset(cif_path, dataset_args)

    # 5. Predict
    print("Predicting...")
    results = predict_cif(models, dataset)

    print("\n--- Predictions ---")
    for name, value in results.items():
        print(f"{name:20s}: {value}")


if __name__ == "__main__":
    main()
