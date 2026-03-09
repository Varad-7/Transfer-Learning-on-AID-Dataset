"""
Scenario 1: Linear Probe Transfer
- Freeze all backbone parameters
- Train only the linear classifier head
- Report: training/val curves, confusion matrix, PCA embeddings
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders, seed_everything, SEED
from src.models import get_model, freeze_backbone, get_model_info, count_parameters, extract_features, get_device
from src.train_utils import train_model, get_final_predictions
from src.visualize import plot_training_curves, plot_confusion_matrix, plot_pca_embeddings

import torch
import numpy as np


def run_linear_probe(model_name, data_dir, output_dir, num_epochs=30, batch_size=32):
    """Run linear probe for a single model."""
    print(f"\n{'='*60}")
    print(f"  Linear Probe: {model_name}")
    print(f"{'='*60}")
    
    seed_everything(SEED)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0
    )
    print(f"Classes: {len(class_names)}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model, config = get_model(model_name, num_classes=len(class_names))
    freeze_backbone(model)

    # Print model info
    info = get_model_info(model, model_name)
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,}, Trainable: {trainable:,}")
    print(f"MACs: {info['MACs']:.2e}, FLOPs: {info['FLOPs']:.2e}")

    # Train
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, lr=1e-3
    )

    # Final evaluation
    final_acc, preds, labels = get_final_predictions(model, val_loader, device)
    print(f"\nFinal Val Accuracy: {final_acc:.2f}%")

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    
    plot_training_curves(
        history, f"Linear Probe — {model_name}",
        os.path.join(output_dir, f"s1_{model_name}_curves.png")
    )
    plot_confusion_matrix(
        labels, preds, class_names,
        f"Confusion Matrix — {model_name} (Linear Probe)",
        os.path.join(output_dir, f"s1_{model_name}_confusion.png")
    )

    # Extract features for PCA
    print(f"  Extracting features for PCA...")
    feats_layer = config["feature_layers"]["late"]
    features, feat_labels = extract_features(model, val_loader, feats_layer, device)
    plot_pca_embeddings(
        features, feat_labels, class_names,
        f"PCA Embeddings — {model_name} (Linear Probe)",
        os.path.join(output_dir, f"s1_{model_name}_pca.png")
    )

    return {
        "model_name": model_name,
        "final_val_acc": final_acc,
        "total_params": total,
        "trainable_params": trainable,
        "MACs": info["MACs"],
        "FLOPs": info["FLOPs"],
        "history": history,
    }


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_1")

    models = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    results = []

    for model_name in models:
        result = run_linear_probe(model_name, data_dir, output_dir, num_epochs=30, batch_size=32)
        results.append(result)
        # Free GPU memory
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SCENARIO 1: LINEAR PROBE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Val Acc (%)':<14} {'Total Params':<16} {'Trainable':<14} {'MACs':<12} {'FLOPs':<12}")
    print("-" * 88)
    for r in results:
        print(f"{r['model_name']:<20} {r['final_val_acc']:<14.2f} {r['total_params']:<16,} "
              f"{r['trainable_params']:<14,} {r['MACs']:<12.2e} {r['FLOPs']:<12.2e}")

    # Save results JSON
    save_results = [{k: v for k, v in r.items() if k != "history"} for r in results]
    with open(os.path.join(output_dir, "s1_results.json"), "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
