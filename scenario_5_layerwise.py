"""
Scenario 5: Layer-Wise Feature Probing
- Extract features from early, middle, and late layers
- Train separate linear classifiers on each
- Report: accuracy vs depth, feature norms, PCA plots
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (get_dataloaders, get_fixed_subset_for_pca,
                          seed_everything, SEED, get_val_transform, TransformedSubset)
from src.models import (get_model, freeze_backbone, get_model_info, count_parameters,
                         extract_features, get_device, MODEL_CONFIGS)
from src.visualize import plot_layerwise_accuracy, plot_feature_norms, plot_pca_embeddings

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def train_linear_probe_sklearn(train_features, train_labels, val_features, val_labels):
    """Train a linear classifier using scikit-learn LogisticRegression."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_val = scaler.transform(val_features)

    clf = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
    clf.fit(X_train, train_labels)

    train_acc = clf.score(X_train, train_labels) * 100
    val_acc = clf.score(X_val, val_labels) * 100
    return train_acc, val_acc


def run_layerwise_probing(model_name, data_dir, output_dir, device, batch_size=32):
    """Run layer-wise feature probing for a single model."""
    print(f"\n{'='*60}")
    print(f"  Layer-Wise Feature Probing: {model_name}")
    print(f"{'='*60}")

    seed_everything(SEED)

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0
    )

    # Create model (frozen backbone)
    model, config = get_model(model_name, num_classes=len(class_names))
    freeze_backbone(model)
    model.to(device)
    model.eval()

    info = get_model_info(model, model_name)
    total, _ = count_parameters(model)
    print(f"  Params: {total:,}, MACs: {info['MACs']:.2e}, FLOPs: {info['FLOPs']:.2e}")

    # Layer config
    feature_layers = config["feature_layers"]
    layer_names = ["early", "middle", "late"]
    results = []
    norm_stats = []

    for depth_idx, layer_key in enumerate(layer_names):
        layer_name = feature_layers[layer_key]
        print(f"\n  Extracting features from {layer_key} layer: {layer_name}")

        # Extract train features
        train_feats, train_labels = extract_features(
            model, train_loader, layer_name, device
        )
        # Extract val features
        val_feats, val_labels = extract_features(
            model, val_loader, layer_name, device
        )
        print(f"    Feature shape: {train_feats.shape}")

        # Feature norm statistics
        norms = np.linalg.norm(val_feats, axis=1)
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        print(f"    Feature norm — mean: {mean_norm:.4f}, std: {std_norm:.4f}")

        norm_stats.append({
            "model_name": model_name,
            "layer": layer_key,
            "layer_name": layer_name,
            "mean_norm": mean_norm,
            "std_norm": std_norm,
        })

        # Train linear probe
        print(f"    Training linear classifier...")
        train_acc, val_acc = train_linear_probe_sklearn(
            train_feats, train_labels, val_feats, val_labels
        )
        print(f"    Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        results.append({
            "model_name": model_name,
            "layer": layer_key,
            "layer_name": layer_name,
            "depth_idx": depth_idx,
            "accuracy": val_acc,
            "train_accuracy": train_acc,
            "feature_dim": train_feats.shape[1],
        })

    return results, norm_stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scenario 5: Layer-Wise Feature Probing")
    parser.add_argument("--model", type=str, default="all", help="Model to run: resnet50, efficientnet_b0, convnext_tiny, or 'all'")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_5")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    if args.model == "all":
        models = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    else:
        models = [args.model]

    all_results = []
    all_norm_stats = []

    for model_name in models:
        results, norm_stats = run_layerwise_probing(
            model_name, data_dir, output_dir, device, batch_size=args.batch_size
        )
        all_results.extend(results)
        all_norm_stats.extend(norm_stats)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Plot accuracy vs depth
    plot_layerwise_accuracy(all_results, os.path.join(output_dir, "s5_acc_vs_depth.png"))
    
    # Plot feature norms
    plot_feature_norms(all_norm_stats, os.path.join(output_dir, "s5_feature_norms.png"))

    # PCA plots for each model and layer (fixed subset)
    print("\n  Generating PCA plots on fixed subset...")
    selected_indices, full_dataset = get_fixed_subset_for_pca(data_dir, samples_per_class=30)
    val_transform = get_val_transform()
    pca_dataset = TransformedSubset(full_dataset, selected_indices, val_transform)
    pca_loader = DataLoader(pca_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pca_labels = np.array([full_dataset.targets[i] for i in selected_indices])
    class_names = full_dataset.classes

    for model_name in models:
        model, config = get_model(model_name, num_classes=len(class_names))
        freeze_backbone(model)
        model.to(device)
        model.eval()

        for layer_key in ["early", "middle", "late"]:
            layer_name = config["feature_layers"][layer_key]
            feats, labels = extract_features(model, pca_loader, layer_name, device)
            plot_pca_embeddings(
                feats, labels, class_names,
                f"PCA - {model_name} ({layer_key}: {layer_name})",
                os.path.join(output_dir, f"s5_{model_name}_{layer_key}_pca.png")
            )

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Summary table
    print(f"\n{'='*100}")
    print(f"  SCENARIO 5: LAYER-WISE FEATURE PROBING SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<18} {'Layer':<10} {'Layer Name':<20} {'Val Acc (%)':<14} {'Feat Dim':<10} {'Norm Mean':<12} {'Norm Std':<12}")
    print("-" * 96)
    for r in all_results:
        ns = [n for n in all_norm_stats if n["model_name"] == r["model_name"] and n["layer"] == r["layer"]][0]
        print(f"{r['model_name']:<18} {r['layer']:<10} {r['layer_name']:<20} {r['accuracy']:<14.2f} "
              f"{r['feature_dim']:<10} {ns['mean_norm']:<12.4f} {ns['std_norm']:<12.4f}")

    # Save results
    s5_res_path = os.path.join(output_dir, "s5_results.json")
    save_results = {"accuracy_results": all_results, "norm_stats": all_norm_stats}
    if os.path.exists(s5_res_path) and args.model != "all":
        with open(s5_res_path, "r") as f:
            existing = json.load(f)
            existing["accuracy_results"].extend(all_results)
            existing["norm_stats"].extend(all_norm_stats)
            save_results = existing
            
    with open(s5_res_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
