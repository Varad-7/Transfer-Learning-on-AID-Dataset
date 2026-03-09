"""
visualize.py — Plotting utilities for all scenarios.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")


def save_fig(fig, path, dpi=150):
    """Save figure and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(history, title, save_path):
    """Plot train/val loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc", markersize=3)
    ax2.plot(epochs, history["val_acc"], "s-", label="Val Acc", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_pca_embeddings(features, labels, class_names, title, save_path):
    """Plot 2D PCA of feature embeddings."""
    pca = PCA(n_components=2)
    feats_2d = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 10))
    num_classes = len(class_names)
    cmap = plt.cm.get_cmap("tab20", num_classes)

    for i in range(num_classes):
        mask = labels == i
        ax.scatter(
            feats_2d[mask, 0], feats_2d[mask, 1],
            c=[cmap(i)], label=class_names[i],
            alpha=0.6, s=15, edgecolors="none"
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=2)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_accuracy_vs_unfrozen(results, save_path):
    """
    Plot validation accuracy vs percentage of unfrozen parameters.
    results: list of dicts with keys: model_name, strategy, pct_unfrozen, val_acc
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = sorted(set(r["model_name"] for r in results))
    markers = ["o", "s", "D"]
    
    for i, model_name in enumerate(models):
        model_data = [r for r in results if r["model_name"] == model_name]
        model_data.sort(key=lambda x: x["pct_unfrozen"])
        x = [r["pct_unfrozen"] for r in model_data]
        y = [r["val_acc"] for r in model_data]
        labels = [r["strategy"] for r in model_data]
        ax.plot(x, y, f"{markers[i]}-", label=model_name, markersize=8)
        for xi, yi, li in zip(x, y, labels):
            ax.annotate(li, (xi, yi), textcoords="offset points",
                       xytext=(5, 5), fontsize=6)

    ax.set_xlabel("% Unfrozen Parameters")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy vs % Unfrozen Parameters", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_gradient_norms(grad_stats_list, model_names, save_path):
    """Plot gradient norm statistics across layer groups."""
    fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]

    for ax, model_name, grad_stats in zip(axes, model_names, grad_stats_list):
        groups = sorted(grad_stats.keys())
        means = [grad_stats[g]["mean"] for g in groups]
        ax.barh(groups, means, color=sns.color_palette("Set2", len(groups)))
        ax.set_xlabel("Mean Gradient Norm")
        ax.set_title(f"{model_name}")

    fig.suptitle("Gradient Norm Statistics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_convergence_comparison(histories, strategy_names, model_name, save_path):
    """Plot training loss convergence for different strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for hist, name in zip(histories, strategy_names):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label=name, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(f"Convergence Comparison — {model_name}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_few_shot_comparison(results, save_path):
    """
    Plot few-shot results as grouped bar chart.
    results: list of dicts with model_name, fraction, val_acc
    """
    import pandas as pd
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(10, 6))
    fractions = sorted(df["fraction"].unique())
    models = sorted(df["model_name"].unique())
    x = np.arange(len(fractions))
    width = 0.25

    for i, model in enumerate(models):
        vals = [df[(df["model_name"] == model) & (df["fraction"] == f)]["val_acc"].values[0]
                for f in fractions]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Few-Shot Learning: Accuracy vs Data Fraction", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fractions])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_corruption_results(results, save_path):
    """
    Plot corruption robustness results.
    results: list of dicts with model_name, corruption, val_acc, corruption_error, relative_robustness
    """
    import pandas as pd
    df = pd.DataFrame(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    models = sorted(df["model_name"].unique())
    corruptions = sorted(df["corruption"].unique())
    x = np.arange(len(corruptions))
    width = 0.25

    # Corruption error
    for i, model in enumerate(models):
        vals = [df[(df["model_name"] == model) & (df["corruption"] == c)]["corruption_error"].values[0]
                for c in corruptions]
        ax1.bar(x + i * width, vals, width, label=model)

    ax1.set_xlabel("Corruption Type")
    ax1.set_ylabel("Corruption Error")
    ax1.set_title("Corruption Error", fontweight="bold")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(corruptions, rotation=30, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Relative robustness
    for i, model in enumerate(models):
        vals = [df[(df["model_name"] == model) & (df["corruption"] == c)]["relative_robustness"].values[0]
                for c in corruptions]
        ax2.bar(x + i * width, vals, width, label=model)

    ax2.set_xlabel("Corruption Type")
    ax2.set_ylabel("Relative Robustness")
    ax2.set_title("Relative Robustness (Corrupted/Clean)", fontweight="bold")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(corruptions, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Corruption Robustness Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_layerwise_accuracy(results, save_path):
    """
    Plot accuracy vs depth for layer-wise probing.
    results: list of dicts with model_name, layer, accuracy, depth_idx
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = sorted(set(r["model_name"] for r in results))
    markers = ["o", "s", "D"]
    
    for i, model in enumerate(models):
        data = sorted([r for r in results if r["model_name"] == model],
                      key=lambda x: x["depth_idx"])
        x = [r["depth_idx"] for r in data]
        y = [r["accuracy"] for r in data]
        labels = [r["layer"] for r in data]
        ax.plot(x, y, f"{markers[i]}-", label=model, markersize=10, linewidth=2)
        for xi, yi, li in zip(x, y, labels):
            ax.annotate(li, (xi, yi), textcoords="offset points",
                       xytext=(0, 10), fontsize=7, ha="center")

    ax.set_xlabel("Network Depth (Layer Index)")
    ax.set_ylabel("Linear Probe Accuracy (%)")
    ax.set_title("Layer-Wise Feature Probing: Accuracy vs Depth", fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Early", "Middle", "Late"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, save_path)


def plot_feature_norms(norm_stats, save_path):
    """
    Plot feature norm statistics across layers.
    norm_stats: list of dicts with model_name, layer, mean_norm, std_norm
    """
    import pandas as pd
    df = pd.DataFrame(norm_stats)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(df["model_name"].unique())
    layers = ["early", "middle", "late"]
    x = np.arange(len(layers))
    width = 0.25
    
    for i, model in enumerate(models):
        means = []
        stds = []
        for layer in layers:
            row = df[(df["model_name"] == model) & (df["layer"] == layer)]
            means.append(row["mean_norm"].values[0])
            stds.append(row["std_norm"].values[0])
        ax.bar(x + i * width, means, width, yerr=stds, label=model, capsize=3)
    
    ax.set_xlabel("Layer Depth")
    ax.set_ylabel("Feature Norm")
    ax.set_title("Feature Norm Statistics Across Layers", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Early", "Middle", "Late"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig(fig, save_path)
