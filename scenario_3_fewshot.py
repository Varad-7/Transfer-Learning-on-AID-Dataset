"""
Scenario 3: Few-Shot Learning Analysis
- Train on 100%, 20%, 5% of data
- Report: validation accuracy, relative drop, train-val gap
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders, seed_everything, SEED
from src.models import get_model, get_model_info, count_parameters, get_device
from src.train_utils import train_model, get_final_predictions
from src.visualize import plot_few_shot_comparison, plot_training_curves

import torch


FRACTIONS = [1.0, 0.2, 0.05]
EPOCH_MAP = {1.0: 30, 0.2: 20, 0.05: 20}  # Max epochs per fraction


def run_few_shot(model_name, fraction, data_dir, device, batch_size=32):
    """Run training with a specific data fraction."""
    num_epochs = EPOCH_MAP[fraction]
    print(f"\n  Fraction: {int(fraction*100)}%, Epochs: {num_epochs}")
    
    seed_everything(SEED)

    frac = fraction if fraction < 1.0 else None
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0,
        few_shot_fraction=frac
    )
    print(f"    Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    model, config = get_model(model_name, num_classes=len(class_names))
    
    # Print model info
    info = get_model_info(model, model_name)
    total, trainable = count_parameters(model)
    print(f"    Params: {total:,}, MACs: {info['MACs']:.2e}, FLOPs: {info['FLOPs']:.2e}")

    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, lr=1e-3
    )

    final_acc, _, _ = get_final_predictions(model, val_loader, device)
    
    # Train-val gap
    final_train_acc = history["train_acc"][-1]
    train_val_gap = final_train_acc - final_acc

    return {
        "model_name": model_name,
        "fraction": fraction,
        "val_acc": final_acc,
        "train_acc": final_train_acc,
        "train_val_gap": train_val_gap,
        "train_samples": len(train_loader.dataset),
        "history": history,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scenario 3: Few-Shot Learning Analysis")
    parser.add_argument("--model", type=str, default="all", help="Model to run: resnet50, efficientnet_b0, convnext_tiny, or 'all'")
    parser.add_argument("--epochs_max", type=int, default=20, help="Max epochs for 100 percent data")
    parser.add_argument("--epochs_few", type=int, default=10, help="Max epochs for 20 percent and 5 percent data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    # Update global EPOCH_MAP
    global EPOCH_MAP
    EPOCH_MAP = {1.0: args.epochs_max, 0.2: args.epochs_few, 0.05: args.epochs_few}

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_3")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    if args.model == "all":
        models = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    else:
        models = [args.model]

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Few-Shot Learning: {model_name}")
        print(f"{'='*60}")

        model_results = []
        for fraction in FRACTIONS:
            result = run_few_shot(model_name, fraction, data_dir, device, batch_size=args.batch_size)
            model_results.append(result)
            all_results.append(result)

            # Plot training curves
            plot_training_curves(
                result["history"],
                f"{model_name} - {int(fraction*100)}% Data",
                os.path.join(output_dir, f"s3_{model_name}_{int(fraction*100)}pct_curves.png")
            )

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Compute relative drop
        acc_100 = [r for r in model_results if r["fraction"] == 1.0][0]["val_acc"]
        acc_5 = [r for r in model_results if r["fraction"] == 0.05][0]["val_acc"]
        rel_drop = (acc_100 - acc_5) / acc_100 if acc_100 > 0 else 0
        print(f"\n  Relative Drop (Δ) for {model_name}: {rel_drop:.4f} ({rel_drop*100:.1f}%)")

    # Plot comparison
    plot_data = [{"model_name": r["model_name"], "fraction": r["fraction"], "val_acc": r["val_acc"]}
                 for r in all_results]
    plot_few_shot_comparison(plot_data, os.path.join(output_dir, "s3_few_shot_comparison.png"))

    # Summary tables
    print(f"\n{'='*100}")
    print(f"  SCENARIO 3: FEW-SHOT LEARNING SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<18} {'Fraction':<10} {'Val Acc (%)':<14} {'Train Acc (%)':<14} {'Train-Val Gap':<14} {'Samples':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model_name']:<18} {int(r['fraction']*100)}%{'':<7} {r['val_acc']:<14.2f} "
              f"{r['train_acc']:<14.2f} {r['train_val_gap']:<14.2f} {r['train_samples']:<10}")

    # Relative drop table
    print(f"\n{'Model':<18} {'Acc_100%':<12} {'Acc_5%':<12} {'Relative Drop (Δ)':<18}")
    print("-" * 60)
    for model_name in models:
        acc_100 = [r for r in all_results if r["model_name"] == model_name and r["fraction"] == 1.0][0]["val_acc"]
        acc_5 = [r for r in all_results if r["model_name"] == model_name and r["fraction"] == 0.05][0]["val_acc"]
        rel_drop = (acc_100 - acc_5) / acc_100 if acc_100 > 0 else 0
        print(f"{model_name:<18} {acc_100:<12.2f} {acc_5:<12.2f} {rel_drop:<18.4f}")

    # Save results
    save_results = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    s3_res_path = os.path.join(output_dir, "s3_results.json")
    if os.path.exists(s3_res_path) and args.model != "all":
        with open(s3_res_path, "r") as f:
            existing = json.load(f)
            existing.extend(save_results)
            save_results = existing
            
    with open(s3_res_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
