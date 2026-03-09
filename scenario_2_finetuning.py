"""
Scenario 2: Fine-Tuning Strategies
- Compare 4 strategies: linear probe, last block FT, full FT, selective 20% unfreeze
- Report: acc vs % unfrozen params, gradient norms, convergence comparison
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders, seed_everything, SEED
from src.models import (get_model, freeze_backbone, unfreeze_last_block,
                         unfreeze_all, selective_unfreeze, count_parameters,
                         get_model_info, get_device)
from src.train_utils import train_model, get_final_predictions
from src.visualize import (plot_accuracy_vs_unfrozen, plot_gradient_norms,
                            plot_convergence_comparison, plot_training_curves)

import torch


STRATEGIES = {
    "linear_probe": {
        "description": "Frozen backbone, train classifier only",
        "setup": lambda model, config: freeze_backbone(model),
    },
    "last_block": {
        "description": "Unfreeze last block + classifier",
        "setup": lambda model, config: unfreeze_last_block(model, config),
    },
    "full_ft": {
        "description": "Full fine-tuning of entire model",
        "setup": lambda model, config: unfreeze_all(model),
    },
    "selective_20pct": {
        "description": "Unfreeze ~20% of backbone params from last layers",
        "setup": lambda model, config: selective_unfreeze(model, config, fraction=0.2),
    },
}


def run_strategy(model_name, strategy_name, data_dir, device, num_epochs=30, batch_size=32):
    """Run a single fine-tuning strategy for a model."""
    print(f"\n  Strategy: {strategy_name}")
    seed_everything(SEED)

    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0
    )

    model, config = get_model(model_name, num_classes=len(class_names))
    
    # Apply strategy
    result = STRATEGIES[strategy_name]["setup"](model, config)
    
    total, trainable = count_parameters(model)
    pct_unfrozen = 100.0 * trainable / total
    print(f"    Total: {total:,}, Trainable: {trainable:,} ({pct_unfrozen:.1f}%)")

    # Train with gradient norm tracking
    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, lr=1e-3,
        track_grad_norms=True
    )

    final_acc, _, _ = get_final_predictions(model, val_loader, device)

    return {
        "model_name": model_name,
        "strategy": strategy_name,
        "pct_unfrozen": pct_unfrozen,
        "val_acc": final_acc,
        "total_params": total,
        "trainable_params": trainable,
        "history": history,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scenario 2: Fine-Tuning Strategies")
    parser.add_argument("--model", type=str, default="all", help="Model to run: resnet50, efficientnet_b0, convnext_tiny, or 'all'")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_2")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    if args.model == "all":
        models = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    else:
        models = [args.model]

    strategy_names = list(STRATEGIES.keys())
    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Fine-Tuning Strategies: {model_name}")
        print(f"{'='*60}")

        # Print model info
        temp_model, _ = get_model(model_name, num_classes=30)
        info = get_model_info(temp_model, model_name)
        print(f"  MACs: {info['MACs']:.2e}, FLOPs: {info['FLOPs']:.2e}")
        del temp_model

        model_histories = []
        for strategy_name in strategy_names:
            result = run_strategy(model_name, strategy_name, data_dir, device,
                                   num_epochs=args.epochs, batch_size=args.batch_size)
            all_results.append(result)
            model_histories.append(result)

            # Free memory
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Plot convergence comparison for this model
        histories = [r["history"] for r in model_histories]
        plot_convergence_comparison(
            histories, strategy_names, model_name,
            os.path.join(output_dir, f"s2_{model_name}_convergence.png")
        )

        # Plot training curves for each strategy
        for r in model_histories:
            plot_training_curves(
                r["history"],
                f"{model_name} - {r['strategy']}",
                os.path.join(output_dir, f"s2_{model_name}_{r['strategy']}_curves.png")
            )

        # Plot gradient norms for strategies that have them
        grad_data = []
        grad_names = []
        for r in model_histories:
            if r["history"]["grad_norms"]:
                grad_data.append(r["history"]["grad_norms"][-1])  # last epoch
                grad_names.append(r["strategy"])
        if grad_data:
            plot_gradient_norms(grad_data, grad_names,
                                os.path.join(output_dir, f"s2_{model_name}_grad_norms.png"))

    # Plot accuracy vs unfrozen parameters
    plot_data = [{"model_name": r["model_name"], "strategy": r["strategy"],
                  "pct_unfrozen": r["pct_unfrozen"], "val_acc": r["val_acc"]}
                 for r in all_results]
    plot_accuracy_vs_unfrozen(plot_data, os.path.join(output_dir, "s2_acc_vs_unfrozen.png"))

    # Summary table
    print(f"\n{'='*100}")
    print(f"  SCENARIO 2: FINE-TUNING STRATEGIES SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<18} {'Strategy':<18} {'% Unfrozen':<12} {'Val Acc (%)':<14} {'Trainable':<14}")
    print("-" * 76)
    for r in all_results:
        print(f"{r['model_name']:<18} {r['strategy']:<18} {r['pct_unfrozen']:<12.1f} "
              f"{r['val_acc']:<14.2f} {r['trainable_params']:<14,}")

    # Save results
    save_results = [{k: v for k, v in r.items() if k != "history"} for r in all_results]
    # Append if exists and we aren't running all models
    s2_res_path = os.path.join(output_dir, "s2_results.json")
    if os.path.exists(s2_res_path) and args.model != "all":
        with open(s2_res_path, "r") as f:
            existing = json.load(f)
            existing.extend(save_results)
            save_results = existing
            
    with open(s2_res_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
