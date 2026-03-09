"""
Scenario 4: Corruption Robustness Evaluation
- Evaluate under Gaussian noise (σ=0.05, 0.1, 0.2), motion blur, brightness shift
- Corruptions applied only at evaluation time
- Report: corruption error, relative robustness
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (get_dataloaders, get_corruption_loader,
                          seed_everything, SEED)
from src.models import get_model, get_model_info, count_parameters, get_device
from src.train_utils import train_model, evaluate, get_final_predictions
from src.visualize import plot_corruption_results

import torch
import torch.nn as nn


CORRUPTIONS = [
    ("gaussian_noise_0.05", "gaussian_noise", 0.05),
    ("gaussian_noise_0.1",  "gaussian_noise", 0.1),
    ("gaussian_noise_0.2",  "gaussian_noise", 0.2),
    ("motion_blur",         "motion_blur",    None),
    ("brightness",          "brightness",     None),
]


def run_corruption_eval(model_name, data_dir, output_dir, device, num_epochs=30, batch_size=32):
    """Train model on clean data, then evaluate under corruptions."""
    print(f"\n{'='*60}")
    print(f"  Corruption Robustness: {model_name}")
    print(f"{'='*60}")

    seed_everything(SEED)

    # Train on clean data (full fine-tuning for best baseline)
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0
    )

    model, config = get_model(model_name, num_classes=len(class_names))
    
    info = get_model_info(model, model_name)
    total, trainable = count_parameters(model)
    print(f"  Params: {total:,}, MACs: {info['MACs']:.2e}, FLOPs: {info['FLOPs']:.2e}")

    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, lr=1e-3
    )

    # Clean accuracy
    clean_acc, _, _ = get_final_predictions(model, val_loader, device)
    print(f"\n  Clean Val Accuracy: {clean_acc:.2f}%")

    # Evaluate under each corruption
    criterion = nn.CrossEntropyLoss()
    results = []
    for corr_name, corr_type, corr_level in CORRUPTIONS:
        corr_loader = get_corruption_loader(
            data_dir, corr_type, level=corr_level,
            batch_size=batch_size, num_workers=0
        )
        _, corr_acc, _, _ = evaluate(model, corr_loader, criterion, device)
        
        corruption_error = 1.0 - corr_acc / 100.0
        relative_robustness = corr_acc / clean_acc if clean_acc > 0 else 0

        results.append({
            "model_name": model_name,
            "corruption": corr_name,
            "clean_acc": clean_acc,
            "val_acc": corr_acc,
            "corruption_error": corruption_error,
            "relative_robustness": relative_robustness,
        })
        print(f"  {corr_name:<25} Acc: {corr_acc:.2f}% | CE: {corruption_error:.4f} | "
              f"RR: {relative_robustness:.4f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scenario 4: Corruption Robustness Evaluation")
    parser.add_argument("--model", type=str, default="all", help="Model to run: resnet50, efficientnet_b0, convnext_tiny, or 'all'")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train baseline clean model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_4")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    if args.model == "all":
        models = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    else:
        models = [args.model]

    all_results = []

    for model_name in models:
        # Evaluate on corruptions
        results = run_corruption_eval(model_name, data_dir, output_dir, device,
                                       num_epochs=args.epochs, batch_size=args.batch_size)
        all_results.extend(results)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Plot results
    plot_corruption_results(all_results, os.path.join(output_dir, "s4_corruption_results.png"))

    # Summary table
    print(f"\n{'='*110}")
    print(f"  SCENARIO 4: CORRUPTION ROBUSTNESS SUMMARY")
    print(f"{'='*110}")
    print(f"{'Model':<18} {'Corruption':<25} {'Clean Acc':<12} {'Corr Acc':<12} {'Corr Error':<12} {'Rel Robust':<12}")
    print("-" * 91)
    for r in all_results:
        print(f"{r['model_name']:<18} {r['corruption']:<25} {r['clean_acc']:<12.2f} "
              f"{r['val_acc']:<12.2f} {r['corruption_error']:<12.4f} {r['relative_robustness']:<12.4f}")

    # Save results
    s4_res_path = os.path.join(output_dir, "s4_results.json")
    save_results = all_results
    if os.path.exists(s4_res_path) and args.model != "all":
        with open(s4_res_path, "r") as f:
            existing = json.load(f)
            existing.extend(save_results)
            save_results = existing
            
    with open(s4_res_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
