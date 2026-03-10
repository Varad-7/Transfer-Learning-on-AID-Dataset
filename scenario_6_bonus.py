import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import argparse
import time
import json
import matplotlib.pyplot as plt

from src.models import get_model
import torchvision.models as models
from src.dataset import get_dataloaders
from src.train_utils import evaluate

def instantiate_model(model_name, num_classes=30):
    model, config = get_model(model_name, num_classes=num_classes)
    return model

def apply_gradient_guided_unfreezing(model, train_loader, device, unfreeze_pct=0.2):
    """
    Novel Automated Tuning Strategy:
    1. Pass one batch of data through the totally frozen network.
    2. Compute the gradient of the loss with respect to all frozen parameters.
    3. Calculate the L2 norm of the gradient for each parameter tensor.
    4. Unfreeze the layers that have the top K% highest gradient magnitudes.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Temporarily require grad to compute the sensitivities
    for param in model.parameters():
        param.requires_grad = True

    # 1. Forward pass on one batch
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # 2. Collect gradient magnitudes for all layers EXCEPT the new head (fc)
    grad_magnitudes = []
    for name, param in model.named_parameters():
        if "fc" not in name and param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_magnitudes.append((name, param, grad_norm, param.numel()))
            
    # Reset gradients and freeze everything again
    model.zero_grad()
    for param in model.parameters():
        param.requires_grad = False
        
    # Always keep the new head unfrozen
    for param in model.fc.parameters():
        param.requires_grad = True

    # 3. Sort layers by gradient magnitude (descending)
    grad_magnitudes.sort(key=lambda x: x[2], reverse=True)
    
    total_backbone_params = sum(x[3] for x in grad_magnitudes)
    target_unfreeze_params = int(total_backbone_params * unfreeze_pct)
    
    unfrozen_count = 0
    unfrozen_layers = []
    
    # 4. Unfreeze the top K% layers based on gradient magnitude
    for name, param, mag, numel in grad_magnitudes:
        if unfrozen_count + numel <= target_unfreeze_params:
            param.requires_grad = True
            unfrozen_count += numel
            unfrozen_layers.append(name)
        elif unfrozen_count < target_unfreeze_params:
            # Partially unfreezing a layer is not possible, so we just unfreeze the whole layer if it fits slightly over, or skip
            param.requires_grad = True
            unfrozen_count += numel
            unfrozen_layers.append(name)
            break
            
    print(f"\n[Gradient-Guided] Automated Tuning Strategy elected to unfreeze {len(unfrozen_layers)} layers:")
    for l in unfrozen_layers[:5]:  # Print top 5 for brevity
        print(f"  - {l}")
    print(f"  ... and {len(unfrozen_layers)-5} more.")
    
    pct_actual = unfrozen_count / total_backbone_params
    print(f"Total backbone parameters unfrozen: {unfrozen_count}/{total_backbone_params} ({pct_actual*100:.1f}%)\n")
    return model

def apply_random_unfreezing(model, unfreeze_pct=0.2):
    # Same as scenario 2
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    backbone_params = [(name, p, p.numel()) for name, p in model.named_parameters() if "fc" not in name]
    import random
    random.shuffle(backbone_params)
    
    total_backbone_params = sum(x[2] for x in backbone_params)
    target_unfreeze = int(total_backbone_params * unfreeze_pct)
    
    unfrozen_count = 0
    for name, p, numel in backbone_params:
        if unfrozen_count + numel <= target_unfreeze:
            p.requires_grad = True
            unfrozen_count += numel
        elif unfrozen_count < target_unfreeze:
            p.requires_grad = True
            unfrozen_count += numel
            break
            
    print(f"\n[Random Unfreezing] Total backbone parameters unfrozen: {unfrozen_count}/{total_backbone_params} ({(unfrozen_count/total_backbone_params)*100:.1f}%)\n")
    return model

def train_short(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4) # Adam for faster convergence in short runs
    
    val_accuracies = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_accuracies.append(val_acc)
        print(f"  Epoch {epoch}/{epochs} - Val Acc: {val_acc:.2f}%")
        
    return val_accuracies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="train_data")
    parser.add_argument("--epochs", type=int, default=5) # Short run to prove the point
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    out_dir = os.path.join("outputs", "scenario_6")
    os.makedirs(out_dir, exist_ok=True)
    
    train_loader, val_loader, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=0
    )
    num_classes = len(class_names)
    
    print("\n" + "="*60)
    print("  Bonus: Gradient-Guided Automated Parameter Selection")
    print("="*60)
    
    # Run Random Unfreezing
    print("\n--- Running Random 20% Unfreezing (Baseline) ---")
    model_random = instantiate_model("resnet50", num_classes=num_classes)
    model_random = apply_random_unfreezing(model_random, unfreeze_pct=0.2)
    model_random.to(device)
    val_acc_random = train_short(model_random, train_loader, val_loader, device, epochs=args.epochs)
    
    # Run Gradient-Guided Unfreezing
    print("\n--- Running Gradient-Guided 20% Unfreezing (Novel Strategy) ---")
    model_guided = instantiate_model("resnet50", num_classes=num_classes)
    model_guided.to(device)
    model_guided = apply_gradient_guided_unfreezing(model_guided, train_loader, device, unfreeze_pct=0.2)
    val_acc_guided = train_short(model_guided, train_loader, val_loader, device, epochs=args.epochs)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, args.epochs + 1)
    plt.plot(epochs_range, val_acc_random, marker='o', label="Random Selection (20%)", color="gray", linestyle="--")
    plt.plot(epochs_range, val_acc_guided, marker='s', label="Gradient-Guided Selection (20%)", color="blue", linewidth=2)
    plt.title("Automated Tuning Strategy: Random vs. Gradient-Guided Parameter Selection")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "s6_automated_tuning.png")
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")
    
    results = {
        "random_val_acc": val_acc_random,
        "guided_val_acc": val_acc_guided
    }
    with open(os.path.join(out_dir, "s6_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nScenario 6 Bonus execution complete!")

if __name__ == "__main__":
    main()
