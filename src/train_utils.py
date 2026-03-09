"""
train_utils.py — Training/evaluation loops, metrics, and logging.
"""
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def get_device():
    """Auto-detect best device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=0):
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def compute_gradient_norms(model):
    """
    Compute gradient L2 norms for each named parameter group.
    Returns dict: {layer_name: norm_value}.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm(2).item()
    return grad_norms


def compute_gradient_norm_stats(model):
    """
    Compute summary gradient norm statistics by layer group.
    Returns dict: {group_name: {"mean": ..., "max": ..., "min": ...}}.
    """
    # Group by top-level module
    groups = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            group = name.split(".")[0]
            if group not in groups:
                groups[group] = []
            groups[group].append(param.grad.data.norm(2).item())

    stats = {}
    for group, norms in groups.items():
        stats[group] = {
            "mean": np.mean(norms),
            "max": np.max(norms),
            "min": np.min(norms),
        }
    return stats


def train_model(model, train_loader, val_loader, device,
                num_epochs=30, lr=1e-3, weight_decay=1e-4,
                track_grad_norms=False):
    """
    Full training loop with logging.
    Returns history dict with train/val losses, accuracies, and optionally grad norms.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "grad_norms": [],
    }

    model.to(device)
    print(f"\nTraining on {device} for {num_epochs} epochs")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Optionally track gradient norms
        if track_grad_norms:
            grad_stats = compute_gradient_norm_stats(model)
            history["grad_norms"].append(grad_stats)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
              f"Time: {elapsed:.1f}s")

    return history


def get_final_predictions(model, dataloader, device):
    """Get predictions and labels for confusion matrix, etc."""
    criterion = nn.CrossEntropyLoss()
    _, accuracy, preds, labels = evaluate(model, dataloader, criterion, device)
    return accuracy, preds, labels
