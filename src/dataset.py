"""
dataset.py — Data loading, transforms, splits, corruptions for AID dataset.
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from PIL import Image, ImageFilter, ImageEnhance


# ---------- Constants ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224
SEED = 42


def seed_everything(seed=SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


# ---------- Transforms ----------
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------- Corruption Transforms ----------
class GaussianNoise:
    """Add pixel-level Gaussian noise."""
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.sigma
        return torch.clamp(tensor + noise, 0.0, 1.0)


class MotionBlur:
    """Simulate motion blur using a directional kernel."""
    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size

    def __call__(self, img):
        # Apply to PIL image before ToTensor
        return img.filter(ImageFilter.BoxBlur(self.kernel_size // 3))


class BrightnessShift:
    """Shift brightness of the image."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)


def get_corruption_transform(corruption_type, level=None):
    """
    Return a transform pipeline with a specific corruption applied.
    Corruptions applied on PIL images before normalization except Gaussian noise.
    """
    base = [transforms.Resize((IMG_SIZE, IMG_SIZE))]

    if corruption_type == "gaussian_noise":
        sigma = level if level else 0.1
        return transforms.Compose(base + [
            transforms.ToTensor(),
            GaussianNoise(sigma=sigma),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif corruption_type == "motion_blur":
        return transforms.Compose(base + [
            MotionBlur(kernel_size=15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif corruption_type == "brightness":
        return transforms.Compose(base + [
            BrightnessShift(factor=1.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")


# ---------- Dataset Utilities ----------
def load_dataset(data_dir, transform=None):
    """Load dataset from image-folder format."""
    return datasets.ImageFolder(root=data_dir, transform=transform)


def get_train_val_split(dataset, val_ratio=0.2, seed=SEED):
    """
    Split dataset into train/val using stratified split.
    Returns (train_indices, val_indices).
    """
    from sklearn.model_selection import train_test_split

    targets = [s[1] for s in dataset.samples]
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio,
        stratify=targets, random_state=seed
    )
    return train_idx, val_idx


def get_few_shot_indices(dataset, indices, fraction, seed=SEED):
    """
    Get a stratified subset of given indices.
    fraction: e.g., 0.2 for 20%, 0.05 for 5%
    """
    from sklearn.model_selection import train_test_split

    targets = [dataset.targets[i] for i in indices]
    keep_idx, _ = train_test_split(
        indices, train_size=fraction,
        stratify=targets, random_state=seed
    )
    return keep_idx


def get_dataloaders(data_dir, batch_size=32, num_workers=0, 
                    few_shot_fraction=None, seed=SEED):
    """
    Create train and val DataLoaders.
    If few_shot_fraction is given, use only that fraction of training data.
    """
    seed_everything(seed)

    # Load full dataset with different transforms
    full_dataset = load_dataset(data_dir, transform=None)
    train_idx, val_idx = get_train_val_split(full_dataset, seed=seed)

    if few_shot_fraction is not None and few_shot_fraction < 1.0:
        train_idx = get_few_shot_indices(full_dataset, train_idx, few_shot_fraction, seed=seed)

    # Create subsets with appropriate transforms
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = TransformedSubset(full_dataset, train_idx, train_transform)
    val_dataset = TransformedSubset(full_dataset, val_idx, val_transform)

    # Disable pin_memory on MPS (not supported)
    import torch
    use_pin = not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )

    class_names = full_dataset.classes
    return train_loader, val_loader, class_names


class TransformedSubset(Dataset):
    """Subset with a custom transform applied."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_corruption_loader(data_dir, corruption_type, level=None,
                          batch_size=32, num_workers=0, seed=SEED):
    """Get a DataLoader for the validation set with corruption applied."""
    seed_everything(seed)
    full_dataset = load_dataset(data_dir, transform=None)
    _, val_idx = get_train_val_split(full_dataset, seed=seed)
    
    corruption_tf = get_corruption_transform(corruption_type, level)
    val_dataset = TransformedSubset(full_dataset, val_idx, corruption_tf)
    
    import torch
    use_pin = not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    return DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )


def get_fixed_subset_for_pca(data_dir, samples_per_class=30, seed=SEED):
    """
    Get a fixed subset of 30 samples per class for PCA visualization.
    Returns (indices, labels).
    """
    seed_everything(seed)
    full_dataset = load_dataset(data_dir, transform=None)
    _, val_idx = get_train_val_split(full_dataset, seed=seed)
    
    # Group val indices by class
    class_to_indices = {}
    for idx in val_idx:
        label = full_dataset.targets[idx]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    # Sample up to `samples_per_class` from each class
    selected = []
    for label in sorted(class_to_indices.keys()):
        idxs = class_to_indices[label]
        random.seed(seed)
        random.shuffle(idxs)
        selected.extend(idxs[:samples_per_class])
    
    return selected, full_dataset
