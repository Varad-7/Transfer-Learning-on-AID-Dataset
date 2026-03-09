"""
models.py — Model definitions and utilities for ResNet50, EfficientNet-B0, ConvNeXt-Tiny.
"""
import torch
import torch.nn as nn
import timm
from collections import OrderedDict


# ---------- Model Factory ----------
MODEL_CONFIGS = {
    "resnet50": {
        "timm_name": "resnet50",
        "last_block_pattern": "layer4",  # Last residual block
        "feature_layers": {
            "early": "layer1",
            "middle": "layer2",
            "late": "layer4",
        },
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "last_block_pattern": "blocks.6",  # Last block
        "feature_layers": {
            "early": "blocks.1",
            "middle": "blocks.3",
            "late": "blocks.6",
        },
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny",
        "last_block_pattern": "stages.3",  # Last stage
        "feature_layers": {
            "early": "stages.0",
            "middle": "stages.1",
            "late": "stages.3",
        },
    },
}


def get_model(model_name, num_classes=30, pretrained=True):
    """
    Load a pretrained model from timm and replace the classifier head.
    Returns (model, config).
    """
    config = MODEL_CONFIGS[model_name]
    model = timm.create_model(
        config["timm_name"],
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model, config


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_info(model, model_name, input_size=(1, 3, 224, 224)):
    """Get model info: parameter count, MACs, FLOPs."""
    total_params, trainable_params = count_parameters(model)

    # Estimate FLOPs/MACs using timm's profiler
    try:
        from timm.utils import profile_fvcore
        macs, _ = profile_fvcore(model, input_size=input_size)
    except Exception:
        # Fallback: rough estimate based on model name
        macs_map = {
            "resnet50": 4.1e9,
            "efficientnet_b0": 0.39e9,
            "convnext_tiny": 4.5e9,
        }
        macs = macs_map.get(model_name, 0)

    flops = macs * 2  # FLOPs ≈ 2 * MACs
    
    info = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "MACs": macs,
        "FLOPs": flops,
    }
    return info


# ---------- Freezing / Unfreezing Strategies ----------
def freeze_backbone(model):
    """Freeze ALL parameters, then unfreeze the classifier head."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the classification head
    _unfreeze_head(model)


def _unfreeze_head(model):
    """Unfreeze only the final classifier layer(s)."""
    # timm models use different head attribute names
    head_attrs = ["fc", "classifier", "head", "head.fc"]
    for attr in head_attrs:
        parts = attr.split(".")
        module = model
        try:
            for p in parts:
                module = getattr(module, p)
            for param in module.parameters():
                param.requires_grad = True
            return
        except AttributeError:
            continue


def unfreeze_last_block(model, config):
    """Unfreeze last block of the backbone + classification head."""
    freeze_backbone(model)
    pattern = config["last_block_pattern"]
    for name, param in model.named_parameters():
        if pattern in name:
            param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze ALL parameters."""
    for param in model.parameters():
        param.requires_grad = True


def selective_unfreeze(model, config, fraction=0.2):
    """
    Unfreeze approximately `fraction` of total backbone parameters.
    Strategy: Unfreeze from the last layers going backwards until ~fraction is met.
    Always keeps the classification head unfrozen.
    """
    freeze_backbone(model)

    # Get all backbone (non-head) parameter groups in reverse order
    all_params = []
    head_attrs = ["fc", "classifier", "head"]
    for name, param in model.named_parameters():
        is_head = any(name.startswith(h) for h in head_attrs)
        if not is_head:
            all_params.append((name, param))

    total_backbone = sum(p.numel() for _, p in all_params)
    target = int(total_backbone * fraction)
    
    # Unfreeze from last to first until budget met
    unfrozen = 0
    for name, param in reversed(all_params):
        if unfrozen >= target:
            break
        param.requires_grad = True
        unfrozen += param.numel()

    return unfrozen, total_backbone


# ---------- Feature Extraction ----------
class FeatureExtractor:
    """
    Hook-based feature extractor for intermediate layers.
    """
    def __init__(self, model, layer_name):
        self.features = None
        self._hook = None
        self._register_hook(model, layer_name)

    def _register_hook(self, model, layer_name):
        """Register a forward hook on the specified layer."""
        parts = layer_name.split(".")
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        self._hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        if self._hook is not None:
            self._hook.remove()


def extract_features(model, dataloader, layer_name, device):
    """
    Extract features from a specific layer for the entire dataloader.
    Returns (features_array, labels_array).
    """
    import numpy as np
    model.eval()
    extractor = FeatureExtractor(model, layer_name)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)
            feats = extractor.features
            
            # Global average pool if feats are spatial
            if feats.dim() == 4:
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
            elif feats.dim() == 3:
                feats = feats.mean(dim=1)
            
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    
    extractor.remove()
    return np.concatenate(all_features), np.concatenate(all_labels)


def get_device():
    """Auto-detect and return the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
