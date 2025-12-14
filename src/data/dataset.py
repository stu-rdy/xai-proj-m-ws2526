"""Dataset loading and transforms for ImageNetSubset."""

import os
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Class names for ImageNetSubset (alphabetical order as loaded by ImageFolder)
CLASS_NAMES = [
    "binder",
    "coffee_mug",
    "computer_keyboard",
    "mouse",
    "notebook",
    "remote_control",
    "soup_bowl",
    "teapot",
    "toilet_tissue",
    "wooden_spoon",
]

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    """Get training data transform with augmentation."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform() -> transforms.Compose:
    """Get validation/inference data transform (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def create_data_loaders(
    data_dir: str, batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, int]:
    """Create training and validation data loaders.

    Args:
        data_dir: Path to dataset directory (should contain train/ and val/)
        batch_size: Batch size for data loading
        num_workers: Number of data loader workers

    Returns:
        Tuple of (train_loader, val_loader, num_classes)
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=get_train_transform())
    val_dataset = datasets.ImageFolder(val_dir, transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_dataset.classes)


def get_dataset_info(data_dir: str) -> dict:
    """Get information about the dataset.

    Args:
        data_dir: Path to dataset directory

    Returns:
        Dictionary with dataset statistics
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=None)
    val_dataset = datasets.ImageFolder(val_dir, transform=None)

    return {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "classes": train_dataset.classes,
        "num_classes": len(train_dataset.classes),
    }
