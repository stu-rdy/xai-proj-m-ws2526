#!/usr/bin/env python3
"""
Training CLI for ImageNetSubset Classification.

Usage:
    python scripts/train.py --model resnet18 --epochs 5
    python scripts/train.py --model resnet34 --epochs 5 --wandb
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_device
from src.utils.device import get_device_name
from src.models import SUPPORTED_MODELS, create_model
from src.data import create_data_loaders, get_dataset_info
from src.training import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN on ImageNetSubset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=SUPPORTED_MODELS,
        help="Model architecture",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train from scratch without pretrained weights",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="ImageNetSubset",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Output arguments
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="imagenet-subset", help="W&B project name"
    )
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"🚀 {args.model.upper()} Training on ImageNetSubset")
    print("=" * 60)

    # Setup device
    device = get_device()
    print(f"\n🖥️  Device: {get_device_name(device)}")

    # Load dataset info
    dataset_info = get_dataset_info(args.data_dir)
    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {dataset_info['train_samples']}")
    print(f"   Validation samples: {dataset_info['val_samples']}")
    print(f"   Classes: {dataset_info['classes']}")
    print(f"   Number of classes: {dataset_info['num_classes']}")

    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    # Create model
    pretrained = not args.no_pretrained
    print(
        f"\n{'🔄 Loading pretrained' if pretrained else '🆕 Creating'} {args.model} weights..."
    )
    model = create_model(args.model, num_classes, pretrained, device)
    print(f"✓ {args.model} ready with {num_classes} output classes")

    # Training config
    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_classes": num_classes,
    }

    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Momentum: {args.momentum}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Pretrained: {pretrained}")
    print(f"   Save directory: {args.save_dir}")
    print(f"   Wandb logging: {args.wandb}")

    # Wandb config
    wandb_config = {
        "project": args.wandb_project,
        "run_name": args.wandb_run_name,
    }

    # Train
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_dir=args.save_dir,
        model_name=args.model,
        wandb_enabled=args.wandb,
        wandb_config=wandb_config,
    )

    return results


if __name__ == "__main__":
    main()
