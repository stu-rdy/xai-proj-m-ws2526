#!/usr/bin/env python3
"""
ResNet18 Training Script for ImageNetSubset
Optimized for Apple M1 Mac using MPS acceleration.

Usage:
    python train_resnet.py [options]

Examples:
    python train_resnet.py                           # Train with defaults
    python train_resnet.py --epochs 50 --lr 0.01    # Custom epochs and learning rate
    python train_resnet.py --no-pretrained          # Train from scratch
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_device():
    """Get the best available device for M1 Mac."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Metal Performance Shaders) acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA acceleration")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (MPS not available)")
    return device


def get_data_transforms():
    """Get training and validation data transforms."""
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir: str, batch_size: int, num_workers: int):
    """Create training and validation data loaders."""
    train_transform, val_transform = get_data_transforms()
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # Print dataset info
    print(f"\n📊 Dataset Summary:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Classes: {train_dataset.classes}")
    print(f"   Number of classes: {len(train_dataset.classes)}")
    
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


def create_model(num_classes: int, pretrained: bool, device: torch.device):
    """Create ResNet18 model with modified classifier."""
    if pretrained:
        print("\n🔄 Loading pretrained ResNet18 weights...")
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        print("\n🆕 Creating ResNet18 from scratch...")
        model = models.resnet18(weights=None)
    
    # Modify the final fully connected layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    print(f"✓ Model ready with {num_classes} output classes")
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"   Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {running_loss / (batch_idx + 1):.4f} | "
                  f"Acc: {100. * correct / total:.2f}%")
    
    epoch_time = time.time() - start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, epoch_time


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 on ImageNetSubset')
    parser.add_argument('--data-dir', type=str, default='ImageNetSubset',
                        help='Path to dataset directory (default: ImageNetSubset)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs to train (default: 25)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train from scratch without pretrained weights')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 ResNet18 Training on ImageNetSubset")
    print("=" * 60)
    
    # Setup
    device = get_device()
    
    # Create data loaders
    train_loader, val_loader, num_classes = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    
    # Create model
    model = create_model(num_classes, not args.no_pretrained, device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Training configuration summary
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Momentum: {args.momentum}")
    print(f"   Weight decay: {args.weight_decay}")
    print(f"   Pretrained: {not args.no_pretrained}")
    print(f"   Save directory: {save_dir}")
    
    # Training loop
    best_val_acc = 0.0
    print("\n" + "=" * 60)
    print("📈 Starting Training")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        print(f"\n🔄 Epoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\n   📊 Epoch Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"   ✨ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Save final model
    final_path = save_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, final_path)
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Best model saved to: {save_dir / 'best_model.pth'}")
    print(f"   Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
