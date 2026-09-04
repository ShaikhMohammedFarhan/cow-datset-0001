#!/usr/bin/env python3
"""
Cow Udder Classification Training Script
Demonstrates how to load and train a model on the cow udder dataset
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Configuration
CONFIG = {
    'dataset_path': './cow_udder_classification',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'model_name': 'resnet50',
}

# Class names
CLASS_NAMES = ['diseased', 'normal']
NUM_CLASSES = 2


def get_data_transforms():
    """Define data augmentation and preprocessing transforms"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
    }
    return data_transforms


def load_datasets(config, transforms_dict):
    """Load train, validation, and test datasets"""
    dataset_path = Path(config['dataset_path'])
    
    datasets_dict = {}
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            datasets_dict[split] = datasets.ImageFolder(
                str(split_path),
                transform=transforms_dict[split]
            )
            print(f"✓ Loaded {split} set: {len(datasets_dict[split])} images")
        else:
            print(f"⚠ {split} directory not found at {split_path}")
    
    return datasets_dict


def create_dataloaders(datasets_dict, config):
    """Create DataLoader objects for each split"""
    dataloaders = {}
    
    for split, dataset in datasets_dict.items():
        shuffle = (split == 'train')
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle,
            num_workers=config['num_workers'],
            pin_memory=True
        )
    
    return dataloaders


def create_model(config):
    """Create pretrained model for transfer learning"""
    if config['model_name'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, NUM_CLASSES)
    elif config['model_name'] == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {config['model_name']}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate model performance"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Cow Udder Classification - Training Script")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Model: {CONFIG['model_name']}")
    print()
    
    # Load data
    print("Loading dataset...")
    transforms_dict = get_data_transforms()
    datasets_dict = load_datasets(CONFIG, transforms_dict)
    dataloaders = create_dataloaders(datasets_dict, CONFIG)
    print()
    
    # Create model
    print(f"Creating {CONFIG['model_name']} model...")
    model = create_model(CONFIG)
    model = model.to(CONFIG['device'])
    print("✓ Model created")
    print()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    best_val_acc = 0.0
    best_model_path = 'best_model.pth'
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, CONFIG['device']
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, dataloaders['val'], criterion, CONFIG['device']
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
    
    print("-" * 60)
    print(f"✓ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print()
    
    # Test evaluation
    if 'test' in dataloaders:
        print("Evaluating on test set...")
        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc = validate(
            model, dataloaders['test'], criterion, CONFIG['device']
        )
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
