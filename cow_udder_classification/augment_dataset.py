#!/usr/bin/env python3
"""
Data Augmentation Script for Cow Udder Classification Dataset
Generates augmented versions of training images to expand the dataset
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path
import random
import argparse


class ImageAugmenter:
    """Apply various augmentation techniques to images"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
    def rotate(self, image, angle_range=(-20, 20)):
        """Randomly rotate image"""
        angle = random.uniform(*angle_range)
        return image.rotate(angle, expand=False, fillcolor='white')
    
    def flip_horizontal(self, image, probability=0.5):
        """Randomly flip image horizontally"""
        if random.random() < probability:
            return ImageOps.mirror(image)
        return image
    
    def flip_vertical(self, image, probability=0.3):
        """Randomly flip image vertically"""
        if random.random() < probability:
            return ImageOps.flip(image)
        return image
    
    def adjust_brightness(self, image, factor_range=(0.7, 1.3)):
        """Randomly adjust brightness"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image, factor_range=(0.8, 1.2)):
        """Randomly adjust contrast"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, image, factor_range=(0.8, 1.2)):
        """Randomly adjust color saturation"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def adjust_sharpness(self, image, factor_range=(0.5, 1.5)):
        """Randomly adjust sharpness"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def add_gaussian_noise(self, image, std_range=(0.01, 0.05)):
        """Add Gaussian noise to image"""
        image_array = np.array(image, dtype=np.float32) / 255.0
        std = random.uniform(*std_range)
        noise = np.random.normal(0, std, image_array.shape)
        noisy_array = np.clip(image_array + noise, 0, 1)
        return Image.fromarray((noisy_array * 255).astype(np.uint8))
    
    def crop_and_resize(self, image, crop_factor_range=(0.8, 0.95)):
        """Randomly crop and resize back to original size"""
        w, h = image.size
        crop_factor = random.uniform(*crop_factor_range)
        crop_w = int(w * crop_factor)
        crop_h = int(h * crop_factor)
        
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), Image.Resampling.LANCZOS)
    
    def augment(self, image, augmentation_type='standard'):
        """Apply augmentation pipeline"""
        if augmentation_type == 'light':
            # Light augmentation
            image = self.flip_horizontal(image, probability=0.5)
            image = self.adjust_brightness(image, factor_range=(0.9, 1.1))
        
        elif augmentation_type == 'standard':
            # Standard augmentation
            image = self.rotate(image, angle_range=(-15, 15))
            image = self.flip_horizontal(image, probability=0.5)
            image = self.adjust_brightness(image, factor_range=(0.8, 1.2))
            image = self.adjust_contrast(image, factor_range=(0.9, 1.1))
        
        elif augmentation_type == 'heavy':
            # Heavy augmentation
            image = self.rotate(image, angle_range=(-25, 25))
            image = self.flip_horizontal(image, probability=0.5)
            image = self.flip_vertical(image, probability=0.3)
            image = self.adjust_brightness(image, factor_range=(0.7, 1.3))
            image = self.adjust_contrast(image, factor_range=(0.8, 1.2))
            image = self.adjust_saturation(image, factor_range=(0.8, 1.2))
            image = self.crop_and_resize(image, crop_factor_range=(0.85, 0.98))
        
        elif augmentation_type == 'extreme':
            # Extreme augmentation
            image = self.rotate(image, angle_range=(-30, 30))
            image = self.flip_horizontal(image, probability=0.5)
            image = self.flip_vertical(image, probability=0.4)
            image = self.adjust_brightness(image, factor_range=(0.6, 1.4))
            image = self.adjust_contrast(image, factor_range=(0.7, 1.3))
            image = self.adjust_saturation(image, factor_range=(0.7, 1.3))
            image = self.adjust_sharpness(image, factor_range=(0.5, 1.5))
            image = self.add_gaussian_noise(image, std_range=(0.01, 0.05))
            image = self.crop_and_resize(image, crop_factor_range=(0.8, 0.95))
        
        return image


def augment_dataset(dataset_path, num_augmentations_per_image=3, augmentation_type='standard'):
    """
    Augment all images in the training set
    
    Args:
        dataset_path: Path to cow_udder_classification folder
        num_augmentations_per_image: How many augmented versions to create per image
        augmentation_type: Type of augmentation ('light', 'standard', 'heavy', 'extreme')
    """
    dataset_path = Path(dataset_path)
    train_path = dataset_path / 'train'
    
    if not train_path.exists():
        print(f"Error: Training directory not found at {train_path}")
        return
    
    augmenter = ImageAugmenter()
    total_augmented = 0
    
    # Process each class
    for class_dir in train_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"\nAugmenting {class_name} images...")
        
        # Get all images
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        original_count = len(image_files)
        
        if original_count == 0:
            print(f"  No images found in {class_name}")
            continue
        
        print(f"  Found {original_count} original images")
        
        # Create augmented versions
        for aug_num, image_file in enumerate(image_files, 1):
            if aug_num % max(1, original_count // 5) == 0 or aug_num == original_count:
                print(f"    Processing {aug_num}/{original_count}...", end='\r')
            try:
                # Open image
                image = Image.open(image_file).convert('RGB')
                
                # Generate augmented versions
                for aug_idx in range(num_augmentations_per_image):
                    augmented_image = augmenter.augment(image, augmentation_type=augmentation_type)
                    
                    # Create filename with augmentation suffix
                    stem = image_file.stem
                    suffix = image_file.suffix
                    aug_filename = f"{stem}_aug_{aug_idx+1}{suffix}"
                    aug_path = image_file.parent / aug_filename
                    
                    # Save augmented image
                    augmented_image.save(aug_path, quality=95)
                    total_augmented += 1
            
            except Exception as e:
                print(f"  Error processing {image_file.name}: {e}")
        
        new_count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png')))
        print(f"  ✓ {class_name}: {original_count} → {new_count} images (+{new_count - original_count})")
    
    return total_augmented


def get_dataset_stats(dataset_path):
    """Get statistics about dataset before and after augmentation"""
    dataset_path = Path(dataset_path)
    
    stats = {}
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            stats[split] = {}
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    image_count = len(list(class_dir.glob('*.jpg')) + 
                                     list(class_dir.glob('*.jpeg')) + 
                                     list(class_dir.glob('*.png')))
                    stats[split][class_name] = image_count
    
    return stats


def print_dataset_stats(stats):
    """Print formatted dataset statistics"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    for split in ['train', 'valid', 'test']:
        if split in stats:
            print(f"\n{split.upper()} Set:")
            total = 0
            for class_name, count in stats[split].items():
                print(f"  {class_name.capitalize():<15} : {count:>4} images")
                total += count
            print(f"  {'-'*25}")
            print(f"  {'Total':<15} : {total:>4} images")
    
    # Calculate overall stats
    total_all = sum(sum(counts.values()) for counts in stats.values())
    print(f"\n{'='*70}")
    print(f"TOTAL IMAGES: {total_all}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Augment training images in the cow udder dataset'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='.',
        help='Path to cow_udder_classification directory'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=3,
        help='Number of augmented versions per image (default: 3)'
    )
    parser.add_argument(
        '--augmentation-type',
        type=str,
        default='standard',
        choices=['light', 'standard', 'heavy', 'extreme'],
        help='Type of augmentation to apply (default: standard)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show statistics without applying augmentation'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("COW UDDER DATASET - DATA AUGMENTATION")
    print("="*70)
    
    # Show initial statistics
    print("\nInitial Dataset Statistics:")
    initial_stats = get_dataset_stats(args.dataset_path)
    print_dataset_stats(initial_stats)
    
    if args.dry_run:
        print("\n(Dry run mode - no augmentation applied)")
        return
    
    # Apply augmentation
    print(f"\nApplying {args.augmentation_type} augmentation...")
    print(f"Creating {args.num_augmentations} augmented versions per image")
    print()
    
    total_augmented = augment_dataset(
        args.dataset_path,
        num_augmentations_per_image=args.num_augmentations,
        augmentation_type=args.augmentation_type
    )
    
    # Show final statistics
    print("\n" + "-"*70)
    print("Final Dataset Statistics:")
    final_stats = get_dataset_stats(args.dataset_path)
    print_dataset_stats(final_stats)
    
    print(f"\n✓ Augmentation complete!")
    print(f"  Total augmented images created: {total_augmented}")
    print(f"  Dataset size increased by {((sum(sum(c.values()) for c in final_stats.values()) / sum(sum(c.values()) for c in initial_stats.values())) - 1) * 100:.1f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
