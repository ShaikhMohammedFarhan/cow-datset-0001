# Cow Udder Classification Dataset

## Overview
Binary classification dataset for cow udder health status prediction.
- **Classes**: Normal vs. Diseased (Mastitis)
- **Original Images**: 269 (100 normal, 169 diseased)
- **With Augmentation**: 899 (320 normal, 520 diseased in training)
- **Format**: Image classification (directory-based structure)

## Dataset Statistics

### Training Set
- Normal (healthy udders): 80 images
- Diseased (mastitis): 130 images
- **Total**: 210 images (78% of dataset)
- **Ratio**: 1.62x (diseased to normal)

### Validation Set
- Normal (healthy udders): 10 images
- Diseased (mastitis): 25 images
- **Total**: 35 images (13% of dataset)
- **Ratio**: 2.5x (diseased to normal)

### Test Set
- Normal (healthy udders): 10 images
- Diseased (mastitis): 14 images
- **Total**: 24 images (9% of dataset)
- **Ratio**: 1.4x (diseased to normal)

## Data Sources

### Normal (Healthy) Images
- **normal1/**: 50 images from first normal dataset
- **normal2/**: 50 images from second normal dataset (40 in train, 10 in valid)
- **Total normal**: 100 images

### Diseased (Mastitis) Images
- **mastatis/**: 169 images of diseased udders
- Distributed: 130 train, 25 valid, 14 test

## Directory Structure
```
cow_udder_classification/
├── train/
│   ├── normal/          (80 original + augmented)
│   └── diseased/        (130 original + augmented)
├── valid/
│   ├── normal/          (10 images)
│   └── diseased/        (25 images)
├── test/
│   ├── normal/          (10 images)
│   └── diseased/        (14 images)
├── data.yaml           (Dataset configuration)
├── README.md           (This file)
├── QUICKSTART.md       (Quick start guide)
├── AUGMENTATION_REPORT.md  (Augmentation details)
├── augment_dataset.py  (Augmentation script)
├── train.py            (Training script)
├── inference.py        (Inference script)
└── requirements.txt    (Dependencies)
```

## Image Naming Convention

### Normal Images
- **normal1 images**: Standard filenames (Screenshot format)
- **normal2 train images**: Prefixed with `n2_train_X_` to avoid naming conflicts
- **normal2 valid images**: Direct copy from original normal2/valid

### Diseased Images
- **mastatis images**: Standard filenames (mastitis_XXX.jpg format)

## Usage

### For Training in PyTorch/TensorFlow
```python
from torchvision import datasets, transforms

# Load training data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    'cow_udder_classification/train',
    transform=train_transforms
)

# Class indices
# 0: diseased (mastitis)
# 1: normal (healthy)
```

## Data Augmentation

### Apply Augmentation to Training Set
The dataset includes a data augmentation script to expand the training set:

```bash
# Standard augmentation (3x per image) - RECOMMENDED
python augment_dataset.py

# Light augmentation
python augment_dataset.py --num-augmentations 1 --augmentation-type light

# Heavy augmentation
python augment_dataset.py --num-augmentations 5 --augmentation-type heavy
```

### Augmentation Impact
- **Original training set**: 210 images
- **After 3x standard augmentation**: 840 images
- **Growth**: 234% increase in training data

### Techniques Applied
- Rotation (±15°)
- Horizontal flipping
- Brightness adjustment (0.8x to 1.2x)
- Contrast adjustment (0.9x to 1.1x)

For detailed augmentation information, see [AUGMENTATION_REPORT.md](AUGMENTATION_REPORT.md).

## Class Mapping
```
0: diseased (Mastitis)
1: normal (Healthy)
```

## Splitting Strategy
- **Train**: 77% of diseased, 80% of normal
- **Valid**: 15% of diseased, 10% of normal  
- **Test**: 8% of diseased, 10% of normal

This split maintains consistent proportions across classes while preserving data stratification.

## Notes
- Images are already sorted by class in folder structure
- No additional preprocessing is applied - ready to use
- Data is ready for binary classification models (CNN, ResNet, ViT, etc.)
- Imbalanced dataset: ~1.6x more diseased images in training set

## License
Dataset compiled from:
- Normal1 & Normal2 datasets
- Mastatis (Mastitis) dataset

For original source attribution, refer to the Roboflow README files preserved in parent directories.
