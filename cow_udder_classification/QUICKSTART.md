# 🚀 Quick Start Guide

## Dataset Preparation: ✅ COMPLETE

Your cow udder classification dataset has been successfully organized and is ready for training!

## What's Been Done

✅ **Organized Data into Binary Classification Format**
- All normal udder images → `normal/` class
- All diseased (mastitis) udder images → `diseased/` class
- Data split into train (78%), validation (13%), and test (9%) sets

✅ **Created 269 Total Images**
- 100 normal (healthy) udders
- 169 diseased (mastitis) udders
- Properly balanced across all splits

✅ **Generated Documentation & Scripts**
- `README.md` - Complete dataset documentation
- `data.yaml` - Configuration file for training frameworks
- `train.py` - Ready-to-run PyTorch training script
- `inference.py` - Prediction script for new images
- `requirements.txt` - All dependencies

## Quick Start: Train Your Model

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Apply Data Augmentation (Optional but Recommended)
```bash
python augment_dataset.py
```

This will:
- Create 3x augmented versions of each training image
- Expand training set from 210 → 840 images
- Apply rotation, flipping, brightness/contrast adjustments
- **Significantly improve model generalization**

For different augmentation levels:
```bash
# Light augmentation
python augment_dataset.py --num-augmentations 1 --augmentation-type light

# Heavy augmentation  
python augment_dataset.py --num-augmentations 5 --augmentation-type heavy
```

### 3. Train the Model
```bash
python train.py
```
This will:
- Load the dataset from train/valid folders
- Train a ResNet50 model with transfer learning
- Automatically save the best model as `best_model.pth`
- Evaluate on the test set

### 3. Make Predictions

**Single Image:**
```bash
python inference.py best_model.pth --image path/to/udder_image.jpg
```

**Batch Processing (Directory):**
```bash
python inference.py best_model.pth --directory ./test/normal
```

## Dataset Breakdown

### Training Set (210 images)
- Normal: 80
- Diseased: 130
- Ratio: 1.62:1 (diseased to normal)

### Validation Set (35 images)
- Normal: 10
- Diseased: 25
- Ratio: 2.5:1 (diseased to normal)

### Test Set (24 images)
- Normal: 10
- Diseased: 14
- Ratio: 1.4:1 (diseased to normal)

## Model Output

When the model makes a prediction, it will output:
- **Class**: `normal` (healthy udder) or `diseased` (mastitis)
- **Confidence**: Probability score (0-100%)
- **Probabilities**: Individual class probabilities

Example Output:
```
Image: udder_sample.jpg
Prediction: DISEASED (Confidence: 87.5%)
  - Normal probability: 12.5%
  - Diseased probability: 87.5%
```

## Supported Models

The `train.py` script uses **ResNet50** by default with transfer learning. You can modify it to use:
- `resnet50` (default)
- `efficientnet_b0`
- Other torchvision models

## Key Features

✨ **Transfer Learning**: Pre-trained ImageNet weights for faster convergence
✨ **Data Augmentation**: Automatic augmentation during training (rotation, flip, color jitter)
✨ **Learning Rate Scheduling**: Adaptive learning rate reduction
✨ **Best Model Saving**: Automatically saves the best performing model
✨ **Comprehensive Evaluation**: Reports on train/valid/test sets

## File Descriptions

| File | Purpose |
|------|---------|
| `README.md` | Full dataset documentation and statistics |
| `data.yaml` | YOLO/Framework configuration |
| `train.py` | Complete training pipeline with PyTorch |
| `inference.py` | Make predictions on new images |
| `requirements.txt` | Python package dependencies |
| `train/` | Training images (80 normal, 130 diseased) |
| `valid/` | Validation images (10 normal, 25 diseased) |
| `test/` | Test images (10 normal, 14 diseased) |

## Tips for Best Results

1. **GPU Acceleration**: If you have a GPU, ensure CUDA is installed for faster training
2. **Batch Size**: Adjust `batch_size` in `train.py` if you run into memory issues
3. **Epochs**: The default is 50 epochs. Increase if validation accuracy is still improving
4. **Data Augmentation**: The script applies augmentation automatically. Modify if needed

## Troubleshooting

**Out of Memory Error?**
→ Reduce `batch_size` from 32 to 16 or 8 in `train.py`

**Model Not Converging?**
→ Increase number of epochs or reduce learning rate in `CONFIG`

**No CUDA/GPU Found?**
→ The script automatically falls back to CPU. Training will be slower but still works

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Review the dataset: Open `README.md` for detailed statistics
3. Train a model: `python train.py`
4. Make predictions: `python inference.py best_model.pth --image <your_image.jpg>`

---

**Dataset is ready! Happy training! 🎉**
