#!/usr/bin/env python3
"""
Cow Udder Classification - Inference Script
Use a trained model to predict if a cow udder is diseased or normal
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


# Class configuration
CLASS_NAMES = ['diseased', 'normal']
NUM_CLASSES = 2

# Image preprocessing
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(model_path, device='cpu'):
    """Load a trained model"""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, NUM_CLASSES)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def predict_image(image_path, model, device='cpu'):
    """Predict class for a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = IMG_TRANSFORMS(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return CLASS_NAMES[predicted_class], confidence, probabilities[0].cpu().numpy()


def predict_batch(image_dir, model, device='cpu'):
    """Predict class for all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    results = []
    
    for image_file in Path(image_dir).glob('*'):
        if image_file.suffix in image_extensions:
            try:
                class_name, confidence, probs = predict_image(str(image_file), model, device)
                results.append({
                    'filename': image_file.name,
                    'class': class_name,
                    'confidence': confidence,
                    'normal_prob': probs[1],
                    'diseased_prob': probs[0]
                })
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
    
    return results


def print_results(results):
    """Print prediction results in a formatted table"""
    print("\nPrediction Results:")
    print("-" * 100)
    print(f"{'Filename':<50} {'Prediction':<15} {'Confidence':<15} {'Normal Prob':<15}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['filename']:<50} {result['class']:<15} {result['confidence']:<15.2%} {result['normal_prob']:<15.2%}")
    
    print("-" * 100)
    
    # Summary statistics
    diseased_count = sum(1 for r in results if r['class'] == 'diseased')
    normal_count = sum(1 for r in results if r['class'] == 'normal')
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print(f"\nSummary:")
    print(f"  Total images: {len(results)}")
    print(f"  Diseased predictions: {diseased_count} ({100*diseased_count/len(results):.1f}%)")
    print(f"  Normal predictions: {normal_count} ({100*normal_count/len(results):.1f}%)")
    print(f"  Average confidence: {avg_confidence:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict cow udder health status (diseased vs normal)'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to trained model (best_model.pth)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for prediction'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to directory containing multiple images'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("Cow Udder Classification - Inference")
    print("=" * 100)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print()
    
    # Load model
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    model = load_model(args.model_path, device=args.device)
    print("✓ Model loaded successfully")
    print()
    
    # Single image prediction
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image file not found at {args.image}")
            return
        
        class_name, confidence, probs = predict_image(args.image, model, args.device)
        print(f"Image: {Path(args.image).name}")
        print(f"Prediction: {class_name.upper()} (Confidence: {confidence:.2%})")
        print(f"  - Normal probability: {probs[1]:.2%}")
        print(f"  - Diseased probability: {probs[0]:.2%}")
    
    # Batch prediction
    elif args.directory:
        if not Path(args.directory).exists():
            print(f"Error: Directory not found at {args.directory}")
            return
        
        print(f"Processing images in {args.directory}...")
        results = predict_batch(args.directory, model, args.device)
        
        if results:
            print_results(results)
        else:
            print("No images found in directory")
    
    else:
        print("Error: Please specify either --image or --directory")
        parser.print_help()
    
    print("=" * 100)


if __name__ == '__main__':
    main()
