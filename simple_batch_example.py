#!/usr/bin/env python3
"""
Simple example: Use your weight estimation model for multiple images
No additional training needed - your existing model works perfectly for batch processing!
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Your exact model architecture (same as training)
class WeightEstimationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def predict_multiple_images(image_paths, model_path='best_weight_model.pth'):
    """
    Predict weights for multiple images using your trained model
    
    Args:
        image_paths: List of image file paths
        model_path: Path to your trained model
    
    Returns:
        List of dictionaries with predictions
    """
    print(f"🔄 Processing {len(image_paths)} images...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load your trained model
    model = WeightEstimationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Image preprocessing (same as training)
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    results = []
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        try:
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            transformed = transform(image=image)
            tensor_image = transformed['image'].unsqueeze(0).to(device)  # Add batch dimension
            
            # Predict weight
            with torch.no_grad():
                predicted_weight = model(tensor_image).item()
            
            results.append({
                'image_name': Path(image_path).name,
                'predicted_weight': round(predicted_weight, 1)
            })
            
            print(f"  ✅ {Path(image_path).name}: {predicted_weight:.1f} kg")
            
        except Exception as e:
            print(f"  ❌ Error processing {Path(image_path).name}: {e}")
            results.append({
                'image_name': Path(image_path).name,
                'predicted_weight': None,
                'error': str(e)
            })
    
    return results

def predict_batch_efficient(image_paths, model_path='best_weight_model.pth', batch_size=4):
    """
    More efficient batch processing - processes multiple images at once
    This is faster for large numbers of images
    """
    print(f"🚀 Efficient batch processing: {len(image_paths)} images (batch size: {batch_size})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = WeightEstimationModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Preprocessing
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    results = []
    
    # Process in batches
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_paths = image_paths[batch_start:batch_end]
        
        print(f"  Processing batch {batch_start//batch_size + 1}: {len(batch_paths)} images")
        
        # Preprocess batch
        batch_tensors = []
        valid_paths = []
        
        for image_path in batch_paths:
            try:
                image = cv2.imread(str(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = transform(image=image)
                batch_tensors.append(transformed['image'])
                valid_paths.append(image_path)
            except Exception as e:
                print(f"    ⚠️ Skipping {Path(image_path).name}: {e}")
                continue
        
        if not batch_tensors:
            continue
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        # Predict for entire batch
        with torch.no_grad():
            predicted_weights = model(batch_tensor).squeeze().cpu().numpy()
        
        # Handle single image case
        if predicted_weights.ndim == 0:
            predicted_weights = [predicted_weights.item()]
        
        # Store results
        for path, weight in zip(valid_paths, predicted_weights):
            results.append({
                'image_name': Path(path).name,
                'predicted_weight': round(weight, 1)
            })
            print(f"    ✅ {Path(path).name}: {weight:.1f} kg")
    
    return results

def main():
    """Example usage"""
    print("🎯 MULTIPLE IMAGE WEIGHT PREDICTION")
    print("=" * 50)
    print("✅ Your model can handle multiple images without any additional training!")
    print("✅ Just process them one by one or in batches for efficiency")
    print()
    
    # Example 1: Process specific images
    print("📁 Example 1: Specific image files")
    image_paths = [
        "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images/1_side_450_M.jpg",
        "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images/2_side_380_F.jpg",
        "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images/3_side_520_M.jpg"
    ]
    
    # Filter to existing files
    existing_images = [img for img in image_paths if Path(img).exists()]
    
    if existing_images:
        print(f"Found {len(existing_images)} existing images")
        results = predict_multiple_images(existing_images)
        
        print(f"\n📊 Results:")
        for result in results:
            if result['predicted_weight'] is not None:
                print(f"  {result['image_name']}: {result['predicted_weight']} kg")
            else:
                print(f"  {result['image_name']}: Error - {result.get('error', 'Unknown')}")
    else:
        print("No sample images found in the dataset folder")
    
    # Example 2: Process all images in a folder
    print(f"\n📁 Example 2: Process entire folder")
    dataset_folder = Path("Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images")
    
    if dataset_folder.exists():
        # Get all image files
        image_files = list(dataset_folder.glob("*.jpg"))[:10]  # Limit to first 10 for demo
        
        if image_files:
            print(f"Processing {len(image_files)} images from folder...")
            results = predict_batch_efficient([str(f) for f in image_files], batch_size=4)
            
            # Show summary
            weights = [r['predicted_weight'] for r in results if r['predicted_weight'] is not None]
            if weights:
                print(f"\n📊 Summary:")
                print(f"  Average weight: {np.mean(weights):.1f} kg")
                print(f"  Weight range: {np.min(weights):.1f} - {np.max(weights):.1f} kg")
        else:
            print("No images found in folder")
    else:
        print(f"Dataset folder not found: {dataset_folder}")
    
    print(f"\n✅ Multiple image prediction completed!")
    print(f"💡 Your model works perfectly for batch processing - no retraining needed!")

if __name__ == "__main__":
    main()
