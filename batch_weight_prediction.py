#!/usr/bin/env python3
"""
Batch Weight Prediction for Multiple Images
Uses the trained weight estimation model to predict weights for multiple images at once
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

class WeightEstimationModel(nn.Module):
    """CNN model for livestock weight estimation - EXACT MATCH to training"""
    
    def __init__(self):
        super().__init__()
        
        # EXACT architecture from training script
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

class BatchWeightPredictor:
    """Class for batch weight prediction on multiple images"""
    
    def __init__(self, model_path='best_weight_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Image preprocessing transform
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Load the trained model
        self.load_model()
        
        print(f"🔧 Using device: {self.device}")
        print(f"✅ Model loaded from: {model_path}")
    
    def load_model(self):
        """Load the trained weight estimation model"""
        try:
            self.model = WeightEstimationModel().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            print("✅ Weight estimation model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess a single image for prediction"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor_image = transformed['image']
        
        return tensor_image
    
    def predict_single(self, image_path):
        """Predict weight for a single image"""
        try:
            tensor_image = self.preprocess_image(image_path)
            tensor_image = tensor_image.unsqueeze(0).to(self.device)  # Add batch dimension
            
            with torch.no_grad():
                predicted_weight = self.model(tensor_image)
                predicted_weight = predicted_weight.item()
            
            return predicted_weight, None
        except Exception as e:
            return None, str(e)
    
    def predict_batch(self, image_paths, batch_size=8):
        """Predict weights for multiple images in batches"""
        print(f"🔄 Processing {len(image_paths)} images in batches of {batch_size}")
        
        results = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            print(f"  Processing batch {batch_idx + 1}/{total_batches} ({len(batch_paths)} images)")
            
            # Process batch
            batch_results = self.process_batch(batch_paths)
            results.extend(batch_results)
        
        return results
    
    def process_batch(self, image_paths):
        """Process a batch of images"""
        batch_tensors = []
        valid_paths = []
        
        # Preprocess all images in the batch
        for image_path in image_paths:
            try:
                tensor_image = self.preprocess_image(image_path)
                batch_tensors.append(tensor_image)
                valid_paths.append(image_path)
            except Exception as e:
                print(f"    ⚠️ Skipping {Path(image_path).name}: {e}")
                continue
        
        if not batch_tensors:
            return []
        
        # Stack tensors into batch
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        # Predict weights for the entire batch
        with torch.no_grad():
            predicted_weights = self.model(batch_tensor)
            predicted_weights = predicted_weights.squeeze().cpu().numpy()
        
        # Handle single image case
        if predicted_weights.ndim == 0:
            predicted_weights = [predicted_weights.item()]
        
        # Create results
        batch_results = []
        for i, (image_path, weight) in enumerate(zip(valid_paths, predicted_weights)):
            batch_results.append({
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'predicted_weight': round(weight, 1),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return batch_results
    
    def predict_folder(self, folder_path, batch_size=8, image_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        """Predict weights for all images in a folder"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find all image files
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(folder_path.glob(f"*{ext}"))
            image_paths.extend(folder_path.glob(f"*{ext.upper()}"))
        
        image_paths = sorted(list(set(image_paths)))  # Remove duplicates and sort
        
        if not image_paths:
            print(f"❌ No images found in {folder_path}")
            return []
        
        print(f"📁 Found {len(image_paths)} images in {folder_path}")
        return self.predict_batch(image_paths, batch_size)
    
    def save_results(self, results, output_file='weight_predictions.csv'):
        """Save prediction results to CSV file"""
        if not results:
            print("❌ No results to save")
            return
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        # Show summary
        weights = [r['predicted_weight'] for r in results]
        print(f"\n📊 Summary:")
        print(f"  Total images: {len(results)}")
        print(f"  Average weight: {np.mean(weights):.1f} kg")
        print(f"  Min weight: {np.min(weights):.1f} kg")
        print(f"  Max weight: {np.max(weights):.1f} kg")
        print(f"  Weight range: {np.max(weights) - np.min(weights):.1f} kg")
    
    def plot_results(self, results, output_file='weight_predictions.png'):
        """Create visualization of weight predictions"""
        if not results:
            print("❌ No results to plot")
            return
        
        weights = [r['predicted_weight'] for r in results]
        
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Weight distribution
        plt.subplot(2, 2, 1)
        plt.hist(weights, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Weight Distribution')
        plt.xlabel('Predicted Weight (kg)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Weight vs Image Index
        plt.subplot(2, 2, 2)
        plt.plot(weights, 'o-', alpha=0.7, color='green')
        plt.title('Weight Predictions by Image')
        plt.xlabel('Image Index')
        plt.ylabel('Predicted Weight (kg)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(weights, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.title('Weight Distribution Box Plot')
        plt.ylabel('Predicted Weight (kg)')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Statistics
        plt.subplot(2, 2, 4)
        stats_text = f"""
        Statistics:
        Count: {len(weights)}
        Mean: {np.mean(weights):.1f} kg
        Median: {np.median(weights):.1f} kg
        Std: {np.std(weights):.1f} kg
        Min: {np.min(weights):.1f} kg
        Max: {np.max(weights):.1f} kg
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {output_file}")

def main():
    """Main function to demonstrate batch weight prediction"""
    print("🚀 BATCH WEIGHT PREDICTION")
    print("=" * 50)
    
    # Initialize predictor
    predictor = BatchWeightPredictor()
    
    # Example 1: Predict weights for a list of specific images
    print("\n📁 Example 1: Specific image files")
    sample_images = [
        "path/to/image1.jpg",
        "path/to/image2.jpg", 
        "path/to/image3.jpg"
    ]
    
    # Filter to only existing files for demo
    existing_images = [img for img in sample_images if Path(img).exists()]
    
    if existing_images:
        results = predictor.predict_batch(existing_images)
        for result in results:
            print(f"  {result['image_name']}: {result['predicted_weight']} kg")
    else:
        print("  No sample images found, skipping...")
    
    # Example 2: Predict weights for all images in a folder
    print("\n📁 Example 2: Process entire folder")
    dataset_folder = "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images"
    
    if Path(dataset_folder).exists():
        results = predictor.predict_folder(dataset_folder, batch_size=4)
        
        if results:
            # Save results
            predictor.save_results(results, 'batch_weight_predictions.csv')
            
            # Create visualization
            predictor.plot_results(results, 'batch_weight_visualization.png')
        else:
            print("  No images processed")
    else:
        print(f"  Dataset folder not found: {dataset_folder}")
        print("  Please provide a valid image folder path")
    
    print("\n✅ Batch prediction completed!")

if __name__ == "__main__":
    main()
