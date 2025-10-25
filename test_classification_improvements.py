#!/usr/bin/env python3
"""
Test Classification Improvements
Demonstrates the improved classification system
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# EXACT model architectures
class LivestockClassificationModel(nn.Module):
    """CNN model for livestock classification (cattle, sheep, goats, camels)"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class HoofedAnimalsModel(nn.Module):
    """CNN model for hoofed animals classification (multi-label)"""
    
    def __init__(self, num_classes=6):
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
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def test_classification_models():
    """Test the improved classification models"""
    print("🔍 TESTING IMPROVED CLASSIFICATION MODELS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Image preprocessing
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Check available models
    model_files = {
        'livestock_classification': 'best_classification_model.pth',
        'hoofed_animals': 'best_hoofed_animals_model.pth'
    }
    
    available_models = {}
    for model_name, filename in model_files.items():
        if Path(filename).exists():
            available_models[model_name] = filename
            print(f"✅ {model_name.replace('_', ' ').title()} model found: {filename}")
        else:
            print(f"❌ {model_name.replace('_', ' ').title()} model not found: {filename}")
    
    if not available_models:
        print("❌ No classification models found!")
        return
    
    # Load and test models
    models = {}
    
    # Load Livestock Classification Model
    if 'livestock_classification' in available_models:
        try:
            print(f"\n🔄 Loading livestock classification model...")
            model = LivestockClassificationModel(num_classes=4).to(device)
            model.load_state_dict(torch.load(available_models['livestock_classification'], map_location=device))
            model.eval()
            models['livestock'] = model
            print("✅ Livestock classification model loaded successfully")
            
            # Test with dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(dummy_input)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            livestock_classes = ['Cattle', 'Sheep', 'Goats', 'Camels']
            print(f"✅ Test prediction: {livestock_classes[predicted_class]} ({confidence*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Failed to load livestock classification model: {e}")
    
    # Load Hoofed Animals Model
    if 'hoofed_animals' in available_models:
        try:
            print(f"\n🔄 Loading hoofed animals model...")
            model = HoofedAnimalsModel(num_classes=6).to(device)
            model.load_state_dict(torch.load(available_models['hoofed_animals'], map_location=device))
            model.eval()
            models['hoofed'] = model
            print("✅ Hoofed animals model loaded successfully")
            
            # Test with dummy input
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(dummy_input)
                probabilities = torch.sigmoid(output)  # Multi-label uses sigmoid
                predictions = (probabilities > 0.5).float()
            
            print(f"✅ Test prediction: Multi-label output with {predictions.sum().item()} active classes")
            
        except Exception as e:
            print(f"❌ Failed to load hoofed animals model: {e}")
    
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    print(f"Models loaded: {len(models)}")
    print(f"Available models: {list(models.keys())}")
    
    if 'livestock' in models:
        print("✅ Livestock classification: Single-label (Cattle, Sheep, Goats, Camels)")
    if 'hoofed' in models:
        print("✅ Hoofed animals: Multi-label (6 classes)")
    
    print(f"\n🎯 IMPROVEMENTS MADE:")
    print("1. ✅ Separate model architectures for different tasks")
    print("2. ✅ Proper single-label vs multi-label handling")
    print("3. ✅ Correct activation functions (softmax vs sigmoid)")
    print("4. ✅ Better error handling and model loading")
    print("5. ✅ Clear distinction between model types")
    
    return models

def test_with_real_image(models, image_path):
    """Test models with a real image"""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n🖼️ Testing with real image: {Path(image_path).name}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Image preprocessing
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformed = transform(image=image)
    tensor_image = transformed['image'].unsqueeze(0).to(device)
    
    # Test livestock classification
    if 'livestock' in models:
        try:
            with torch.no_grad():
                outputs = models['livestock'](tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            livestock_classes = ['Cattle', 'Sheep', 'Goats', 'Camels']
            print(f"🐄 Livestock Classification: {livestock_classes[predicted_class]} ({confidence*100:.1f}%)")
        except Exception as e:
            print(f"❌ Livestock classification error: {e}")
    
    # Test hoofed animals classification
    if 'hoofed' in models:
        try:
            with torch.no_grad():
                outputs = models['hoofed'](tensor_image)
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
            print(f"🦌 Hoofed Animals: {predictions.sum().item()} active classes")
        except Exception as e:
            print(f"❌ Hoofed animals classification error: {e}")

def main():
    """Main function"""
    print("🚀 IMPROVED CLASSIFICATION TESTING")
    print("=" * 50)
    
    # Test model loading and architecture
    models = test_classification_models()
    
    # Test with sample image if available
    sample_images = [
        "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images/1_side_450_M.jpg",
        "Dataset - BMGF-LivestockWeight-CV/Pixel/B3/images/2_side_380_F.jpg"
    ]
    
    for image_path in sample_images:
        if Path(image_path).exists():
            test_with_real_image(models, image_path)
            break
    else:
        print("\n💡 No sample images found for testing")
        print("To test with real images, provide a valid image path")

if __name__ == "__main__":
    main()
