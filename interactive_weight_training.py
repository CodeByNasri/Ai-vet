# corrected_weight_test.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

class SimpleWeightModel(nn.Module):
    """Simple CNN for weight estimation"""
    
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

class LivestockWeightDataset(Dataset):
    """Dataset for livestock weight estimation"""
    
    def __init__(self, image_paths, weights, transform=None):
        self.image_paths = image_paths
        self.weights = weights
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load weight
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, weight

def extract_weight_from_filename(filename):
    """Extract weight from filename"""
    # Pattern: <id>_<side/rear>_<weight>_<sex>
    match = re.search(r'_(\d+)_[MF]', filename)
    if match:
        return int(match.group(1))
    return None

def load_dataset():
    """Load dataset with weights"""
    print("📁 Loading dataset...")
    
    dataset_path = Path("Dataset - BMGF-LivestockWeight-CV")
    image_paths = []
    weights = []
    
    # Process B3 batch
    b3_path = dataset_path / "Pixel" / "B3" / "images"
    if b3_path.exists():
        for img_path in b3_path.glob("*.jpg"):
            weight = extract_weight_from_filename(img_path.name)
            if weight is not None:
                image_paths.append(img_path)
                weights.append(weight)
    
    print(f"Found {len(image_paths)} images with weight data")
    
    if len(image_paths) == 0:
        print("❌ No images with weight data found!")
        return None, None
    
    # Show sample data
    print("📊 Sample data:")
    for i in range(min(5, len(image_paths))):
        print(f"  {image_paths[i].name} -> {weights[i]} kg")
    
    return image_paths, weights

def train_model(model, train_loader, device, epochs=20):
    """Train the weight estimation model"""
    print(f"\n🏋️ Training model for {epochs} epochs...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'best_weight_model.pth')
    print(f"✅ Training completed! Final loss: {train_losses[-1]:.4f}")
    print(f"💾 Model saved as 'best_weight_model.pth'")
    return train_losses

def test_trained_model(model, test_loader, device):
    """Test the trained model"""
    print("\n🧪 Testing trained model...")
    
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # FIX: Handle tensor conversion properly
            pred_np = outputs.squeeze().cpu().numpy()
            target_np = targets.cpu().numpy()
            
            # Ensure we're working with arrays
            if pred_np.ndim == 0:  # scalar
                test_predictions.append(pred_np.item())
                test_targets.append(target_np.item())
            else:  # array
                test_predictions.extend(pred_np.tolist())
                test_targets.extend(target_np.tolist())
    
    # Calculate metrics
    mae = mean_absolute_error(test_targets, test_predictions)
    r2 = 1 - (np.sum((np.array(test_targets) - np.array(test_predictions))**2) / 
              np.sum((np.array(test_targets) - np.mean(test_targets))**2))
    
    print(f"📊 Test Results:")
    print(f"  Mean Absolute Error: {mae:.2f} kg")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Average Weight: {np.mean(test_targets):.2f} kg")
    
    # Show sample predictions
    print(f"\n🔍 Sample Predictions:")
    for i in range(min(5, len(test_predictions))):
        actual = test_targets[i]
        predicted = test_predictions[i]
        error = abs(actual - predicted)
        print(f"  Actual: {actual:.0f}kg, Predicted: {predicted:.0f}kg, Error: {error:.0f}kg")
    
    return test_predictions, test_targets

def predict_single_image(model, image_path, device, transform):
    """Predict weight for a single image"""
    print(f"\n🖼️ Predicting weight for: {Path(image_path).name}")
    
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transformed = transform(image=image)
    tensor_image = transformed['image'].unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        predicted_weight = model(tensor_image).item()
    
    print(f"⚖️ Predicted Weight: {predicted_weight:.1f} kg")
    return predicted_weight

def main():
    """Main function to train and test the model"""
    print("🚀 TRAINED WEIGHT ESTIMATION TEST")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    image_paths, weights = load_dataset()
    if image_paths is None:
        return
    
    # Create transforms
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Split data
    train_paths, test_paths, train_weights, test_weights = train_test_split(
        image_paths, weights, test_size=0.2, random_state=42
    )
    
    print(f"\n📈 Data Split:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Testing: {len(test_paths)} images")
    
    # Create datasets
    train_dataset = LivestockWeightDataset(train_paths, train_weights, transform=transform)
    test_dataset = LivestockWeightDataset(test_paths, test_weights, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Create and train model
    model = SimpleWeightModel().to(device)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    train_losses = train_model(model, train_loader, device, epochs=20)
    
    # Test the model
    test_predictions, test_targets = test_trained_model(model, test_loader, device)
    
    # Test on a single image
    if test_paths:
        sample_image = test_paths[0]
        predicted_weight = predict_single_image(model, sample_image, device, transform)
        
        # Show results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.scatter(test_targets, test_predictions, alpha=0.6)
        plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], 'r--')
        plt.xlabel('Actual Weight (kg)')
        plt.ylabel('Predicted Weight (kg)')
        plt.title('Weight Prediction Results')
        
        plt.tight_layout()
        plt.savefig('trained_weight_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"\n✅ Model training and testing completed!")
    print(f"📈 Results saved to 'trained_weight_results.png'")
    
    return model, transform

if __name__ == "__main__":
    model, transform = main()