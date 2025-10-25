# train_hoofed_animals_classification.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class HoofedAnimalsDataset(Dataset):
    """Dataset for HoofedAnimals classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

class HoofedAnimalsModel(nn.Module):
    """CNN model for hoofed animals classification"""
    
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

def load_hoofed_animals_dataset():
    """Load the HoofedAnimals dataset"""
    print("📁 Loading HoofedAnimals dataset...")
    
    dataset_path = Path("HoofedAnimals")
    org_path = dataset_path / "org"
    
    # Load ground truth
    labels = []
    with open(dataset_path / "ground_truth.txt", 'r') as f:
        for line in f:
            if line.strip():
                row = [int(x) for x in line.strip().split()]
                labels.append(row)
    
    # Get image paths
    image_paths = []
    for i in range(1, len(labels) + 1):
        # Try different extensions
        for ext in ['.pgm', '.tif']:
            img_path = org_path / f"{i}{ext}"
            if img_path.exists():
                image_paths.append(img_path)
                break
    
    print(f"📊 Dataset loaded:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Total labels: {len(labels)}")
    
    # Analyze class distribution
    all_values = set()
    for row in labels:
        for val in row:
            all_values.add(val)
    
    print(f"  Classes found: {sorted(all_values)}")
    
    # Count non-zero values in each column
    for col in range(len(labels[0])):
        non_zero = sum(1 for row in labels if row[col] != 0)
        print(f"  Column {col}: {non_zero} non-zero values")
    
    return image_paths, labels

def train_hoofed_animals_model():
    """Train the hoofed animals classification model"""
    print("🐄 TRAINING HOOFED ANIMALS CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    image_paths, labels = load_hoofed_animals_dataset()
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Split dataset
    split_idx = int(0.8 * len(image_paths))
    train_paths = image_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_paths = image_paths[split_idx:]
    val_labels = labels[split_idx:]
    
    # Create datasets
    train_dataset = HoofedAnimalsDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = HoofedAnimalsDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = HoofedAnimalsModel(num_classes=6).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    start_time = time.time()
    
    for epoch in range(30):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                predictions = torch.sigmoid(outputs) > 0.5
                total += targets.size(0) * targets.size(1)
                correct += (predictions == targets).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        val_acc = 100 * correct / total
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_hoofed_animals_model.pth')
        
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/30: Train Loss = {avg_train_loss:.4f}, Val Acc = {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'hoofed_animals_model_final.pth')
    print("💾 Models saved:")
    print("  - best_hoofed_animals_model.pth")
    print("  - hoofed_animals_model_final.pth")
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('hoofed_animals_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Training results saved as 'hoofed_animals_training_results.png'")

if __name__ == "__main__":
    train_hoofed_animals_model()
