# train_classification_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

class LivestockClassificationModel(nn.Module):
    """CNN model for livestock classification (cattle, sheep, goats, camels)"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Classification head
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

class LivestockClassificationDataset(Dataset):
    """Dataset for livestock classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

def create_synthetic_classification_dataset():
    """Create synthetic dataset for classification (since we only have cattle)"""
    print("📁 Creating synthetic classification dataset...")
    
    dataset_path = Path("Dataset - BMGF-LivestockWeight-CV")
    image_paths = []
    labels = []
    
    # Class mapping
    class_names = ['cattle', 'sheep', 'goats', 'camels']
    
    # Load all cattle images and assign them to cattle class
    cattle_paths = []
    for batch in ['B2', 'B3', 'B4']:
        batch_path = dataset_path / "Pixel" / batch
        
        if batch == 'B3':
            images_path = batch_path / "images"
            if images_path.exists():
                cattle_paths.extend(list(images_path.glob("*.jpg")))
        else:
            for view in ['Side', 'Rear', 'Back']:
                view_path = batch_path / view
                if view_path.exists():
                    images_path = view_path / "images"
                    if images_path.exists():
                        cattle_paths.extend(list(images_path.glob("*.jpg")))
    
    # Create synthetic multi-class dataset by splitting cattle images
    # This is a temporary solution - for production, collect real sheep/goat/camel images
    total_images = min(1000, len(cattle_paths))
    images_per_class = total_images // 4
    
    # Split cattle images into 4 "synthetic" classes
    for i, img_path in enumerate(cattle_paths[:total_images]):
        image_paths.append(img_path)
        # Assign different classes based on image index
        if i < images_per_class:
            labels.append(0)  # "cattle" class
        elif i < images_per_class * 2:
            labels.append(1)  # "sheep" class (synthetic)
        elif i < images_per_class * 3:
            labels.append(2)  # "goats" class (synthetic)
        else:
            labels.append(3)  # "camels" class (synthetic)
    
    print(f"📊 Synthetic Dataset created:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Cattle (synthetic): {labels.count(0)}")
    print(f"  Sheep (synthetic): {labels.count(1)}")
    print(f"  Goats (synthetic): {labels.count(2)}")
    print(f"  Camels (synthetic): {labels.count(3)}")
    print(f"  ⚠️  WARNING: This is synthetic data - all images are actually cattle!")
    print(f"  For production, collect real sheep, goats, and camel images")
    
    return image_paths, labels, class_names

def train_classification_model():
    """Train the livestock classification model"""
    print("🐄 TRAINING LIVESTOCK CLASSIFICATION MODEL")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    image_paths, labels, class_names = create_synthetic_classification_dataset()
    
    # Create transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomRotate90(p=0.2),
        A.ShiftScaleRotate(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📈 Data Split:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    
    # Create datasets
    train_dataset = LivestockClassificationDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = LivestockClassificationDataset(val_paths, val_labels, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=2)
    
    # Create model
    model = LivestockClassificationModel(num_classes=4).to(device)
    print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    print(f"\n🏋️ Starting training...")
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
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        val_acc = 100 * correct / total
        
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_classification_model.pth')
        
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/30: Train Loss = {avg_train_loss:.4f}, Val Acc = {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'classification_model_final.pth')
    
    # Test on validation set
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Classification Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('classification_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Final Results:")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved as 'best_classification_model.pth'")

if __name__ == "__main__":
    train_classification_model()