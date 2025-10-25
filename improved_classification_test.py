#!/usr/bin/env python3
"""
Improved Classification Testing
Properly handles both livestock classification and hoofed animals classification models
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import time
from datetime import datetime

# EXACT model architectures from training scripts
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

class WeightEstimationModel(nn.Module):
    """CNN model for livestock weight estimation"""
    
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

class ImprovedModelTester:
    """Improved model tester with proper classification handling"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Class names
        self.livestock_classes = ['Cattle', 'Sheep', 'Goats', 'Camels']
        self.hoofed_animals_classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']  # Update based on your dataset
        
        print(f"🔧 Using device: {self.device}")
    
    def check_model_files(self):
        """Check which model files exist and prioritize best models"""
        print("📁 Checking for trained models...")
        
        # Define model files with priority (best models first)
        model_files = {
            'weight': 'best_weight_model.pth',
            'livestock_classification': 'best_classification_model.pth',  # Single-label classification
            'hoofed_animals': 'best_hoofed_animals_model.pth',  # Multi-label classification
        }
        
        available_models = {}
        
        for model_name, filename in model_files.items():
            if Path(filename).exists():
                file_size = Path(filename).stat().st_size / (1024 * 1024)  # MB
                mod_time = datetime.fromtimestamp(Path(filename).stat().st_mtime)
                available_models[model_name] = {
                    'file': filename,
                    'size': f"{file_size:.1f} MB",
                    'modified': mod_time.strftime("%Y-%m-%d %H:%M:%S")
                }
                print(f"✅ {model_name.replace('_', ' ').title()} model found: {filename} ({file_size:.1f} MB)")
            else:
                print(f"❌ {model_name.replace('_', ' ').title()} model not found: {filename}")
        
        return available_models
    
    def load_models(self, available_models):
        """Load available trained models with proper architectures"""
        print("\n🔄 Loading trained models...")
        
        # Load Weight Estimation Model
        if 'weight' in available_models:
            try:
                self.models['weight'] = WeightEstimationModel().to(self.device)
                self.models['weight'].load_state_dict(torch.load(available_models['weight']['file'], map_location=self.device))
                self.models['weight'].eval()
                print("✅ Weight estimation model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load weight model: {e}")
        
        # Load Livestock Classification Model (single-label)
        if 'livestock_classification' in available_models:
            try:
                self.models['livestock_classification'] = LivestockClassificationModel(num_classes=4).to(self.device)
                self.models['livestock_classification'].load_state_dict(torch.load(available_models['livestock_classification']['file'], map_location=self.device))
                self.models['livestock_classification'].eval()
                print("✅ Livestock classification model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load livestock classification model: {e}")
        
        # Load Hoofed Animals Model (multi-label)
        if 'hoofed_animals' in available_models:
            try:
                self.models['hoofed_animals'] = HoofedAnimalsModel(num_classes=6).to(self.device)
                self.models['hoofed_animals'].load_state_dict(torch.load(available_models['hoofed_animals']['file'], map_location=self.device))
                self.models['hoofed_animals'].eval()
                print("✅ Hoofed animals model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load hoofed animals model: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model inference"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor_image = transformed['image'].unsqueeze(0).to(self.device)
        
        return tensor_image, image
    
    def predict_weight(self, image_path):
        """Predict weight for an image"""
        if 'weight' not in self.models:
            return None, "Weight model not available"
        
        try:
            tensor_image, original_image = self.preprocess_image(image_path)
            
            with torch.no_grad():
                predicted_weight = self.models['weight'](tensor_image)
                predicted_weight = predicted_weight.item()
            
            return predicted_weight, None
        except Exception as e:
            return None, f"Error predicting weight: {e}"
    
    def predict_livestock_classification(self, image_path):
        """Predict livestock classification (single-label)"""
        if 'livestock_classification' not in self.models:
            return None, None, "Livestock classification model not available"
        
        try:
            tensor_image, original_image = self.preprocess_image(image_path)
            
            with torch.no_grad():
                outputs = self.models['livestock_classification'](tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            class_name = self.livestock_classes[predicted_class]
            return class_name, confidence, None
        except Exception as e:
            return None, None, f"Error predicting livestock classification: {e}"
    
    def predict_hoofed_animals(self, image_path):
        """Predict hoofed animals classification (multi-label)"""
        if 'hoofed_animals' not in self.models:
            return None, None, "Hoofed animals model not available"
        
        try:
            tensor_image, original_image = self.preprocess_image(image_path)
            
            with torch.no_grad():
                outputs = self.models['hoofed_animals'](tensor_image)
                probabilities = torch.sigmoid(outputs)  # Multi-label uses sigmoid
                
                # Get predictions above threshold
                threshold = 0.5
                predictions = (probabilities > threshold).float()
                
                # Get confidence scores
                confidences = probabilities[0].cpu().numpy()
                
            # Format results
            predicted_classes = []
            for i, (class_name, conf) in enumerate(zip(self.hoofed_animals_classes, confidences)):
                if predictions[0][i] > 0:
                    predicted_classes.append(f"{class_name} ({conf:.2f})")
            
            if not predicted_classes:
                predicted_classes = ["No clear classification"]
            
            return predicted_classes, confidences, None
        except Exception as e:
            return None, None, f"Error predicting hoofed animals: {e}"
    
    def test_all_models(self, image_path):
        """Test all available models on an image"""
        print(f"\n🧪 Testing all models on: {Path(image_path).name}")
        
        results = {
            'image_path': image_path,
            'image_name': Path(image_path).name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Weight prediction
        if 'weight' in self.models:
            weight, error = self.predict_weight(image_path)
            if error:
                results['weight'] = {'error': error}
            else:
                results['weight'] = {
                    'predicted_weight': f"{weight:.1f} kg",
                    'confidence': 'N/A'
                }
                print(f"⚖️ Predicted Weight: {weight:.1f} kg")
        
        # Livestock Classification prediction
        if 'livestock_classification' in self.models:
            class_name, confidence, error = self.predict_livestock_classification(image_path)
            if error:
                results['livestock_classification'] = {'error': error}
            else:
                results['livestock_classification'] = {
                    'predicted_class': class_name,
                    'confidence': f"{confidence*100:.1f}%"
                }
                print(f"🐄 Predicted Livestock Class: {class_name} ({confidence*100:.1f}%)")
        
        # Hoofed Animals prediction
        if 'hoofed_animals' in self.models:
            classes, confidences, error = self.predict_hoofed_animals(image_path)
            if error:
                results['hoofed_animals'] = {'error': error}
            else:
                results['hoofed_animals'] = {
                    'predicted_classes': classes,
                    'confidence_scores': [f"{c:.2f}" for c in confidences]
                }
                print(f"🦌 Predicted Hoofed Animals: {', '.join(classes)}")
        
        return results

class ImprovedModelTestGUI:
    """Improved GUI for testing models"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Livestock AI - Improved Model Testing")
        self.root.geometry("1000x800")
        
        self.tester = ImprovedModelTester()
        self.current_image_path = None
        
        self.setup_ui()
        self.load_available_models()
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Title
        title_label = tk.Label(self.root, text="🐄 Livestock AI - Improved Model Testing", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Model status frame
        self.status_frame = tk.Frame(self.root)
        self.status_frame.pack(pady=10, fill="x")
        
        # Upload section
        upload_frame = tk.Frame(self.root)
        upload_frame.pack(pady=20)
        
        upload_btn = tk.Button(upload_frame, text="📁 Upload Image", 
                             command=self.upload_image, 
                             font=("Arial", 12), bg="lightblue")
        upload_btn.pack(side="left", padx=10)
        
        self.image_label = tk.Label(upload_frame, text="No image selected", 
                                   font=("Arial", 10))
        self.image_label.pack(side="left", padx=10)
        
        # Test buttons
        test_frame = tk.Frame(self.root)
        test_frame.pack(pady=20)
        
        self.test_weight_btn = tk.Button(test_frame, text="⚖️ Test Weight", 
                                        command=self.test_weight, 
                                        font=("Arial", 10), bg="lightgreen",
                                        state="disabled")
        self.test_weight_btn.pack(side="left", padx=5)
        
        self.test_livestock_btn = tk.Button(test_frame, text="🐄 Test Livestock Class", 
                                          command=self.test_livestock_classification, 
                                          font=("Arial", 10), bg="lightyellow",
                                          state="disabled")
        self.test_livestock_btn.pack(side="left", padx=5)
        
        self.test_hoofed_btn = tk.Button(test_frame, text="🦌 Test Hoofed Animals", 
                                       command=self.test_hoofed_animals, 
                                       font=("Arial", 10), bg="lightcoral",
                                       state="disabled")
        self.test_hoofed_btn.pack(side="left", padx=5)
        
        self.test_all_btn = tk.Button(test_frame, text="🧪 Test All Models", 
                                    command=self.test_all, 
                                    font=("Arial", 10), bg="lightgray",
                                    state="disabled")
        self.test_all_btn.pack(side="left", padx=5)
        
        # Results display
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=20, fill="both", expand=True)
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=20, width=100, 
                                   font=("Courier", 10))
        self.results_text.pack(fill="both", expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side="right", fill="y")
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
    
    def load_available_models(self):
        """Load and display available models"""
        available_models = self.tester.check_model_files()
        self.tester.load_models(available_models)
        
        # Update status
        for widget in self.status_frame.winfo_children():
            widget.destroy()
        
        status_text = f"Available Models: {', '.join(available_models.keys())}"
        status_label = tk.Label(self.status_frame, text=status_text, font=("Arial", 10))
        status_label.pack()
    
    def upload_image(self):
        """Upload and display image"""
        file_path = filedialog.askopenfilename(
            title="Select Livestock Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            
            # Load and display image
            try:
                image = Image.open(file_path)
                image.thumbnail((200, 200), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
                
                # Enable test buttons
                self.test_weight_btn.configure(state="normal")
                self.test_livestock_btn.configure(state="normal")
                self.test_hoofed_btn.configure(state="normal")
                self.test_all_btn.configure(state="normal")
                
                self.log_result(f"📸 Image loaded: {Path(file_path).name}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def test_weight(self):
        """Test weight prediction"""
        if not self.current_image_path:
            return
        
        self.log_result(f"\n⚖️ Testing Weight Prediction...")
        weight, error = self.tester.predict_weight(self.current_image_path)
        
        if error:
            self.log_result(f"❌ Error: {error}")
        else:
            self.log_result(f"✅ Predicted Weight: {weight:.1f} kg")
    
    def test_livestock_classification(self):
        """Test livestock classification"""
        if not self.current_image_path:
            return
        
        self.log_result(f"\n🐄 Testing Livestock Classification...")
        class_name, confidence, error = self.tester.predict_livestock_classification(self.current_image_path)
        
        if error:
            self.log_result(f"❌ Error: {error}")
        else:
            self.log_result(f"✅ Predicted Class: {class_name} ({confidence*100:.1f}%)")
    
    def test_hoofed_animals(self):
        """Test hoofed animals classification"""
        if not self.current_image_path:
            return
        
        self.log_result(f"\n🦌 Testing Hoofed Animals Classification...")
        classes, confidences, error = self.tester.predict_hoofed_animals(self.current_image_path)
        
        if error:
            self.log_result(f"❌ Error: {error}")
        else:
            self.log_result(f"✅ Predicted Classes: {', '.join(classes)}")
    
    def test_all(self):
        """Test all models"""
        if not self.current_image_path:
            return
        
        self.log_result(f"\n🧪 Testing All Models on {Path(self.current_image_path).name}...")
        self.log_result("=" * 60)
        
        results = self.tester.test_all_models(self.current_image_path)
        
        # Display results
        for model_name, result in results.items():
            if model_name in ['image_path', 'image_name', 'timestamp']:
                continue
            
            if 'error' in result:
                self.log_result(f"❌ {model_name.replace('_', ' ').title()}: {result['error']}")
            else:
                for key, value in result.items():
                    self.log_result(f"✅ {model_name.replace('_', ' ').title()} - {key.replace('_', ' ').title()}: {value}")
    
    def log_result(self, message):
        """Log result to text widget"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()

def run_improved_gui():
    """Run improved GUI testing application"""
    root = tk.Tk()
    app = ImprovedModelTestGUI(root)
    root.mainloop()

def run_command_line_test():
    """Run command line testing"""
    print("🧪 IMPROVED LIVESTOCK AI - MODEL TESTING")
    print("=" * 60)
    
    tester = ImprovedModelTester()
    available_models = tester.check_model_files()
    tester.load_models(available_models)
    
    if not available_models:
        print("❌ No trained models found!")
        return
    
    while True:
        print(f"\n📁 Enter image path (or 'quit' to exit):")
        image_path = input("> ").strip()
        
        if image_path.lower() == 'quit':
            break
        
        if not Path(image_path).exists():
            print("❌ File not found! Please try again.")
            continue
        
        # Test all models
        results = tester.test_all_models(image_path)
        
        # Show results
        print(f"\n📊 Results for {Path(image_path).name}:")
        for model_name, result in results.items():
            if model_name in ['image_path', 'image_name', 'timestamp']:
                continue
            
            if 'error' in result:
                print(f"❌ {model_name.replace('_', ' ').title()}: {result['error']}")
            else:
                for key, value in result.items():
                    print(f"✅ {model_name.replace('_', ' ').title()} - {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    print("🚀 IMPROVED LIVESTOCK AI - MODEL TESTING")
    print("=" * 60)
    print("Choose your preferred interface:")
    print("1. GUI Application (recommended)")
    print("2. Command Line Interface")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("🖥️ Starting improved GUI application...")
        run_improved_gui()
    elif choice == "2":
        print("💻 Starting command line interface...")
        run_command_line_test()
    else:
        print("❌ Invalid choice. Starting GUI application...")
        run_improved_gui()
