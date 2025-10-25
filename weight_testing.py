# test_trained_models.py
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

# Import the model classes (same as training scripts)
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

class LivestockClassificationModel(nn.Module):
    """CNN model for livestock classification"""
    
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

class DiseaseDetectionModel(nn.Module):
    """CNN model for disease detection"""
    
    def __init__(self, num_classes=2):
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
        
        self.disease_head = nn.Sequential(
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
        output = self.disease_head(features)
        return output

class ModelTester:
    """Main class for testing trained models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Class names
        self.classification_classes = ['Cattle', 'Sheep', 'Goats', 'Camels']
        self.disease_classes = ['Healthy', 'Diseased']
        
        print(f"🔧 Using device: {self.device}")
    
    def check_model_files(self):
        """Check which model files exist"""
        print("📁 Checking for trained models...")
        
        model_files = {
            'weight': 'best_weight_model.pth',
            'classification': ['best_hoofed_animals_model.pth', 'best_classification_model.pth'], 
            'disease': 'best_disease_model.pth'
        }
        
        available_models = {}
        
        for model_name, filename in model_files.items():
            # Handle both string and list filenames
            if isinstance(filename, list):
                # Check if any file in the list exists
                found_file = None
                for f in filename:
                    if Path(f).exists():
                        found_file = f
                        break
                if found_file:
                    file_size = Path(found_file).stat().st_size / (1024 * 1024)  # MB
                    mod_time = datetime.fromtimestamp(Path(found_file).stat().st_mtime)
                    available_models[model_name] = {
                        'file': found_file,
                        'size': f"{file_size:.1f} MB",
                        'modified': mod_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    print(f"✅ {model_name.title()} model found: {found_file} ({file_size:.1f} MB)")
                else:
                    print(f"❌ {model_name.title()} model not found: {filename}")
            else:
                # Handle single filename
                if Path(filename).exists():
                    file_size = Path(filename).stat().st_size / (1024 * 1024)  # MB
                    mod_time = datetime.fromtimestamp(Path(filename).stat().st_mtime)
                    available_models[model_name] = {
                        'file': filename,
                        'size': f"{file_size:.1f} MB",
                        'modified': mod_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    print(f"✅ {model_name.title()} model found: {filename} ({file_size:.1f} MB)")
                else:
                    print(f"❌ {model_name.title()} model not found: {filename}")
        
        return available_models
    
    def load_models(self, available_models):
        """Load available trained models"""
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
        
        # Load Classification Model
        if 'classification' in available_models:
            try:
                self.models['classification'] = LivestockClassificationModel(num_classes=4).to(self.device)
                self.models['classification'].load_state_dict(torch.load(available_models['classification']['file'], map_location=self.device))
                self.models['classification'].eval()
                print("✅ Classification model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load classification model: {e}")
        
        # Load Disease Detection Model
        if 'disease' in available_models:
            try:
                self.models['disease'] = DiseaseDetectionModel(num_classes=2).to(self.device)
                self.models['disease'].load_state_dict(torch.load(available_models['disease']['file'], map_location=self.device))
                self.models['disease'].eval()
                print("✅ Disease detection model loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load disease model: {e}")
    
    def preprocess_image(self, image_path):
        """Preprocess image for model inference"""
        # Load image
        image = cv2.imread(str(image_path))
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
    
    def predict_classification(self, image_path):
        """Predict livestock classification"""
        if 'classification' not in self.models:
            return None, None, "Classification model not available"
        
        try:
            tensor_image, original_image = self.preprocess_image(image_path)
            
            with torch.no_grad():
                outputs = self.models['classification'](tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            class_name = self.classification_classes[predicted_class]
            return class_name, confidence, None
        except Exception as e:
            return None, None, f"Error predicting classification: {e}"
    
    def predict_disease(self, image_path):
        """Predict disease status"""
        if 'disease' not in self.models:
            return None, None, "Disease model not available"
        
        try:
            tensor_image, original_image = self.preprocess_image(image_path)
            
            with torch.no_grad():
                outputs = self.models['disease'](tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            class_name = self.disease_classes[predicted_class]
            return class_name, confidence, None
        except Exception as e:
            return None, None, f"Error predicting disease: {e}"
    
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
        
        # Classification prediction
        if 'classification' in self.models:
            class_name, confidence, error = self.predict_classification(image_path)
            if error:
                results['classification'] = {'error': error}
            else:
                results['classification'] = {
                    'predicted_class': class_name,
                    'confidence': f"{confidence*100:.1f}%"
                }
                print(f"🐄 Predicted Class: {class_name} ({confidence*100:.1f}%)")
        
        # Disease prediction
        if 'disease' in self.models:
            disease_status, confidence, error = self.predict_disease(image_path)
            if error:
                results['disease'] = {'error': error}
            else:
                results['disease'] = {
                    'predicted_status': disease_status,
                    'confidence': f"{confidence*100:.1f}%"
                }
                print(f"🏥 Predicted Disease Status: {disease_status} ({confidence*100:.1f}%)")
        
        return results

class ModelTestGUI:
    """GUI for testing models"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Livestock AI - Model Testing")
        self.root.geometry("900x700")
        
        self.tester = ModelTester()
        self.current_image_path = None
        
        self.setup_ui()
        self.load_available_models()
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Title
        title_label = tk.Label(self.root, text="🐄 Livestock AI - Model Testing", 
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
        
        self.test_classification_btn = tk.Button(test_frame, text="🐄 Test Classification", 
                                               command=self.test_classification, 
                                               font=("Arial", 10), bg="lightyellow",
                                               state="disabled")
        self.test_classification_btn.pack(side="left", padx=5)
        
        self.test_disease_btn = tk.Button(test_frame, text="🏥 Test Disease", 
                                        command=self.test_disease, 
                                        font=("Arial", 10), bg="lightcoral",
                                        state="disabled")
        self.test_disease_btn.pack(side="left", padx=5)
        
        self.test_all_btn = tk.Button(test_frame, text="🧪 Test All Models", 
                                    command=self.test_all, 
                                    font=("Arial", 10), bg="lightgray",
                                    state="disabled")
        self.test_all_btn.pack(side="left", padx=5)
        
        # Results display
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=20, fill="both", expand=True)
        
        # Results text
        self.results_text = tk.Text(self.results_frame, height=15, width=80, 
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
        
        status_label = tk.Label(self.status_frame, 
                               text=f"Available Models: {', '.join(available_models.keys())}", 
                               font=("Arial", 10))
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
                self.test_classification_btn.configure(state="normal")
                self.test_disease_btn.configure(state="normal")
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
    
    def test_classification(self):
        """Test classification prediction"""
        if not self.current_image_path:
            return
        
        self.log_result(f"\n🐄 Testing Classification...")
        class_name, confidence, error = self.tester.predict_classification(self.current_image_path)
        
        if error:
            self.log_result(f"❌ Error: {error}")
        else:
            self.log_result(f"✅ Predicted Class: {class_name} ({confidence*100:.1f}%)")
    
    def test_disease(self):
        """Test disease prediction"""
        if not self.current_image_path:
            return
        
        self.log_result(f"\n🏥 Testing Disease Detection...")
        disease_status, confidence, error = self.tester.predict_disease(self.current_image_path)
        
        if error:
            self.log_result(f"❌ Error: {error}")
        else:
            self.log_result(f"✅ Predicted Status: {disease_status} ({confidence*100:.1f}%)")
    
    def test_all(self):
        """Test all models"""
        if not self.current_image_path:
            return
        
        self.log_result(f"\n🧪 Testing All Models on {Path(self.current_image_path).name}...")
        self.log_result("=" * 50)
        
        results = self.tester.test_all_models(self.current_image_path)
        
        # Display results
        for model_name, result in results.items():
            if model_name in ['image_path', 'image_name', 'timestamp']:
                continue
            
            if 'error' in result:
                self.log_result(f"❌ {model_name.title()}: {result['error']}")
            else:
                for key, value in result.items():
                    self.log_result(f"✅ {model_name.title()} - {key.replace('_', ' ').title()}: {value}")
    
    def log_result(self, message):
        """Log result to text widget"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()

def run_gui_test():
    """Run GUI testing application"""
    root = tk.Tk()
    app = ModelTestGUI(root)
    root.mainloop()

def run_command_line_test():
    """Run command line testing"""
    print("🧪 LIVESTOCK AI - MODEL TESTING")
    print("=" * 50)
    
    tester = ModelTester()
    available_models = tester.check_model_files()
    tester.load_models(available_models)
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please train models first using:")
        print("  python train_weight_model.py")
        print("  python train_classification_model.py")
        print("  python train_disease_model.py")
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
                print(f"❌ {model_name.title()}: {result['error']}")
            else:
                for key, value in result.items():
                    print(f"✅ {model_name.title()} - {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    print("🚀 LIVESTOCK AI - MODEL TESTING")
    print("=" * 60)
    print("Choose your preferred interface:")
    print("1. GUI Application (recommended)")
    print("2. Command Line Interface")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("🖥️ Starting GUI application...")
        run_gui_test()
    elif choice == "2":
        print("💻 Starting command line interface...")
        run_command_line_test()
    else:
        print("❌ Invalid choice. Starting GUI application...")
        run_gui_test()