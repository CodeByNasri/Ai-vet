"""
Machine Learning Models Manager
Handles loading and inference for all trained models
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf
from tensorflow import keras
from django.conf import settings
import io

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

class ModelManager:
    """Manages all ML models and predictions"""
    
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
        self.hoofed_animals_classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
        self.disease_classes = ['Lumpy Skin', 'Normal Skin']
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        try:
            # Load weight estimation model
            if settings.MODEL_PATHS['weight_model'].exists():
                self.models['weight'] = WeightEstimationModel().to(self.device)
                self.models['weight'].load_state_dict(
                    torch.load(settings.MODEL_PATHS['weight_model'], map_location=self.device)
                )
                self.models['weight'].eval()
                print("✅ Weight model loaded")
            
            # Load classification model
            if settings.MODEL_PATHS['classification_model'].exists():
                self.models['classification'] = LivestockClassificationModel(num_classes=4).to(self.device)
                self.models['classification'].load_state_dict(
                    torch.load(settings.MODEL_PATHS['classification_model'], map_location=self.device)
                )
                self.models['classification'].eval()
                print("✅ Classification model loaded")
            
            # Load hoofed animals model
            if settings.MODEL_PATHS['hoofed_animals_model'].exists():
                self.models['hoofed_animals'] = HoofedAnimalsModel(num_classes=6).to(self.device)
                self.models['hoofed_animals'].load_state_dict(
                    torch.load(settings.MODEL_PATHS['hoofed_animals_model'], map_location=self.device)
                )
                self.models['hoofed_animals'].eval()
                print("✅ Hoofed animals model loaded")
            
            # Load disease detection model
            if settings.MODEL_PATHS['disease_model'].exists():
                self.models['disease'] = keras.models.load_model(settings.MODEL_PATHS['disease_model'])
                print("✅ Disease model loaded")
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def _preprocess_image(self, image_file):
        """Preprocess image for PyTorch models"""
        # Read image from file
        image_bytes = image_file.read()
        image_file.seek(0)  # Reset file pointer
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image)
        tensor_image = transformed['image'].unsqueeze(0).to(self.device)
        
        return tensor_image, image
    
    def _preprocess_image_tf(self, image_file):
        """Preprocess image for TensorFlow models"""
        # Read image
        image_bytes = image_file.read()
        image_file.seek(0)  # Reset file pointer
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image).astype('float32')
        image_array = np.expand_dims(image_array, axis=0)
        
        # Preprocess for EfficientNet
        image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
        
        return image_array
    
    def predict_weight(self, image_file):
        """Predict weight for an image"""
        if 'weight' not in self.models:
            raise Exception("Weight model not available")
        
        try:
            tensor_image, original_image = self._preprocess_image(image_file)
            
            with torch.no_grad():
                predicted_weight = self.models['weight'](tensor_image)
                predicted_weight = predicted_weight.item()
            
            return {
                'predicted_weight': round(predicted_weight, 1),
                'confidence': 'N/A',
                'model_type': 'Weight Estimation',
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def predict_classification(self, image_file):
        """Predict animal classification"""
        if 'classification' not in self.models:
            raise Exception("Classification model not available")
        
        try:
            tensor_image, original_image = self._preprocess_image(image_file)
            
            with torch.no_grad():
                outputs = self.models['classification'](tensor_image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()
            
            predicted_class = self.classification_classes[predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.classification_classes):
                class_probabilities[class_name] = round(float(probabilities[0][i].item()) * 100, 2)
            
            return {
                'predicted_class': predicted_class,
                'confidence': round(float(confidence) * 100, 2),
                'class_probabilities': class_probabilities,
                'model_type': 'Animal Classification',
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def predict_hoofed_animals(self, image_file):
        """Predict hoofed animals classification"""
        if 'hoofed_animals' not in self.models:
            raise Exception("Hoofed animals model not available")
        
        try:
            tensor_image, original_image = self._preprocess_image(image_file)
            
            with torch.no_grad():
                outputs = self.models['hoofed_animals'](tensor_image)
                probabilities = torch.sigmoid(outputs)
                predicted_classes = (probabilities > 0.5).float()
            
            # Get class predictions
            class_predictions = {}
            for i, class_name in enumerate(self.hoofed_animals_classes):
                class_predictions[class_name] = {
                    'predicted': bool(predicted_classes[0][i].item()),
                    'confidence': round(float(probabilities[0][i].item()) * 100, 2)
                }
            
            return {
                'class_predictions': class_predictions,
                'model_type': 'Hoofed Animals Classification',
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def predict_disease(self, image_file):
        """Predict disease detection"""
        if 'disease' not in self.models:
            raise Exception("Disease model not available")
        
        try:
            image_array = self._preprocess_image_tf(image_file)
            
            predictions = self.models['disease'].predict(image_array, verbose=0)
            probabilities = predictions[0]
            
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.disease_classes[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]
            
            # Get all class probabilities
            class_probabilities = {}
            for i, class_name in enumerate(self.disease_classes):
                class_probabilities[class_name] = round(float(probabilities[i]) * 100, 2)
            
            return {
                'predicted_class': predicted_class,
                'confidence': round(float(confidence) * 100, 2),
                'class_probabilities': class_probabilities,
                'model_type': 'Disease Detection',
                'status': 'success'
            }
        except Exception as e:
            return {
                'error': str(e),
                'status': 'error'
            }
