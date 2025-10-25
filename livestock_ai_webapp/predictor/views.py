import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tensorflow as tf
from tensorflow import keras
import json
from .models import PredictionResult

# Model classes (same as your training scripts)
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

class LivestockClassificationModel(nn.Module):
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

# Global model instances
models_loaded = False
weight_model = None
classification_model = None
hoofed_animals_model = None
disease_model = None

def load_models():
    """Load all models into memory"""
    global models_loaded, weight_model, classification_model, hoofed_animals_model, disease_model
    
    if models_loaded:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load weight estimation model
        weight_model = WeightEstimationModel().to(device)
        weight_model.load_state_dict(torch.load(settings.MODEL_PATHS['weight_model'], map_location=device))
        weight_model.eval()
        print("✅ Weight model loaded")
    except Exception as e:
        print(f"❌ Failed to load weight model: {e}")
    
    try:
        # Load classification model
        classification_model = LivestockClassificationModel(num_classes=4).to(device)
        classification_model.load_state_dict(torch.load(settings.MODEL_PATHS['classification_model'], map_location=device))
        classification_model.eval()
        print("✅ Classification model loaded")
    except Exception as e:
        print(f"❌ Failed to load classification model: {e}")
    
    try:
        # Load hoofed animals model
        hoofed_animals_model = HoofedAnimalsModel(num_classes=6).to(device)
        hoofed_animals_model.load_state_dict(torch.load(settings.MODEL_PATHS['hoofed_animals_model'], map_location=device))
        hoofed_animals_model.eval()
        print("✅ Hoofed animals model loaded")
    except Exception as e:
        print(f"❌ Failed to load hoofed animals model: {e}")
    
    try:
        # Load disease detection model
        disease_model = keras.models.load_model(settings.MODEL_PATHS['disease_model'])
        print("✅ Disease model loaded")
    except Exception as e:
        print(f"❌ Failed to load disease model: {e}")
    
    models_loaded = True

def preprocess_image_pytorch(image_path, size=(224, 224)):
    """Preprocess image for PyTorch models"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(size[0], size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)

def preprocess_image_tensorflow(image_path, size=(224, 224)):
    """Preprocess image for TensorFlow models"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict_weight(image_path):
    """Predict weight using the weight estimation model"""
    if weight_model is None:
        return None, "Weight model not loaded"
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_image = preprocess_image_pytorch(image_path).to(device)
        
        with torch.no_grad():
            predicted_weight = weight_model(tensor_image)
            weight = predicted_weight.item()
        
        return {
            'weight_kg': round(weight, 1),
            'confidence': 'N/A'
        }, None
    except Exception as e:
        return None, str(e)

def predict_classification(image_path):
    """Predict animal classification"""
    if classification_model is None:
        return None, "Classification model not loaded"
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_image = preprocess_image_pytorch(image_path).to(device)
        
        class_names = ['Cattle', 'Sheep', 'Goats', 'Camels']
        
        with torch.no_grad():
            outputs = classification_model(tensor_image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': class_names[predicted_class],
            'confidence': round(confidence * 100, 1),
            'all_probabilities': {
                name: round(prob.item() * 100, 1) 
                for name, prob in zip(class_names, probabilities[0])
            }
        }, None
    except Exception as e:
        return None, str(e)

def predict_hoofed_animals(image_path):
    """Predict hoofed animals classification"""
    if hoofed_animals_model is None:
        return None, "Hoofed animals model not loaded"
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor_image = preprocess_image_pytorch(image_path).to(device)
        
        class_names = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']
        
        with torch.no_grad():
            outputs = hoofed_animals_model(tensor_image)
            probabilities = torch.sigmoid(outputs)
            predictions = probabilities > 0.5
        
        return {
            'predictions': {
                name: bool(pred.item()) 
                for name, pred in zip(class_names, predictions[0])
            },
            'probabilities': {
                name: round(prob.item() * 100, 1) 
                for name, prob in zip(class_names, probabilities[0])
            }
        }, None
    except Exception as e:
        return None, str(e)

def predict_disease(image_path):
    """Predict disease using the TensorFlow model"""
    if disease_model is None:
        return None, "Disease model not loaded"
    
    try:
        x = preprocess_image_tensorflow(image_path)
        preds = disease_model.predict(x, verbose=0)
        probs = preds[0]
        
        class_names = ["Lumpy Skin", "Normal Skin"]
        idx = int(np.argmax(probs))
        predicted_class = class_names[idx]
        confidence = float(probs[idx])
        
        return {
            'predicted_class': predicted_class,
            'confidence': round(confidence * 100, 1),
            'all_probabilities': {
                name: round(prob * 100, 1) 
                for name, prob in zip(class_names, probs)
            }
        }, None
    except Exception as e:
        return None, str(e)

def home(request):
    """Home page"""
    return render(request, 'predictor/home.html')

@csrf_exempt
def predict(request):
    """Handle prediction requests"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    
    # Load models if not already loaded
    load_models()
    
    # Get uploaded file
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image file provided'}, status=400)
    
    image_file = request.FILES['image']
    prediction_type = request.POST.get('type', 'all')
    
    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        for chunk in image_file.chunks():
            tmp_file.write(chunk)
        tmp_path = tmp_file.name
    
    try:
        results = {}
        
        # Weight prediction
        if prediction_type in ['all', 'weight']:
            weight_result, error = predict_weight(tmp_path)
            if error:
                results['weight'] = {'error': error}
            else:
                results['weight'] = weight_result
        
        # Classification prediction
        if prediction_type in ['all', 'classification']:
            classification_result, error = predict_classification(tmp_path)
            if error:
                results['classification'] = {'error': error}
            else:
                results['classification'] = classification_result
        
        # Hoofed animals prediction
        if prediction_type in ['all', 'hoofed']:
            hoofed_result, error = predict_hoofed_animals(tmp_path)
            if error:
                results['hoofed_animals'] = {'error': error}
            else:
                results['hoofed_animals'] = hoofed_result
        
        # Disease prediction
        if prediction_type in ['all', 'disease']:
            disease_result, error = predict_disease(tmp_path)
            if error:
                results['disease'] = {'error': error}
            else:
                results['disease'] = disease_result
        
        # Save result to database
        PredictionResult.objects.create(
            image_name=image_file.name,
            prediction_type=prediction_type,
            result=results,
            confidence=max([
                results.get(key, {}).get('confidence', 0) 
                for key in results 
                if isinstance(results[key], dict) and 'confidence' in results[key]
            ], default=0)
        )
        
        return JsonResponse({
            'success': True,
            'results': results,
            'image_name': image_file.name
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def history(request):
    """Show prediction history"""
    predictions = PredictionResult.objects.all()[:50]  # Last 50 predictions
    return render(request, 'predictor/history.html', {'predictions': predictions})
