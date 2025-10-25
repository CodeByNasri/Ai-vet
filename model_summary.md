# Livestock AI Models - Project Summary

## 🎯 Project Overview
This is a comprehensive livestock AI system with multiple trained models for:
- **Weight Estimation**: Predict livestock weight from images
- **Classification**: Identify livestock types (cattle, sheep, goats, camels)
- **Hoofed Animals Classification**: Multi-label classification for hoofed animals

## 🏗️ Environment Setup
✅ **Virtual Environment**: `vet_model_env` (already created and configured)
✅ **Dependencies**: All required packages installed
✅ **Python Version**: Compatible with the environment
✅ **PyTorch**: Version 2.9.0 (CPU version)

## 📦 Dependencies Installed
- **Core ML**: torch, torchvision, torchaudio
- **Computer Vision**: opencv-python, Pillow, albumentations
- **Data Science**: numpy, pandas, matplotlib, seaborn, scikit-learn
- **Built-in**: tkinter, pathlib, json, time, datetime, re, os

## 🤖 Available Trained Models

### 1. Weight Estimation Model
- **Files**: `best_weight_model.pth`
- **Size**: 9.99 MB
- **Parameters**: 2,617,665
- **Purpose**: Predict livestock weight from images
- **Architecture**: CNN with 4 convolutional blocks + classifier
- **Input**: 224x224 RGB images
- **Output**: Single weight value (kg)

### 2. Classification Model
- **Files**: `best_classification_model.pth`, `classification_model_final.pth`
- **Size**: 39.96 MB each
- **Parameters**: 10,471,304 each
- **Purpose**: Classify livestock into 4 categories (cattle, sheep, goats, camels)
- **Architecture**: Deep CNN with backbone + classifier
- **Input**: 224x224 RGB images
- **Output**: 4-class probabilities

### 3. Hoofed Animals Model
- **Files**: `best_hoofed_animals_model.pth`, `hoofed_animals_model_final.pth`
- **Size**: 10.0 MB each
- **Parameters**: 2,618,950 each
- **Purpose**: Multi-label classification for hoofed animals
- **Architecture**: CNN with 4 convolutional blocks
- **Input**: 224x224 RGB images
- **Output**: 6-class multi-label predictions

## 🚀 How to Run the Models

### Option 1: GUI Application (Recommended)
```bash
# Activate virtual environment
source vet_model_env/Scripts/activate

# Run the interactive testing application
python weight_testing.py
# Choose option 1 for GUI
```

### Option 2: Command Line Interface
```bash
# Activate virtual environment
source vet_model_env/Scripts/activate

# Run the command line interface
python weight_testing.py
# Choose option 2 for CLI
```

### Option 3: Test Environment
```bash
# Activate virtual environment
source vet_model_env/Scripts/activate

# Run environment test
python test_environment.py
```

## 📊 Model Performance Summary

| Model Type | File Size | Parameters | Layers | Status |
|------------|-----------|------------|--------|--------|
| Weight Estimation | 9.99 MB | 2,617,665 | 14 | ✅ Ready |
| Classification | 39.96 MB | 10,471,304 | 34 | ✅ Ready |
| Hoofed Animals | 10.0 MB | 2,618,950 | 14 | ✅ Ready |

## 🔧 Technical Details

### Model Architectures
1. **Weight Model**: Simple CNN with 4 conv blocks + 3-layer classifier
2. **Classification Model**: Deep CNN with backbone + batch normalization
3. **Hoofed Animals Model**: CNN with 4 conv blocks + multi-label output

### Data Preprocessing
- **Image Size**: 224x224 pixels
- **Normalization**: ImageNet mean/std values
- **Augmentation**: Random flips, rotations, brightness/contrast
- **Format**: RGB images

### Training Information
- **Framework**: PyTorch
- **Optimizer**: Adam/AdamW
- **Loss Functions**: CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
- **Training Date**: October 25, 2025
- **Device**: CPU (no GPU detected)

## 🎯 Usage Examples

### Weight Prediction
```python
# Load model
model = WeightEstimationModel()
model.load_state_dict(torch.load('best_weight_model.pth'))
model.eval()

# Predict weight
with torch.no_grad():
    prediction = model(image_tensor)
    weight_kg = prediction.item()
```

### Classification
```python
# Load model
model = LivestockClassificationModel(num_classes=4)
model.load_state_dict(torch.load('best_classification_model.pth'))
model.eval()

# Predict class
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
```

## 📁 Project Structure
```
Vet-Model-main/
├── vet_model_env/          # Virtual environment
├── requirements.txt         # Dependencies
├── *.pth files            # Trained models
├── train_*.py             # Training scripts
├── test_environment.py    # Environment test
├── weight_testing.py      # Model testing interface
└── analyze_models.py       # Model analysis script
```

## ✅ Ready to Use!
All models are trained, tested, and ready for inference. The environment is properly configured with all dependencies installed.
