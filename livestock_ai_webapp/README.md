# Livestock AI Web Application

A modern Django web application for livestock analysis using AI models for:
- **Weight Estimation**: Predict livestock weight from images
- **Disease Detection**: Detect lumpy skin disease and other health issues
- **Animal Classification**: Identify different types of livestock
- **Hoofed Animals Classification**: Multi-label classification for hoofed animals

## Features

- 🎨 **Modern UI**: Beautiful, responsive design with Bootstrap 5
- 🤖 **AI-Powered**: Multiple trained models for comprehensive analysis
- 📊 **Real-time Results**: Instant predictions with confidence scores
- 📈 **History Tracking**: View and analyze past predictions
- 📱 **Mobile Friendly**: Works on all devices

## Setup Instructions

### 1. Install Dependencies

```bash
# Activate your virtual environment
source vet_model_env/Scripts/activate

# Install Django and other requirements
pip install -r requirements.txt
```

### 2. Setup Django Project

```bash
# Navigate to the web app directory
cd livestock_ai_webapp

# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser
```

### 3. Copy Model Files

Make sure these model files are in the parent directory:
- `best_weight_model.pth`
- `best_classification_model.pth`
- `best_hoofed_animals_model.pth`
- `final_improved_model.keras`

### 4. Run the Application

```bash
# Start the development server
python manage.py runserver

# Open your browser and go to:
# http://127.0.0.1:8000
```

## Usage

1. **Upload Image**: Drag and drop or click to upload a livestock image
2. **Select Analysis Type**: Choose between all analysis, weight only, disease only, or classification only
3. **Get Results**: View detailed predictions with confidence scores
4. **View History**: Check your prediction history in the History tab

## Model Information

### Weight Estimation Model
- **Purpose**: Predict livestock weight from images
- **Input**: 224x224 RGB images
- **Output**: Weight in kilograms
- **Architecture**: CNN with 4 convolutional blocks

### Classification Model
- **Purpose**: Classify livestock into 4 categories
- **Classes**: Cattle, Sheep, Goats, Camels
- **Input**: 224x224 RGB images
- **Output**: Class probabilities

### Disease Detection Model
- **Purpose**: Detect lumpy skin disease
- **Classes**: Lumpy Skin, Normal Skin
- **Input**: 224x224 RGB images
- **Output**: Disease status with confidence

### Hoofed Animals Model
- **Purpose**: Multi-label classification for hoofed animals
- **Input**: 224x224 RGB images
- **Output**: Multiple class predictions

## Technical Details

- **Framework**: Django 4.2+
- **AI Models**: PyTorch + TensorFlow
- **Frontend**: Bootstrap 5 + jQuery
- **Database**: SQLite (default)
- **Image Processing**: OpenCV + Albumentations

## Troubleshooting

### Model Loading Issues
- Ensure model files are in the correct location
- Check file permissions
- Verify model file integrity

### Performance Issues
- Use GPU if available (CUDA)
- Reduce image size for faster processing
- Consider model optimization

### UI Issues
- Clear browser cache
- Check JavaScript console for errors
- Ensure all static files are served correctly

## License

This project is for educational and research purposes.
