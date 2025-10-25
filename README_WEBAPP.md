# 🐄 Livestock AI Web Application

A modern Django web application that integrates all your trained machine learning models for comprehensive livestock analysis.

## 🚀 Features

### 🤖 Integrated ML Models
- **Weight Estimation**: Predict livestock weight from images using PyTorch CNN
- **Animal Classification**: Identify livestock types (cattle, sheep, goats, camels)
- **Hoofed Animals**: Multi-label classification for detailed analysis
- **Disease Detection**: Early detection of lumpy skin disease using TensorFlow

### 🎨 Modern Web Interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Bootstrap 5**: Modern, clean UI with smooth animations
- **Drag & Drop**: Easy image upload with drag-and-drop support
- **Real-time Results**: Instant predictions with confidence scores
- **History Tracking**: View all past predictions and results

### 🔧 Technical Features
- **Django 4.2+**: Modern Python web framework
- **PyTorch Integration**: Seamless model loading and inference
- **TensorFlow Integration**: Disease detection model support
- **Database Storage**: SQLite database for prediction history
- **Admin Panel**: Full Django admin interface
- **Static Files**: Optimized CSS and JavaScript

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (vet_model_env)
- All trained model files

### Quick Start
```bash
# 1. Activate virtual environment
source vet_model_env/Scripts/activate

# 2. Start the web application
python start_webapp.py

# 3. Open your browser
# Go to: http://127.0.0.1:8000
```

### Manual Setup
```bash
# 1. Activate virtual environment
source vet_model_env/Scripts/activate

# 2. Install dependencies
cd livestock_ai_webapp
pip install -r requirements.txt

# 3. Run migrations
python manage.py migrate

# 4. Collect static files
python manage.py collectstatic --noinput

# 5. Start server
python manage.py runserver
```

## 🌐 Web Application Structure

### 📁 Directory Structure
```
livestock_ai_webapp/
├── livestock_ai_webapp/          # Django project settings
│   ├── settings.py               # Configuration
│   ├── urls.py                   # URL routing
│   └── wsgi.py                   # WSGI configuration
├── predictions/                  # Main app
│   ├── models.py                 # Database models
│   ├── views.py                  # View logic
│   ├── ml_models.py              # ML model integration
│   ├── urls.py                   # App URLs
│   └── admin.py                  # Admin configuration
├── templates/                    # HTML templates
│   ├── base.html                 # Base template
│   └── predictions/              # App templates
│       ├── home.html             # Home page
│       ├── weight_prediction.html
│       ├── classification.html
│       ├── hoofed_animals.html
│       ├── disease_detection.html
│       └── history.html          # Prediction history
├── static/                       # Static files
│   └── css/
│       └── custom.css            # Custom styles
└── requirements.txt              # Dependencies
```

### 🎯 Available Pages

#### 🏠 Home Page (`/`)
- **Overview**: Welcome page with feature cards
- **Recent Predictions**: Display latest 5 predictions
- **Quick Access**: Direct links to all features
- **How It Works**: Step-by-step guide

#### ⚖️ Weight Estimation (`/weight/`)
- **Upload**: Drag-and-drop image upload
- **Prediction**: Real-time weight estimation
- **Results**: Detailed weight prediction with confidence
- **Tips**: Best practices for accurate results

#### 🏷️ Animal Classification (`/classification/`)
- **Upload**: Image upload for classification
- **Prediction**: Identify animal type (cattle, sheep, goats, camels)
- **Results**: Class probabilities and confidence scores
- **Info**: Supported animal types and characteristics

#### 🐾 Hoofed Animals (`/hoofed-animals/`)
- **Upload**: Image upload for multi-label analysis
- **Prediction**: Multi-label classification results
- **Results**: Detailed class predictions with confidence
- **Info**: About multi-label classification

#### 🏥 Disease Detection (`/disease/`)
- **Upload**: Image upload for disease analysis
- **Prediction**: Lumpy skin disease detection
- **Results**: Disease status with confidence scores
- **Alerts**: Important notices for disease detection

#### 📊 History (`/history/`)
- **View All**: Complete prediction history
- **Filter**: By prediction type and date
- **Delete**: Remove unwanted predictions
- **Export**: Download results (future feature)

## 🔧 Model Integration

### 🤖 Weight Estimation Model
```python
# PyTorch CNN model
- Architecture: 4 conv blocks + classifier
- Input: 224x224 RGB images
- Output: Weight in kg
- File: best_weight_model.pth
```

### 🏷️ Classification Model
```python
# PyTorch CNN model
- Architecture: Deep CNN with backbone
- Input: 224x224 RGB images
- Output: 4-class probabilities
- File: best_classification_model.pth
```

### 🐾 Hoofed Animals Model
```python
# PyTorch CNN model
- Architecture: Multi-label CNN
- Input: 224x224 RGB images
- Output: 6-class multi-label predictions
- File: best_hoofed_animals_model.pth
```

### 🏥 Disease Detection Model
```python
# TensorFlow/Keras model
- Architecture: EfficientNet-based
- Input: 224x224 RGB images
- Output: 2-class disease detection
- File: final_improved_model.keras
```

## 🎨 UI/UX Features

### 🎨 Modern Design
- **Color Scheme**: Professional green theme
- **Typography**: Clean, readable fonts
- **Icons**: Bootstrap Icons throughout
- **Animations**: Smooth hover effects and transitions

### 📱 Responsive Layout
- **Mobile First**: Optimized for all screen sizes
- **Grid System**: Bootstrap 5 responsive grid
- **Flexible Cards**: Adaptive content layout
- **Touch Friendly**: Large buttons and touch targets

### 🚀 Interactive Features
- **Drag & Drop**: File upload with visual feedback
- **Loading States**: Progress indicators during processing
- **Real-time Updates**: Instant result display
- **Error Handling**: User-friendly error messages

## 🔐 Admin Panel

### 👤 Access
- **URL**: http://127.0.0.1:8000/admin
- **Username**: admin
- **Password**: admin123

### 📊 Features
- **Prediction Management**: View all predictions
- **User Management**: Admin user accounts
- **Database**: Direct database access
- **Logs**: System logs and debugging

## 🚀 Deployment

### 🏠 Development
```bash
# Start development server
python manage.py runserver

# With custom port
python manage.py runserver 8080
```

### 🌐 Production (Future)
- **WSGI Server**: Gunicorn or uWSGI
- **Web Server**: Nginx or Apache
- **Database**: PostgreSQL or MySQL
- **Static Files**: CDN or cloud storage
- **SSL**: HTTPS configuration

## 🐛 Troubleshooting

### ❌ Common Issues

#### Model Loading Errors
```bash
# Check model files exist
ls -la *.pth *.keras

# Verify file permissions
chmod 644 *.pth *.keras
```

#### Django Errors
```bash
# Check Django installation
python -c "import django; print(django.get_version())"

# Run migrations
python manage.py migrate

# Collect static files
python manage.py collectstatic --noinput
```

#### Virtual Environment Issues
```bash
# Activate virtual environment
source vet_model_env/Scripts/activate

# Check Python path
which python
```

### 🔧 Debug Mode
```python
# In settings.py
DEBUG = True
ALLOWED_HOSTS = ['127.0.0.1', 'localhost']
```

## 📈 Performance

### ⚡ Optimization
- **Model Caching**: Models loaded once at startup
- **Image Processing**: Optimized preprocessing pipeline
- **Database**: Efficient queries and indexing
- **Static Files**: Minified CSS and JavaScript

### 📊 Monitoring
- **Prediction Tracking**: All predictions logged
- **Performance Metrics**: Response time monitoring
- **Error Logging**: Comprehensive error tracking
- **User Analytics**: Usage statistics (future)

## 🔮 Future Enhancements

### 🚀 Planned Features
- **Batch Processing**: Multiple image upload
- **API Endpoints**: REST API for external access
- **Mobile App**: React Native mobile application
- **Advanced Analytics**: Detailed prediction analytics
- **Export Features**: CSV/PDF result export
- **User Authentication**: Multi-user support
- **Cloud Deployment**: AWS/Azure deployment

### 🤖 Model Improvements
- **Model Updates**: Easy model replacement
- **A/B Testing**: Multiple model comparison
- **Confidence Calibration**: Improved confidence scores
- **Ensemble Methods**: Multiple model voting

## 📞 Support

### 🆘 Getting Help
- **Documentation**: This README file
- **Code Comments**: Inline code documentation
- **Error Messages**: Detailed error descriptions
- **Logs**: Django debug logs

### 🐛 Reporting Issues
1. Check the troubleshooting section
2. Review Django logs
3. Verify model files exist
4. Test with sample images

## 🎉 Success!

Your Livestock AI Web Application is now ready! 🚀

**Access your application at**: http://127.0.0.1:8000

**Admin panel**: http://127.0.0.1:8000/admin

**Features available**:
- ✅ Weight estimation
- ✅ Animal classification  
- ✅ Hoofed animals analysis
- ✅ Disease detection
- ✅ Prediction history

Happy analyzing! 🐄🤖
