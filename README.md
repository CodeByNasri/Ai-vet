# 🐄 AI Vet - Livestock Health & Management System

A comprehensive Django web application that uses AI models to analyze livestock health, estimate weight, classify animals, and detect diseases.

## 🚀 Features

### 🤖 AI-Powered Analysis
- **Weight Estimation**: Predict livestock weight using computer vision
- **Animal Classification**: Identify cattle, sheep, goats, and camels
- **Disease Detection**: Detect Lumpy Skin Disease in cattle
- **Comprehensive Analysis**: Combined AI analysis in a single interface

### 📱 Mobile-First Design
- **Modern Chat Interface**: WhatsApp-style messaging with AI
- **Mobile App Experience**: Bottom navigation with 4 main sections
- **Image Upload**: Easy drag-and-drop image selection
- **Real-time Results**: Instant AI analysis and feedback

### 🎨 Beautiful UI/UX
- **Custom Color Scheme**: 
  - Primary Green: `#7FA465`
  - Background Beige: `#FEF0CB`
  - Accent Yellow: `#EEB41E`
- **Responsive Design**: Works on all devices
- **Modern Animations**: Smooth transitions and interactions

## 🛠️ Technology Stack

- **Backend**: Django 4.2.0
- **AI Models**: PyTorch, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **Database**: SQLite (development)
- **Image Processing**: OpenCV, PIL, Albumentations

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/CodeByNasri/Ai-vet.git
cd Ai-vet
```

### 2. Create Virtual Environment
```bash
python -m venv vet_model_env
```

### 3. Activate Virtual Environment
**Windows:**
```bash
vet_model_env\Scripts\activate
```

**Linux/Mac:**
```bash
source vet_model_env/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Navigate to Django Project
```bash
cd livestock_ai_webapp
```

### 6. Run Database Migrations
```bash
python manage.py migrate
```

### 7. Start the Development Server
```bash
python manage.py runserver
```

## 🌐 Usage

### Web Interface
- **Home**: http://127.0.0.1:8000/
- **Mobile App**: http://127.0.0.1:8000/mobile/
- **Weight Prediction**: http://127.0.0.1:8000/weight/
- **Classification**: http://127.0.0.1:8000/classification/
- **Disease Detection**: http://127.0.0.1:8000/disease/

### Mobile App Features
1. **Dashboard**: Overview of the system
2. **Vet AI Chat**: Upload images and get comprehensive analysis
3. **Herd Management**: Manage your livestock records
4. **Recent Diseases**: View disease detection history

## 🤖 AI Models

### Weight Estimation Model
- **Type**: PyTorch CNN
- **Input**: Livestock images
- **Output**: Weight in kilograms
- **File**: `best_weight_model.pth`

### Classification Model
- **Type**: PyTorch CNN
- **Classes**: Cattle, Sheep, Goats, Camels
- **File**: `best_classification_model.pth`

### Hoofed Animals Model
- **Type**: PyTorch CNN
- **Classes**: 6 different hoofed animal types
- **File**: `best_hoofed_animals_model.pth`

### Disease Detection Model
- **Type**: TensorFlow/Keras
- **Purpose**: Lumpy Skin Disease detection
- **File**: `final_improved_model.keras`

## 📱 Mobile App Interface

The mobile app provides a modern chat-based interface:

- **Full-page chat experience** (no grid layout)
- **Plus button for image upload** (modern + icon)
- **Images appear in chat** (immediate preview)
- **Modern message bubbles** (WhatsApp-style)
- **AI and user avatars** (distinct identities)
- **Smooth animations** (polished interactions)
- **Loading states** (visual feedback)
- **Auto-scroll** (always see latest messages)

## 🔧 Development

### Project Structure
```
Ai-vet/
├── livestock_ai_webapp/          # Django project
│   ├── livestock_ai_webapp/     # Main project settings
│   ├── predictions/             # AI prediction app
│   ├── templates/               # HTML templates
│   ├── static/                  # CSS, JS, images
│   └── media/                   # Uploaded files
├── *.pth                       # PyTorch model files
├── *.keras                     # Keras model files
└── requirements.txt            # Python dependencies
```

### Key Files
- `livestock_ai_webapp/predictions/ml_models.py` - AI model integration
- `livestock_ai_webapp/predictions/views.py` - Django views
- `livestock_ai_webapp/templates/mobile_app.html` - Mobile interface
- `livestock_ai_webapp/static/css/custom.css` - Custom styling

## 🚀 Quick Start

1. **Clone and setup** (see Installation above)
2. **Start the server**: `python manage.py runserver`
3. **Open mobile app**: http://127.0.0.1:8000/mobile/
4. **Click "Vet AI" tab** (chat icon at bottom)
5. **Upload an image** using the green + button
6. **Get comprehensive analysis** in chat format

## 📊 Features Overview

| Feature | Description | Status |
|---------|-------------|-------|
| Weight Estimation | AI-powered weight prediction | ✅ Working |
| Animal Classification | Identify livestock types | ✅ Working |
| Disease Detection | Lumpy Skin Disease detection | ✅ Working |
| Mobile Interface | Modern chat-based UI | ✅ Working |
| Image Upload | Drag-and-drop functionality | ✅ Working |
| Real-time Analysis | Instant AI feedback | ✅ Working |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Contact: [Your Contact Information]

## 🎯 Future Enhancements

- [ ] User authentication system
- [ ] Database integration for herd management
- [ ] Advanced disease detection models
- [ ] Mobile app (React Native/Flutter)
- [ ] Cloud deployment (AWS/Heroku)
- [ ] API endpoints for mobile apps
- [ ] Real-time notifications
- [ ] Multi-language support

---

**Built with ❤️ for livestock health and management**