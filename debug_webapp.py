#!/usr/bin/env python3
"""
Debug the web application issues
"""

import os
import sys
import django
from pathlib import Path

# Add the webapp to Python path
sys.path.append('livestock_ai_webapp')

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'livestock_ai_webapp.settings')
django.setup()

def test_model_manager_initialization():
    """Test ModelManager initialization in detail"""
    print("🔍 DEBUGGING MODEL MANAGER INITIALIZATION")
    print("=" * 60)
    
    try:
        from predictions.ml_models import ModelManager
        print("✅ ModelManager import successful")
        
        # Check model paths
        from django.conf import settings
        print(f"✅ Django settings loaded")
        print(f"✅ Model paths configured: {settings.MODEL_PATHS}")
        
        # Check if model files exist
        for model_name, model_path in settings.MODEL_PATHS.items():
            if model_path.exists():
                size = model_path.stat().st_size / (1024 * 1024)
                print(f"✅ {model_name}: {model_path} ({size:.1f} MB)")
            else:
                print(f"❌ {model_name}: {model_path} (NOT FOUND)")
        
        # Initialize model manager
        print("\n🔄 Initializing ModelManager...")
        model_manager = ModelManager()
        print("✅ ModelManager initialized")
        
        # Check loaded models
        loaded_models = list(model_manager.models.keys())
        print(f"✅ Loaded models: {loaded_models}")
        
        return model_manager, loaded_models
        
    except Exception as e:
        print(f"❌ ModelManager initialization error: {e}")
        import traceback
        traceback.print_exc()
        return None, []

def test_prediction_with_real_image():
    """Test prediction with a real image file"""
    print("\n🖼️ TESTING PREDICTION WITH REAL IMAGE")
    print("=" * 60)
    
    try:
        from predictions.ml_models import ModelManager
        from django.core.files.uploadedfile import SimpleUploadedFile
        import numpy as np
        from PIL import Image
        import io
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Create a test image file
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
        
        # Create Django file object
        test_file = SimpleUploadedFile(
            "test_image.jpg",
            img_bytes,
            content_type="image/jpeg"
        )
        
        print("✅ Test image created")
        
        # Test weight prediction
        if 'weight' in model_manager.models:
            print("🔄 Testing weight prediction...")
            result = model_manager.predict_weight(test_file)
            print(f"✅ Weight prediction result: {result}")
            
            if result.get('status') == 'success':
                print(f"✅ Predicted weight: {result.get('predicted_weight')} kg")
            else:
                print(f"❌ Weight prediction failed: {result.get('error')}")
        
        # Test classification prediction
        if 'classification' in model_manager.models:
            print("🔄 Testing classification prediction...")
            test_file.seek(0)  # Reset file pointer
            result = model_manager.predict_classification(test_file)
            print(f"✅ Classification prediction result: {result}")
            
            if result.get('status') == 'success':
                print(f"✅ Predicted class: {result.get('predicted_class')}")
                print(f"✅ Confidence: {result.get('confidence')}%")
            else:
                print(f"❌ Classification prediction failed: {result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_django_form_handling():
    """Test Django form handling"""
    print("\n📝 TESTING DJANGO FORM HANDLING")
    print("=" * 60)
    
    try:
        from django.test import Client
        from django.core.files.uploadedfile import SimpleUploadedFile
        import numpy as np
        from PIL import Image
        import io
        
        # Create test client
        client = Client()
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
        
        # Test weight prediction form
        print("🔄 Testing weight prediction form...")
        with open('test_image.jpg', 'wb') as f:
            f.write(img_bytes)
        
        with open('test_image.jpg', 'rb') as f:
            response = client.post('/weight/', {
                'image': f
            })
        
        print(f"✅ Weight form response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Weight form processed successfully")
        else:
            print(f"❌ Weight form error: {response.status_code}")
            print(f"Response content: {response.content.decode()[:500]}")
        
        # Clean up
        os.remove('test_image.jpg')
        
        return True
        
    except Exception as e:
        print(f"❌ Form handling test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_django_logs():
    """Check for Django errors in logs"""
    print("\n📋 CHECKING DJANGO CONFIGURATION")
    print("=" * 60)
    
    try:
        from django.conf import settings
        
        print(f"✅ DEBUG mode: {settings.DEBUG}")
        print(f"✅ ALLOWED_HOSTS: {settings.ALLOWED_HOSTS}")
        print(f"✅ MEDIA_ROOT: {settings.MEDIA_ROOT}")
        print(f"✅ MEDIA_URL: {settings.MEDIA_URL}")
        print(f"✅ STATIC_ROOT: {settings.STATIC_ROOT}")
        print(f"✅ STATIC_URL: {settings.STATIC_URL}")
        
        # Check if media directory exists
        if settings.MEDIA_ROOT.exists():
            print(f"✅ Media directory exists: {settings.MEDIA_ROOT}")
        else:
            print(f"❌ Media directory missing: {settings.MEDIA_ROOT}")
            print("Creating media directory...")
            settings.MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
            print("✅ Media directory created")
        
        return True
        
    except Exception as e:
        print(f"❌ Django configuration error: {e}")
        return False

def main():
    """Main debug function"""
    print("🐛 LIVESTOCK AI WEBAPP - DEBUG MODE")
    print("=" * 60)
    
    # Test Django configuration
    config_ok = check_django_logs()
    
    # Test model manager
    model_manager, loaded_models = test_model_manager_initialization()
    
    # Test predictions
    if model_manager:
        prediction_ok = test_prediction_with_real_image()
    else:
        prediction_ok = False
    
    # Test form handling
    form_ok = test_django_form_handling()
    
    # Summary
    print("\n📊 DEBUG SUMMARY")
    print("=" * 60)
    print(f"✅ Django Config: {'PASS' if config_ok else 'FAIL'}")
    print(f"✅ Model Manager: {'PASS' if model_manager else 'FAIL'}")
    print(f"✅ Predictions: {'PASS' if prediction_ok else 'FAIL'}")
    print(f"✅ Form Handling: {'PASS' if form_ok else 'FAIL'}")
    
    if all([config_ok, model_manager, prediction_ok, form_ok]):
        print("\n🎉 ALL SYSTEMS WORKING!")
        print("The issue might be in the frontend JavaScript or form submission.")
    else:
        print("\n⚠️  ISSUES DETECTED:")
        if not config_ok:
            print("❌ Django configuration issues")
        if not model_manager:
            print("❌ Model loading issues")
        if not prediction_ok:
            print("❌ Prediction processing issues")
        if not form_ok:
            print("❌ Form handling issues")

if __name__ == "__main__":
    main()
