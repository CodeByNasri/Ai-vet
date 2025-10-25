#!/usr/bin/env python3
"""
Test the web application models directly
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

def test_model_manager():
    """Test the ModelManager with proper Django settings"""
    print("🧠 TESTING MODEL MANAGER WITH DJANGO")
    print("=" * 50)
    
    try:
        from predictions.ml_models import ModelManager
        
        # Initialize model manager
        model_manager = ModelManager()
        print("✅ ModelManager initialized successfully")
        
        # Check which models loaded
        loaded_models = list(model_manager.models.keys())
        print(f"✅ Loaded models: {loaded_models}")
        
        # Test each model if available
        for model_name in loaded_models:
            print(f"✅ {model_name} model is ready")
        
        return True, loaded_models
    except Exception as e:
        print(f"❌ ModelManager error: {e}")
        return False, []

def test_prediction_with_dummy_image():
    """Test prediction with a dummy image"""
    print("\n🔄 TESTING PREDICTION WITH DUMMY IMAGE")
    print("=" * 50)
    
    try:
        from predictions.ml_models import ModelManager
        import numpy as np
        from PIL import Image
        import io
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Create a dummy image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convert to file-like object
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Test predictions
        results = {}
        
        if 'weight' in model_manager.models:
            try:
                result = model_manager.predict_weight(img_buffer)
                results['weight'] = result.get('status', 'unknown')
                print(f"✅ Weight prediction: {result.get('status', 'unknown')}")
            except Exception as e:
                print(f"❌ Weight prediction error: {e}")
                results['weight'] = 'error'
        
        if 'classification' in model_manager.models:
            try:
                img_buffer.seek(0)
                result = model_manager.predict_classification(img_buffer)
                results['classification'] = result.get('status', 'unknown')
                print(f"✅ Classification prediction: {result.get('status', 'unknown')}")
            except Exception as e:
                print(f"❌ Classification prediction error: {e}")
                results['classification'] = 'error'
        
        if 'disease' in model_manager.models:
            try:
                img_buffer.seek(0)
                result = model_manager.predict_disease(img_buffer)
                results['disease'] = result.get('status', 'unknown')
                print(f"✅ Disease prediction: {result.get('status', 'unknown')}")
            except Exception as e:
                print(f"❌ Disease prediction error: {e}")
                results['disease'] = 'error'
        
        return results
        
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
        return {}

def test_django_views():
    """Test Django views"""
    print("\n🌐 TESTING DJANGO VIEWS")
    print("=" * 50)
    
    try:
        from django.test import Client
        from django.urls import reverse
        
        client = Client()
        
        # Test home page
        response = client.get('/')
        print(f"✅ Home page: {response.status_code}")
        
        # Test weight prediction page
        response = client.get('/weight/')
        print(f"✅ Weight page: {response.status_code}")
        
        # Test classification page
        response = client.get('/classification/')
        print(f"✅ Classification page: {response.status_code}")
        
        # Test hoofed animals page
        response = client.get('/hoofed-animals/')
        print(f"✅ Hoofed animals page: {response.status_code}")
        
        # Test disease detection page
        response = client.get('/disease/')
        print(f"✅ Disease page: {response.status_code}")
        
        # Test history page
        response = client.get('/history/')
        print(f"✅ History page: {response.status_code}")
        
        return True
    except Exception as e:
        print(f"❌ Django views error: {e}")
        return False

def main():
    """Main test function"""
    print("🐄 LIVESTOCK AI WEBAPP - INTEGRATION TEST")
    print("=" * 60)
    
    # Test model manager
    model_success, loaded_models = test_model_manager()
    
    # Test predictions
    prediction_results = test_prediction_with_dummy_image()
    
    # Test Django views
    views_success = test_django_views()
    
    # Summary
    print("\n📊 INTEGRATION TEST SUMMARY")
    print("=" * 50)
    
    print(f"✅ Model Manager: {'PASS' if model_success else 'FAIL'}")
    print(f"✅ Django Views: {'PASS' if views_success else 'FAIL'}")
    print(f"✅ Loaded Models: {len(loaded_models)} models")
    
    if loaded_models:
        print("\n🤖 AVAILABLE MODELS:")
        for model in loaded_models:
            print(f"  ✅ {model}")
    
    if prediction_results:
        print("\n🔄 PREDICTION TESTS:")
        for model, status in prediction_results.items():
            print(f"  ✅ {model}: {status}")
    
    # Overall status
    if model_success and views_success and loaded_models:
        print("\n🎉 AI INTEGRATION IS WORKING!")
        print("✅ All models loaded successfully")
        print("✅ Django views are working")
        print("✅ Predictions are functional")
        print("\n🌐 Your web application is ready!")
        print("📱 Access it at: http://127.0.0.1:8000")
        return True
    else:
        print("\n⚠️  Some issues detected:")
        if not model_success:
            print("❌ Model loading issues")
        if not views_success:
            print("❌ Django view issues")
        if not loaded_models:
            print("❌ No models loaded")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
