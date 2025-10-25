#!/usr/bin/env python3
"""
Test AI Integration for Livestock AI Web Application
Checks if all models are working properly
"""

import os
import sys
import torch
import tensorflow as tf
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 CHECKING DEPENDENCIES")
    print("=" * 50)
    
    try:
        print(f"✅ Python version: {sys.version}")
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        return True
    except Exception as e:
        print(f"❌ Dependency error: {e}")
        return False

def check_model_files():
    """Check if all model files exist"""
    print("\n📁 CHECKING MODEL FILES")
    print("=" * 50)
    
    model_files = {
        'best_weight_model.pth': 'Weight estimation model',
        'best_classification_model.pth': 'Classification model', 
        'best_hoofed_animals_model.pth': 'Hoofed animals model',
        'final_improved_model.keras': 'Disease detection model'
    }
    
    all_found = True
    for file, description in model_files.items():
        if Path(file).exists():
            size = Path(file).stat().st_size / (1024 * 1024)
            print(f"✅ {description}: {file} ({size:.1f} MB)")
        else:
            print(f"❌ {description}: {file} (NOT FOUND)")
            all_found = False
    
    return all_found

def test_pytorch_models():
    """Test PyTorch model loading"""
    print("\n🧠 TESTING PYTORCH MODELS")
    print("=" * 50)
    
    try:
        # Test weight model
        if Path('best_weight_model.pth').exists():
            weight_model = torch.load('best_weight_model.pth', map_location='cpu')
            print("✅ Weight model loaded successfully")
        else:
            print("❌ Weight model file not found")
        
        # Test classification model
        if Path('best_classification_model.pth').exists():
            classification_model = torch.load('best_classification_model.pth', map_location='cpu')
            print("✅ Classification model loaded successfully")
        else:
            print("❌ Classification model file not found")
        
        # Test hoofed animals model
        if Path('best_hoofed_animals_model.pth').exists():
            hoofed_model = torch.load('best_hoofed_animals_model.pth', map_location='cpu')
            print("✅ Hoofed animals model loaded successfully")
        else:
            print("❌ Hoofed animals model file not found")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch model loading error: {e}")
        return False

def test_tensorflow_model():
    """Test TensorFlow model loading"""
    print("\n🤖 TESTING TENSORFLOW MODEL")
    print("=" * 50)
    
    try:
        if Path('final_improved_model.keras').exists():
            disease_model = tf.keras.models.load_model('final_improved_model.keras')
            print("✅ Disease detection model loaded successfully")
            print(f"✅ Model input shape: {disease_model.input_shape}")
            print(f"✅ Model output shape: {disease_model.output_shape}")
            return True
        else:
            print("❌ Disease model file not found")
            return False
    except Exception as e:
        print(f"❌ TensorFlow model loading error: {e}")
        return False

def test_django_integration():
    """Test Django integration"""
    print("\n🌐 TESTING DJANGO INTEGRATION")
    print("=" * 50)
    
    try:
        import django
        from django.conf import settings
        print(f"✅ Django version: {django.get_version()}")
        
        # Check if we can import our models
        from livestock_ai_webapp.predictions.ml_models import ModelManager
        print("✅ ModelManager import successful")
        
        # Try to initialize (this will test model loading)
        model_manager = ModelManager()
        print("✅ ModelManager initialized")
        
        # Check which models loaded
        loaded_models = list(model_manager.models.keys())
        print(f"✅ Loaded models: {loaded_models}")
        
        return True
    except Exception as e:
        print(f"❌ Django integration error: {e}")
        return False

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("\n🔄 TESTING PREDICTION PIPELINE")
    print("=" * 50)
    
    try:
        from livestock_ai_webapp.predictions.ml_models import ModelManager
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Create a dummy image for testing
        import numpy as np
        from PIL import Image
        import io
        
        # Create a test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convert to file-like object
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Test weight prediction
        if 'weight' in model_manager.models:
            result = model_manager.predict_weight(img_buffer)
            print(f"✅ Weight prediction test: {result.get('status', 'unknown')}")
        
        # Test classification prediction
        if 'classification' in model_manager.models:
            img_buffer.seek(0)  # Reset buffer
            result = model_manager.predict_classification(img_buffer)
            print(f"✅ Classification prediction test: {result.get('status', 'unknown')}")
        
        # Test disease prediction
        if 'disease' in model_manager.models:
            img_buffer.seek(0)  # Reset buffer
            result = model_manager.predict_disease(img_buffer)
            print(f"✅ Disease prediction test: {result.get('status', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Prediction pipeline error: {e}")
        return False

def main():
    """Main test function"""
    print("🐄 LIVESTOCK AI - INTEGRATION TEST")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Dependencies", check_dependencies),
        ("Model Files", check_model_files),
        ("PyTorch Models", test_pytorch_models),
        ("TensorFlow Model", test_tensorflow_model),
        ("Django Integration", test_django_integration),
        ("Prediction Pipeline", test_prediction_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! AI integration is working perfectly!")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
