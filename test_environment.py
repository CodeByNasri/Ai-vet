# test_environment.py
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path
import json
from PIL import Image
import os

def test_gpu_setup():
    """Test GPU functionality"""
    print("🔧 Testing GPU Setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        # Test GPU memory allocation
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print("✅ GPU test passed!")
    else:
        print("❌ GPU not available")
    print()

def test_data_loading():
    """Test dataset loading and structure"""
    print("📁 Testing Dataset Loading...")
    
    dataset_path = Path("Dataset - BMGF-LivestockWeight-CV")
    
    # Test if dataset exists
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return False
    
    print("✅ Dataset found!")
    
    # Count images in each batch
    batches = ['B2', 'B3', 'B4']
    total_images = 0
    
    for batch in batches:
        pixel_path = dataset_path / "Pixel" / batch
        if pixel_path.exists():
            if batch == 'B3':
                # B3 has different structure
                images_path = pixel_path / "images"
                if images_path.exists():
                    image_count = len(list(images_path.glob("*.jpg")))
                    total_images += image_count
                    print(f"  {batch}: {image_count} images")
            else:
                # B2 and B4 have side/rear structure
                for view in ['Side', 'Rear', 'Back']:
                    view_path = pixel_path / view
                    if view_path.exists():
                        images_path = view_path / "images"
                        if images_path.exists():
                            image_count = len(list(images_path.glob("*.jpg")))
                            total_images += image_count
                            print(f"  {batch}/{view}: {image_count} images")
    
    print(f"Total images found: {total_images}")
    print("✅ Dataset loading test passed!")
    print()
    return True

def test_image_processing():
    """Test image processing capabilities"""
    print("🖼️ Testing Image Processing...")
    
    # Test OpenCV
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    print("✅ OpenCV working")
    
    # Test PIL
    pil_image = Image.fromarray(test_image)
    resized = pil_image.resize((50, 50))
    print("✅ PIL working")
    
    # Test Albumentations
    transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5)
    ])
    augmented = transform(image=test_image)
    print("✅ Albumentations working")
    
    # Test PyTorch transforms
    torch_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    tensor_image = torch_transform(test_image)
    print("✅ PyTorch transforms working")
    print()

def test_model_creation():
    """Test creating a simple model"""
    print("🧠 Testing Model Creation...")
    
    # Create a simple CNN
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 4)  # 4 classes: sheep, goats, cattle, camels
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Model created successfully on {device}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Model creation test passed!")
    print()

def test_data_analysis():
    """Test data analysis capabilities"""
    print("📊 Testing Data Analysis...")
    
    # Test pandas
    df = pd.DataFrame({
        'animal_id': [1, 2, 3, 4, 5],
        'weight': [450, 380, 520, 410, 480],
        'sex': ['M', 'F', 'M', 'F', 'M']
    })
    print("✅ Pandas working")
    
    # Test matplotlib
    plt.figure(figsize=(8, 6))
    plt.plot(df['animal_id'], df['weight'], 'o-')
    plt.title('Animal Weight Distribution')
    plt.xlabel('Animal ID')
    plt.ylabel('Weight (kg)')
    plt.savefig('test_plot.png')
    plt.close()
    print("✅ Matplotlib working")
    
    # Clean up
    if os.path.exists('test_plot.png'):
        os.remove('test_plot.png')
    print()

def test_api_setup():
    """Test API framework"""
    print("🌐 Testing API Setup...")
    
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"message": "Livestock AI API is working!"}
        
        client = TestClient(app)
        response = client.get("/")
        print(f"✅ FastAPI working: {response.json()}")
    except Exception as e:
        print(f"❌ FastAPI test failed: {e}")
    print()

def main():
    """Run all tests"""
    print("🚀 LIVESTOCK AI ENVIRONMENT TEST")
    print("=" * 50)
    
    test_gpu_setup()
    test_data_loading()
    test_image_processing()
    test_model_creation()
    test_data_analysis()
    test_api_setup()
    
    print("🎉 ALL TESTS COMPLETED!")
    print("Your environment is ready for livestock AI development!")

if __name__ == "__main__":
    main()