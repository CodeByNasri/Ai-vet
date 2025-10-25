#!/usr/bin/env python3
"""
Livestock AI Web Application Startup Script
This script starts the Django web application with all models integrated
"""

import os
import sys
import subprocess
from pathlib import Path

def check_models():
    """Check if all required model files exist"""
    print("🔍 Checking for required model files...")
    
    model_files = {
        'best_weight_model.pth': 'Weight estimation model',
        'best_classification_model.pth': 'Classification model', 
        'best_hoofed_animals_model.pth': 'Hoofed animals model',
        'final_improved_model.keras': 'Disease detection model'
    }
    
    missing_models = []
    for file, description in model_files.items():
        if Path(file).exists():
            size = Path(file).stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {description}: {file} ({size:.1f} MB)")
        else:
            print(f"❌ {description}: {file} (NOT FOUND)")
            missing_models.append(file)
    
    if missing_models:
        print(f"\n⚠️  Missing model files: {', '.join(missing_models)}")
        print("The web app will still work, but some features may not be available.")
        return False
    
    return True

def start_server():
    """Start the Django development server"""
    print("\n🚀 Starting Livestock AI Web Application...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('livestock_ai_webapp').exists():
        print("❌ Error: livestock_ai_webapp directory not found!")
        print("Please run this script from the project root directory.")
        return False
    
    # Check models
    models_ok = check_models()
    
    # Change to web app directory
    os.chdir('livestock_ai_webapp')
    
    print(f"\n🌐 Starting web server...")
    print(f"📱 Open your browser and go to: http://127.0.0.1:8000")
    print(f"🔧 Admin panel: http://127.0.0.1:8000/admin")
    print(f"👤 Admin credentials: admin / admin123")
    print(f"")
    print(f"Press Ctrl+C to stop the server")
    print(f"=" * 50)
    
    try:
        # Start Django server
        subprocess.run([
            sys.executable, 'manage.py', 'runserver'
        ], check=True)
    except KeyboardInterrupt:
        print(f"\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🐄 LIVESTOCK AI WEB APPLICATION")
    print("=" * 50)
    print("Modern Django web app with integrated ML models")
    print("")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Start server
    success = start_server()
    
    if success:
        print("\n🎉 Web application started successfully!")
        print("\n📋 Available Features:")
        print("✅ Weight Estimation - Predict livestock weight from images")
        print("✅ Animal Classification - Identify cattle, sheep, goats, camels")
        print("✅ Hoofed Animals - Multi-label classification")
        print("✅ Disease Detection - Lumpy skin disease detection")
        print("✅ Prediction History - View all past predictions")
        print("\n💡 Tips:")
        print("- Upload clear, well-lit images for best results")
        print("- Use side-view images for weight estimation")
        print("- Check the History tab to view past predictions")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
