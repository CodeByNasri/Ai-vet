#!/usr/bin/env python3
"""
Setup script for Livestock AI Web Application
This script will help you set up the Django web application with all your models
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_models():
    """Check if required model files exist"""
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

def setup_django():
    """Setup Django application"""
    print("\n🚀 Setting up Django application...")
    
    # Change to web app directory
    os.chdir('livestock_ai_webapp')
    
    # Install requirements
    if not run_command('pip install -r requirements.txt', 'Installing Python dependencies'):
        return False
    
    # Run migrations
    if not run_command('python manage.py migrate', 'Running Django migrations'):
        return False
    
    # Collect static files
    if not run_command('python manage.py collectstatic --noinput', 'Collecting static files'):
        return False
    
    print("✅ Django setup completed successfully!")
    return True

def create_startup_script():
    """Create a startup script for easy running"""
    startup_script = """#!/bin/bash
# Livestock AI Web Application Startup Script

echo "🚀 Starting Livestock AI Web Application..."
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "../vet_model_env" ]; then
    echo "📦 Activating virtual environment..."
    source ../vet_model_env/Scripts/activate
fi

# Change to web app directory
cd livestock_ai_webapp

# Start Django development server
echo "🌐 Starting web server..."
echo "Open your browser and go to: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python manage.py runserver
"""
    
    with open('start_webapp.sh', 'w') as f:
        f.write(startup_script)
    
    # Make it executable on Unix systems
    if os.name != 'nt':
        os.chmod('start_webapp.sh', 0o755)
    
    print("✅ Created startup script: start_webapp.sh")

def main():
    """Main setup function"""
    print("🐄 LIVESTOCK AI WEB APPLICATION SETUP")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('livestock_ai_webapp').exists():
        print("❌ Error: livestock_ai_webapp directory not found!")
        print("Please run this script from the project root directory.")
        return False
    
    # Check model files
    models_ok = check_models()
    
    # Setup Django
    if not setup_django():
        print("❌ Django setup failed!")
        return False
    
    # Create startup script
    create_startup_script()
    
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📋 Next steps:")
    print("1. Run: python manage.py runserver")
    print("2. Or use: bash start_webapp.sh")
    print("3. Open: http://127.0.0.1:8000")
    print("")
    print("🔧 Features available:")
    if models_ok:
        print("✅ Weight estimation")
        print("✅ Animal classification") 
        print("✅ Disease detection")
        print("✅ Hoofed animals classification")
    else:
        print("⚠️  Some models missing - check model files")
    
    print("\n💡 Tips:")
    print("- Upload clear, well-lit images for best results")
    print("- Use side-view images for weight estimation")
    print("- Check the History tab to view past predictions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
