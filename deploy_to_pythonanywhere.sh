#!/bin/bash

# PythonAnywhere Deployment Script for AI Vet Project
# Run this script in your PythonAnywhere console

echo "🚀 Starting PythonAnywhere Deployment for AI Vet Project..."

# Step 1: Clone the repository
echo "📥 Cloning repository from GitHub..."
git clone https://github.com/CodeByNasri/Ai-vet.git
cd Ai-vet

# Step 2: Create virtual environment
echo "🐍 Creating virtual environment..."
python3.10 -m venv ai_vet_env
source ai_vet_env/bin/activate

# Step 3: Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Step 4: Install dependencies
echo "📦 Installing dependencies..."
pip install -r pythonanywhere_requirements.txt

# Step 5: Navigate to Django project
echo "📁 Navigating to Django project..."
cd livestock_ai_webapp

# Step 6: Configure settings
echo "⚙️ Configuring Django settings..."
cp livestock_ai_webapp/settings_production.py livestock_ai_webapp/settings.py

# Step 7: Run migrations
echo "🗄️ Running database migrations..."
python manage.py migrate

# Step 8: Collect static files
echo "📄 Collecting static files..."
python manage.py collectstatic --noinput

# Step 9: Create superuser (optional)
echo "👤 Creating superuser (optional)..."
echo "You can skip this step by pressing Ctrl+C"
python manage.py createsuperuser

echo "✅ Deployment setup complete!"
echo "📋 Next steps:"
echo "1. Go to the 'Web' tab in PythonAnywhere"
echo "2. Create a new web app with manual configuration"
echo "3. Configure the WSGI file as shown in the guide"
echo "4. Set up static file mappings"
echo "5. Reload your web app"
echo ""
echo "🌐 Your app will be available at: https://yourusername.pythonanywhere.com"
echo "📱 Mobile interface: https://yourusername.pythonanywhere.com/mobile/"
