# 🚀 PythonAnywhere Deployment Guide for AI Vet Project

This guide will help you deploy your Django AI Vet project to PythonAnywhere.

## 📋 Prerequisites

1. **PythonAnywhere Account**: Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. **GitHub Repository**: Your project should be on GitHub (✅ Already done!)
3. **Basic Python Knowledge**: Understanding of Django and virtual environments

## 🎯 Step-by-Step Deployment

### Step 1: Create PythonAnywhere Account

1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Click "Create a Beginner account" (Free tier available)
3. Choose a username (this will be your domain: `yourusername.pythonanywhere.com`)
4. Verify your email address

### Step 2: Access the Console

1. Log into your PythonAnywhere account
2. Click on the **"Consoles"** tab
3. Click **"Start a new console"** → **"Bash"**

### Step 3: Clone Your Repository

```bash
# Clone your GitHub repository
git clone https://github.com/CodeByNasri/Ai-vet.git

# Navigate to the project directory
cd Ai-vet
```

### Step 4: Set Up Virtual Environment

```bash
# Create a virtual environment
python3.10 -m venv ai_vet_env

# Activate the virtual environment
source ai_vet_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 5: Install Dependencies

```bash
# Install production requirements
pip install -r pythonanywhere_requirements.txt

# Navigate to Django project
cd livestock_ai_webapp
```

### Step 6: Configure Django Settings

```bash
# Copy production settings
cp livestock_ai_webapp/settings_production.py livestock_ai_webapp/settings.py

# Edit the settings file to update your username
nano livestock_ai_webapp/settings.py
```

**Update these lines in settings.py:**
```python
# Change 'yourusername' to your actual PythonAnywhere username
ALLOWED_HOSTS = ['yourusername.pythonanywhere.com', '127.0.0.1', 'localhost']

# Update the path in wsgi_pythonanywhere.py as well
```

### Step 7: Run Database Migrations

```bash
# Run migrations
python manage.py migrate

# Create a superuser (optional)
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic --noinput
```

### Step 8: Configure Web App

1. Go to the **"Web"** tab in PythonAnywhere
2. Click **"Add a new web app"**
3. Choose **"Manual configuration"**
4. Select **Python 3.10**
5. Click **"Next"**

### Step 9: Configure WSGI File

1. In the Web tab, click on the **WSGI configuration file** link
2. Replace the entire content with:

```python
import os
import sys

# Add your project directory to the Python path
path = '/home/yourusername/Ai-vet/livestock_ai_webapp'
if path not in sys.path:
    sys.path.append(path)

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'livestock_ai_webapp.settings')

# Import Django's WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
```

**Important**: Replace `yourusername` with your actual PythonAnywhere username!

### Step 10: Configure Static Files

1. In the Web tab, scroll down to **"Static files"**
2. Add these mappings:

| URL | Directory |
|-----|-----------|
| `/static/` | `/home/yourusername/Ai-vet/livestock_ai_webapp/staticfiles/` |
| `/media/` | `/home/yourusername/Ai-vet/livestock_ai_webapp/media/` |

### Step 11: Reload Web App

1. Click the **"Reload"** button in the Web tab
2. Your app should now be live at: `https://yourusername.pythonanywhere.com`

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. **Import Errors**
```bash
# Make sure you're in the right directory
cd /home/yourusername/Ai-vet/livestock_ai_webapp

# Check if all dependencies are installed
pip list
```

#### 2. **Static Files Not Loading**
- Check that `STATIC_ROOT` is set correctly
- Run `python manage.py collectstatic --noinput`
- Verify static file mappings in Web tab

#### 3. **Model Files Not Found**
- Ensure all `.pth` and `.keras` files are in the project root
- Check file permissions: `ls -la *.pth *.keras`

#### 4. **Memory Issues**
- PythonAnywhere free accounts have memory limits
- Consider upgrading to a paid plan for better performance

### Debugging Steps

1. **Check the Error Log**:
   - Go to Web tab → "Error log" link
   - Look for specific error messages

2. **Test in Console**:
   ```bash
   # Activate virtual environment
   source ai_vet_env/bin/activate
   
   # Navigate to project
   cd /home/yourusername/Ai-vet/livestock_ai_webapp
   
   # Test Django
   python manage.py check
   ```

3. **Check File Permissions**:
   ```bash
   # Make sure files are readable
   chmod -R 755 /home/yourusername/Ai-vet/
   ```

## 📱 Testing Your Deployment

### 1. **Basic Functionality**
- Visit: `https://yourusername.pythonanywhere.com`
- Test the mobile interface: `https://yourusername.pythonanywhere.com/mobile/`

### 2. **AI Model Testing**
- Upload a test image
- Check if all models are working
- Verify image upload and processing

### 3. **Performance Testing**
- Test with different image sizes
- Check response times
- Monitor memory usage

## 🚀 Production Optimizations

### 1. **Security Settings**
```python
# In settings.py, add these for production:
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
```

### 2. **Static Files Optimization**
- Use WhiteNoise for serving static files
- Enable compression for better performance

### 3. **Database Optimization**
- Consider using PostgreSQL for production
- Set up database backups

## 📊 Monitoring and Maintenance

### 1. **Regular Updates**
```bash
# Update dependencies
pip install --upgrade -r pythonanywhere_requirements.txt

# Run migrations if needed
python manage.py migrate
```

### 2. **Backup Strategy**
- Regular database backups
- Code backups via Git
- Model file backups

### 3. **Performance Monitoring**
- Monitor memory usage
- Check response times
- Optimize model loading

## 🎯 Next Steps After Deployment

1. **Custom Domain**: Set up a custom domain (paid feature)
2. **SSL Certificate**: Enable HTTPS (automatic on PythonAnywhere)
3. **Database Upgrade**: Consider PostgreSQL for better performance
4. **CDN**: Use a CDN for static files
5. **Monitoring**: Set up error tracking and performance monitoring

## 📞 Support

If you encounter issues:

1. **Check PythonAnywhere Documentation**: [help.pythonanywhere.com](https://help.pythonanywhere.com)
2. **Django Documentation**: [docs.djangoproject.com](https://docs.djangoproject.com)
3. **Community Forums**: PythonAnywhere and Django communities

## 🎉 Success!

Once deployed, your AI Vet application will be accessible at:
**`https://yourusername.pythonanywhere.com`**

Share your live application with others and start using it for livestock health management!

---

**Happy Deploying! 🚀🐄🤖**
