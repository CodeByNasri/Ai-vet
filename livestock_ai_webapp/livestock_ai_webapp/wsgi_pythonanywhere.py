"""
WSGI configuration for PythonAnywhere deployment.
"""

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
