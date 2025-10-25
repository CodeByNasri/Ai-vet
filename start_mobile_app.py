#!/usr/bin/env python3
"""
Start the Livestock AI Mobile App
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the mobile app"""
    print("🐄 LIVESTOCK AI - MOBILE APP")
    print("=" * 50)
    print("Modern mobile-style web application")
    print("")
    
    # Check if we're in the right directory
    if not Path('livestock_ai_webapp').exists():
        print("❌ Error: livestock_ai_webapp directory not found!")
        print("Please run this script from the project root directory.")
        return False
    
    # Change to web app directory
    os.chdir('livestock_ai_webapp')
    
    print(f"🌐 Starting mobile app server...")
    print(f"📱 Mobile App: http://127.0.0.1:8000/mobile/")
    print(f"🔧 Admin panel: http://127.0.0.1:8000/admin")
    print(f"")
    print(f"🎨 App Features:")
    print(f"✅ Mobile-first design with bottom navigation")
    print(f"✅ Comprehensive AI analysis (weight, type, disease)")
    print(f"✅ Chat-style interface")
    print(f"✅ Custom color scheme (green/beige/yellow)")
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

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
