#!/usr/bin/env python3
"""
Setup script for MediSense AI
Simple script to install dependencies and run the application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_app():
    """Run the Streamlit application"""
    print("Starting MediSense AI...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using MediSense AI!")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    print("🏥 MediSense AI Setup")
    print("=" * 30)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install dependencies
    if not install_requirements():
        return
    
    print("\n🚀 Setup complete! Starting the application...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application\n")
    
    # Run the app
    run_app()

if __name__ == "__main__":
    main()
