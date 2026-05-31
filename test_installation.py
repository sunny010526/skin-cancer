import sys
import os

def check_installation():
    print("Checking installation...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'torch', 'torchvision', 'flask', 'PIL', 'numpy',
        'matplotlib', 'plotly'
    ]
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✓ {package} installed")
            else:
                __import__(package)
                print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
    
    # Check directory structure
    required_dirs = [
        'app',
        'app/models',
        'app/utils', 
        'app/training',
        'app/static/uploads',
        'app/models/saved_models',
        'templates'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
    
    print("\nInstallation check completed!")

if __name__ == '__main__':
    check_installation()