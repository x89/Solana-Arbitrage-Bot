#!/usr/bin/env python3
"""
Installation Checker for AI Momentum Real-time Predicting System
Checks if all required dependencies are installed
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False

def main():
    """Check all required packages"""
    print("=" * 70)
    print("AI Momentum Real-time Predicting System - Installation Check")
    print("=" * 70)
    print()
    
    # Core dependencies
    packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("joblib", "Joblib"),
        ("yfinance", "yfinance"),
        ("matplotlib", "Matplotlib"),
    ]
    
    # Optional dependencies
    optional_packages = [
        ("tensorflow", "TensorFlow"),
        ("onnxruntime", "ONNX Runtime"),
        ("talib", "TA-Lib (Optional - system works without it)"),
    ]
    
    print("Required Dependencies:")
    print("-" * 70)
    
    missing = []
    installed = []
    
    for import_name, display_name in packages:
        if check_package(import_name):
            print(f"✓ {display_name}")
            installed.append(display_name)
        else:
            print(f"✗ {display_name} - MISSING")
            missing.append(display_name)
    
    print()
    print("Optional Dependencies:")
    print("-" * 70)
    
    for import_name, display_name in optional_packages:
        if check_package(import_name):
            print(f"✓ {display_name}")
        else:
            print(f"○ {display_name} - Not installed")
    
    print()
    print("=" * 70)
    
    if missing:
        print(f"❌ {len(missing)} required packages are missing:")
        for pkg in missing:
            print(f"   - {pkg}")
        print()
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
        print()
        print("Note: TA-Lib may require special installation:")
        print("  - Windows: Download .whl from https://ta-lib.org/install/")
        print("  - Linux: Install libta-lib-dev then pip install TA-Lib")
        return 1
    else:
        print(f"✓ All required dependencies installed ({len(installed)} packages)")
        print()
        print("You can now run:")
        print("  python test_modules.py    # Test module imports")
        print("  python run_main.py        # Run full system demo")
        return 0

if __name__ == "__main__":
    sys.exit(main())

