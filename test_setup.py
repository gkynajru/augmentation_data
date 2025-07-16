#!/usr/bin/env python3
"""
Test Setup Script
=================

This script verifies that your Vietnamese SLU augmentation setup is correct.

Usage:
    python test_setup.py
"""

import os
import sys
import importlib
import subprocess

def check_python_version():
    """Check Python version"""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 6:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.6+")
        return False

def check_dependencies():
    """Check required packages"""
    print("\n2. Checking dependencies...")
    required = ['pandas', 'numpy', 'yaml', 'tqdm']
    missing = []
    
    for package in required:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package} - installed")
        except ImportError:
            print(f"   ❌ {package} - missing")
            missing.append(package)
    
    if missing:
        print(f"\n   Install missing packages with: pip install {' '.join(missing)}")
        return False
    return True

def check_directory_structure():
    """Check directory structure"""
    print("\n3. Checking directory structure...")
    
    required_dirs = [
        'src',
        'scripts', 
        'dashboard',
        'data',
        'data/input',
        'data/output',
        'data/examples'
    ]
    
    required_files = [
        'config.yaml',
        'src/__init__.py',
        'src/augmenter.py',
        'scripts/1_generate_augmentations.py',
        'scripts/3_integrate_approved_data.py',
        'dashboard/index.html'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}/ - exists")
        else:
            print(f"   ❌ {dir_path}/ - missing")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} - exists")
        else:
            print(f"   ❌ {file_path} - missing")
            all_good = False
    
    return all_good

def test_import():
    """Test importing the augmenter"""
    print("\n4. Testing imports...")
    try:
        from src.augmenter import VietnameseSLUAugmenter
        print("   ✅ Can import VietnameseSLUAugmenter")
        return True
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_sample_generation():
    """Test with sample data"""
    print("\n5. Testing sample generation...")
    
    sample_file = "data/examples/sample_data.jsonl"
    if not os.path.exists(sample_file):
        print(f"   ❌ Sample file {sample_file} not found")
        print("   Creating sample file...")
        
        os.makedirs("data/examples", exist_ok=True)
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write('{"sentence": "bật đèn phòng khách", "intent": "bật thiết bị", "entities": [{"type": "device", "filler": "đèn"}, {"type": "location", "filler": "phòng khách"}]}\n')
            f.write('{"sentence": "tắt máy lạnh", "intent": "tắt thiết bị", "entities": [{"type": "device", "filler": "máy lạnh"}]}\n')
        print("   ✅ Sample file created")
    
    # Try running generation script
    try:
        result = subprocess.run([
            sys.executable, 
            "scripts/1_generate_augmentations.py",
            "--input", sample_file,
            "--test-mode"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Generation script runs successfully")
            return True
        else:
            print(f"   ❌ Generation script failed:")
            print(f"      {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Error running generation script: {e}")
        return False

def check_web_server():
    """Check if web server can be started"""
    print("\n6. Testing web server...")
    try:
        import http.server
        print("   ✅ http.server module available")
        print("   To start dashboard: python -m http.server 8000")
        print("   Then open: http://localhost:8000/dashboard/index.html")
        return True
    except:
        print("   ❌ http.server module not available")
        return False

def create_missing_files():
    """Offer to create missing files"""
    print("\n7. Creating missing files...")
    
    # Create src/__init__.py if missing
    if not os.path.exists('src/__init__.py'):
        os.makedirs('src', exist_ok=True)
        with open('src/__init__.py', 'w') as f:
            f.write('# This file makes src a Python package\n')
        print("   ✅ Created src/__init__.py")
    
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
        print("   ✅ Created logs/ directory")

def main():
    print("Vietnamese SLU Augmentation - Setup Test")
    print("=" * 50)
    
    results = []
    
    # Run all checks
    results.append(("Python version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Directory structure", check_directory_structure()))
    
    # Create missing files
    create_missing_files()
    
    # Continue checks
    results.append(("Import test", test_import()))
    results.append(("Sample generation", test_sample_generation()))
    results.append(("Web server", check_web_server()))
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<30} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Start the web server: python -m http.server 8000")
        print("2. Open dashboard: http://localhost:8000/dashboard/index.html")
        print("3. Generate augmentations: python scripts/1_generate_augmentations.py --input your_data.jsonl")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nFor the Export button issue in the dashboard:")
        print("1. Open browser console (F12)")
        print("2. Check for JavaScript errors when clicking Export")
        print("3. Try the manual export methods in the troubleshooting guide")

if __name__ == "__main__":
    main()