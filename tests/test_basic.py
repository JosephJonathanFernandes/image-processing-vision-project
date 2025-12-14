# Basic test file
# Add your tests here

import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports():
    """Test that all required packages can be imported"""
    try:
        import streamlit
        import cv2
        import numpy
        from PIL import Image
        from skimage.morphology import thin, dilation, disk
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

if __name__ == "__main__":
    if test_imports():
        print("✓ All imports successful")
    else:
        print("✗ Import test failed")
