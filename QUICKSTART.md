# Quick Start Guide

## Running the Application

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Application**
   - Open your browser
   - Navigate to `http://localhost:8501`

## Features Overview

### 💫 Basic Operations
- Brightness & Contrast adjustment
- Blur filters (Gaussian, Median, Bilateral)
- Image rotation and flipping
- Resize and crop
- Sharpening
- Histogram equalization

### 🎨 Color Transformations
- Color space conversions (RGB, HSV, LAB, YCrCb)
- Color masking and isolation
- Channel mixing
- Thresholding (binary and adaptive)
- Color balance adjustment

### 🔍 Edge Detection
- Sobel operator
- Prewitt operator
- Laplacian
- Canny edge detection
- Edge overlay on original

### 📐 Morphological Operations
- Dilation and erosion
- Opening and closing
- Morphological gradient
- Top hat and black hat
- Skeletonization
- Connected components
- Contour detection

### ✨ Special Effects
- Sepia tone
- Pencil sketch
- Cartoon effect
- Emboss
- Vintage effect
- Vignette
- Watercolor
- Oil painting
- Pixelation
- Posterization

## Tips for Best Results

1. **Upload High-Quality Images**: Better input = better output
2. **Use Comparison Mode**: Enable to see before/after side-by-side
3. **Experiment with Parameters**: Adjust sliders to find optimal settings
4. **Download Multiple Formats**: Save as PNG for lossless or JPG for smaller file size

## Keyboard Shortcuts

- `Ctrl/Cmd + R` - Refresh the application
- `Ctrl/Cmd + K` - Clear cache

## Troubleshooting

**Issue**: Application won't start
**Solution**: Ensure all dependencies are installed with `pip install -r requirements.txt`

**Issue**: Image upload fails
**Solution**: Check file format (JPG, PNG, BMP, TIFF supported) and size (<200MB)

**Issue**: Processing is slow
**Solution**: Try with smaller images or reduce parameter values

## Need Help?

- Check the [Full Documentation](docs/README.md)
- Report issues on [GitHub Issues](https://github.com/yourusername/image-processing-vision-project/issues)
- Read the [Contributing Guidelines](CONTRIBUTING.md)

---

**Version 2.0.0** | Last updated: December 2025
