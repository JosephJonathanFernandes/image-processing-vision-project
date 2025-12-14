"""
Configuration file for Image Processing Vision Project
"""

# Application Settings
APP_TITLE = "🖼️ Image Enhancer Pro"
APP_VERSION = "2.0.0"
MAX_FILE_SIZE_MB = 200

# Supported file types
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff", "webp"]

# Theme Colors
PRIMARY_COLOR = "#4F8BF9"
SECONDARY_COLOR = "#1E88E5"
ACCENT_COLOR = "#0D47A1"
SUCCESS_COLOR = "#4CAF50"
WARNING_COLOR = "#FF9800"
ERROR_COLOR = "#F44336"

# UI Settings
DEFAULT_COLUMN_RATIO = [1, 1]
PREVIEW_THUMBNAIL_WIDTH = 250
DEFAULT_IMAGE_QUALITY = 95

# Processing Settings
DEFAULT_KERNEL_SIZE = 3
DEFAULT_BLUR_AMOUNT = 5
DEFAULT_THRESHOLD = 127

# Categories and Operations
OPERATION_CATEGORIES = {
    "Basic Operations": {
        "icon": "💫",
        "operations": [
            "Grayscale",
            "Brightness & Contrast",
            "Gaussian Blur",
            "Median Blur",
            "Bilateral Filter",
            "Rotation",
            "Flip",
            "Resize",
            "Crop",
            "Invert Image",
            "Sharpening",
            "Histogram Equalization"
        ]
    },
    "Color Transformations": {
        "icon": "🎨",
        "operations": [
            "RGB to HSV",
            "HSV to RGB",
            "RGB to LAB",
            "LAB to RGB",
            "RGB to YCrCb",
            "Color Mask (HSV)",
            "Color Balance",
            "Channel Mixing",
            "Thresholding",
            "Adaptive Thresholding",
            "Color Quantization"
        ]
    },
    "Edge Detection": {
        "icon": "🔍",
        "operations": [
            "Sobel Edge Detection",
            "Prewitt Edge Detection",
            "Laplacian Edge Detection",
            "Canny Edge Detection",
            "Overlay Edges on Original",
            "Roberts Cross"
        ]
    },
    "Morphological Operations": {
        "icon": "📐",
        "operations": [
            "Dilation",
            "Erosion",
            "Opening",
            "Closing",
            "Gradient",
            "Top Hat",
            "Black Hat",
            "Skeletonization",
            "Connected Components",
            "Contour Detection"
        ]
    },
    "Special Effects": {
        "icon": "✨",
        "operations": [
            "Sepia Tone",
            "Pencil Sketch",
            "Cartoon Effect",
            "Emboss Effect",
            "Vintage Effect",
            "Vignette Effect",
            "Watercolor Effect",
            "Oil Painting Effect",
            "Pixelate",
            "Posterize",
            "HDR Effect",
            "Cross Process"
        ]
    },
    "Advanced Filters": {
        "icon": "🎯",
        "operations": [
            "Unsharp Mask",
            "Motion Blur",
            "Radial Blur",
            "Lens Blur",
            "Noise Reduction",
            "Detail Enhancement"
        ]
    }
}
