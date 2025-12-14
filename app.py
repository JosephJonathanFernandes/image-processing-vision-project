import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import thin, dilation, disk
import io
from datetime import datetime
import json

# Page configuration with custom theme
st.set_page_config(
    page_title="Image Enhancer Pro",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/image-processing-vision-project',
        'Report a bug': "https://github.com/yourusername/image-processing-vision-project/issues",
        'About': "# Image Enhancer Pro v2.0\n\nProfessional image processing and computer vision application."
    }
)

# Initialize session state for better UX
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False
if 'show_histogram' not in st.session_state:
    st.session_state.show_histogram = False
if 'operation_history' not in st.session_state:
    st.session_state.operation_history = []
if 'history_index' not in st.session_state:
    st.session_state.history_index = -1
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'current_operation' not in st.session_state:
    st.session_state.current_operation = None
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# Enhanced CSS styling with animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease;
    }
    
    .main-subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        animation: slideInLeft 0.6s ease;
    }
    .category-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        color: #0D47A1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: white;
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 0 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f0f0;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease;
    }
    .info-box p {
        margin: 0;
        color: #2c3e50;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-top: 1rem;
    }
    .tool-description {
        font-size: 0.9rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .parameter-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 16px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(17, 153, 142, 0.4);
    }
    
    .stats-card {
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 16px;
        border-left: 4px solid #667eea;
    }
    
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        margin: 16px 0;
        font-weight: 500;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        animation: slideInRight 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .preset-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #e9ecef;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }
    
    .preset-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2);
    }
    
    .history-item {
        background: #f8f9fa;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 8px;
        border-left: 3px solid #667eea;
        font-size: 0.9rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .shortcut-key {
        background: #2c3e50;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        margin-left: 8px;
    }
    
    .help-panel {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        border: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for HCD features
def add_to_history(operation_name, image_data, params=None):
    """Add operation to history for undo/redo"""
    # Remove any redo history when new operation is added
    if st.session_state.history_index < len(st.session_state.operation_history) - 1:
        st.session_state.operation_history = st.session_state.operation_history[:st.session_state.history_index + 1]
    
    st.session_state.operation_history.append({
        'operation': operation_name,
        'image': image_data,
        'params': params,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    st.session_state.history_index = len(st.session_state.operation_history) - 1
    
    # Limit history to 10 items
    if len(st.session_state.operation_history) > 10:
        st.session_state.operation_history.pop(0)
        st.session_state.history_index -= 1

def undo_operation():
    """Undo last operation"""
    if st.session_state.history_index > 0:
        st.session_state.history_index -= 1
        st.session_state.processed_image = st.session_state.operation_history[st.session_state.history_index]['image']
        return True
    return False

def redo_operation():
    """Redo previously undone operation"""
    if st.session_state.history_index < len(st.session_state.operation_history) - 1:
        st.session_state.history_index += 1
        st.session_state.processed_image = st.session_state.operation_history[st.session_state.history_index]['image']
        return True
    return False

def toggle_favorite(operation_name):
    """Add/remove operation from favorites"""
    if operation_name in st.session_state.favorites:
        st.session_state.favorites.remove(operation_name)
    else:
        st.session_state.favorites.append(operation_name)

# Preset filters based on common use cases
PRESETS = {
    "📸 Portrait Enhancement": {
        "operations": ["Brightness +20", "Contrast +15", "Slight Blur"],
        "description": "Soften skin and enhance facial features"
    },
    "🌆 Landscape Boost": {
        "operations": ["Saturation +30", "Contrast +20", "Sharpen"],
        "description": "Make landscapes pop with vivid colors"
    },
    "🎨 Vintage Film": {
        "operations": ["Sepia", "Reduce Brightness -10", "Add Grain"],
        "description": "Classic film photography look"
    },
    "✨ Social Media": {
        "operations": ["Brightness +15", "Saturation +20", "Slight Sharpen"],
        "description": "Perfect for Instagram and social posts"
    },
    "🖤 Dramatic B&W": {
        "operations": ["Grayscale", "High Contrast", "Sharpen"],
        "description": "Bold black and white conversion"
    }
}

# Enhanced app header
st.markdown('<h1 class="main-header">🖼️ Image Enhancer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Transform your images with professional-grade processing tools | Version 2.0</p>', unsafe_allow_html=True)

# Add keyboard shortcut handler
st.markdown("""
<script>
document.addEventListener('keydown', function(e) {
    // Ctrl+Z for Undo
    if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        document.querySelector('[data-testid="stButton"] button:contains("Undo")').click();
    }
    // Ctrl+Y for Redo
    if (e.ctrlKey && e.key === 'y') {
        e.preventDefault();
        document.querySelector('[data-testid="stButton"] button:contains("Redo")').click();
    }
    // ? for Help
    if (e.key === '?') {
        e.preventDefault();
        document.querySelector('[data-testid="stButton"] button:contains("Help")').click();
    }
});
</script>
""", unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.markdown("## 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Supported formats: JPG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        st.markdown('<div class="success-message">✓ Image uploaded successfully!</div>', unsafe_allow_html=True)
        
        # Store original image
        if st.session_state.original_image is None:
            image = Image.open(uploaded_file)
            st.session_state.original_image = np.array(image)
        
        # Display image information
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        
        st.markdown("### 📊 Image Information")
        
        # Stats cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <h4 style="margin:0;color:#2c3e50;font-size:0.9rem;">Width</h4>
                <p style="margin:8px 0 0 0;color:#667eea;font-size:1.5rem;font-weight:700;">{img_np.shape[1]}px</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <h4 style="margin:0;color:#2c3e50;font-size:0.9rem;">Height</h4>
                <p style="margin:8px 0 0 0;color:#667eea;font-size:1.5rem;font-weight:700;">{img_np.shape[0]}px</p>
            </div>
            """, unsafe_allow_html=True)
        
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        file_size = len(uploaded_file.getvalue()) / 1024
        
        st.markdown(f"""
        <div class="stats-card">
            <h4 style="margin:0;color:#2c3e50;font-size:0.9rem;">Channels & Size</h4>
            <p style="margin:8px 0 0 0;color:#667eea;font-size:1.2rem;font-weight:700;">{channels} channels | {file_size:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Preview
        st.markdown("### 👁️ Preview")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Presets Section
        st.markdown("---")
        st.markdown("### 🎯 Quick Presets")
        st.markdown("<p style='font-size: 0.85rem; color: #666; margin-bottom: 10px;'>Apply professional filters instantly</p>", unsafe_allow_html=True)
        
        for preset_name, preset_data in PRESETS.items():
            with st.expander(preset_name):
                st.markdown(f"**{preset_data['description']}**")
                st.markdown("**Steps:**")
                for op in preset_data['operations']:
                    st.markdown(f"• {op}")
                if st.button(f"Apply {preset_name}", key=f"preset_{preset_name}", use_container_width=True):
                    st.info(f"💡 {preset_name} preset selected! Apply operations in the tabs below.")
        
        # Operation History and Undo/Redo
        st.markdown("---")
        st.markdown("### 📜 Operation History")
        
        col_undo, col_redo = st.columns(2)
        with col_undo:
            if st.button("⬅️ Undo", use_container_width=True, disabled=st.session_state.history_index <= 0, help="Undo last operation (Ctrl+Z)"):
                if undo_operation():
                    st.success("✓ Undone")
                    st.rerun()
        
        with col_redo:
            if st.button("➡️ Redo", use_container_width=True, disabled=st.session_state.history_index >= len(st.session_state.operation_history) - 1, help="Redo operation (Ctrl+Y)"):
                if redo_operation():
                    st.success("✓ Redone")
                    st.rerun()
        
        # Display history
        if st.session_state.operation_history:
            st.markdown(f"<p style='font-size: 0.85rem; color: #666; margin: 10px 0;'>{len(st.session_state.operation_history)} operations in history</p>", unsafe_allow_html=True)
            
            with st.expander("View History", expanded=False):
                for idx, item in enumerate(reversed(st.session_state.operation_history[-5:])):
                    is_current = (len(st.session_state.operation_history) - 1 - idx) == st.session_state.history_index
                    marker = "➡️" if is_current else "•"
                    st.markdown(f"<div class='history-item'>{marker} <b>{item['operation']}</b><br/><small>{item['timestamp']}</small></div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 0.85rem; color: #999;'>No operations yet</p>", unsafe_allow_html=True)
        
        # Favorites section
        st.markdown("---")
        st.markdown("### ⭐ Favorite Operations")
        if st.session_state.favorites:
            for fav in st.session_state.favorites:
                st.markdown(f"• {fav}")
        else:
            st.markdown("<p style='font-size: 0.85rem; color: #999;'>No favorites yet. Star operations to add them here!</p>", unsafe_allow_html=True)
        
        # Help and Keyboard Shortcuts
        st.markdown("---")
        if st.button("❓ Help & Shortcuts", use_container_width=True):
            st.session_state.show_help = not st.session_state.show_help
        
        # Options
        st.markdown("---")
        st.markdown("### ⚙️ Options")
        st.session_state.comparison_mode = st.checkbox(
            "Show Comparison View",
            value=st.session_state.comparison_mode,
            help="Display original and processed images side-by-side"
        )
        
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.processed_image = None
            st.rerun()

# Main content area
if uploaded_file is not None:
    # Help Panel
    if st.session_state.show_help:
        st.markdown("""
        <div class="help-panel">
            <h3 style="margin-top: 0; color: #2c3e50;">⌨️ Keyboard Shortcuts & Tips</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                <div>
                    <p style="margin: 5px 0;"><span class="shortcut-key">Ctrl + Z</span> Undo last operation</p>
                    <p style="margin: 5px 0;"><span class="shortcut-key">Ctrl + Y</span> Redo operation</p>
                    <p style="margin: 5px 0;"><span class="shortcut-key">Ctrl + S</span> Save image</p>
                </div>
                <div>
                    <p style="margin: 5px 0;"><span class="shortcut-key">Ctrl + R</span> Reset all changes</p>
                    <p style="margin: 5px 0;"><span class="shortcut-key">Tab</span> Navigate between controls</p>
                    <p style="margin: 5px 0;"><span class="shortcut-key">?</span> Toggle this help</p>
                </div>
            </div>
            <hr style="margin: 20px 0; border: none; border-top: 1px solid #ccc;">
            <h4 style="color: #2c3e50; margin-bottom: 10px;">💡 Tips for Best Results</h4>
            <ul style="margin: 0; padding-left: 20px; color: #555;">
                <li>Use presets for quick professional edits</li>
                <li>Check comparison mode to see before/after</li>
                <li>Star your favorite operations for quick access</li>
                <li>Operations are saved in history - you can undo anytime</li>
                <li>Download in PNG for lossless quality or JPG for smaller files</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    
    # Create tabs for different categories of operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💫 Basic Operations", 
        "🎨 Color Transformations", 
        "🔍 Edge Detection", 
        "📐 Morphological Operations",
        "✨ Special Effects"
    ])
    
    # Tab 1: Basic Operations
    with tab1:
        st.markdown('<h2 class="sub-header">💫 Basic Image Operations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Add favorite button
            col_select, col_fav = st.columns([5, 1])
            with col_select:
                basic_op = st.selectbox(
                    "Choose a basic operation", 
                    [
                        "Grayscale",
                        "Brightness & Contrast",
                        "Gaussian Blur",
                        "Median Blur",
                        "Bilateral Filter",
                        "Rotation",
                        "Flip",
                        "Resize",
                        "Invert Image",
                        "Sharpening",
                        "Histogram Equalization"
                    ],
                    help="Select an operation to apply to your image"
                )
            with col_fav:
                is_favorite = basic_op in st.session_state.favorites
                if st.button("⭐" if is_favorite else "☆", key=f"fav_basic_{basic_op}", help="Add to favorites"):
                    toggle_favorite(basic_op)
                    st.rerun()
            
            # Show operation description
            operation_descriptions = {
                "Grayscale": "Converts the image to grayscale (black and white) by removing color information.",
                "Brightness & Contrast": "Adjusts the brightness and contrast levels of the image.",
                "Gaussian Blur": "Applies a Gaussian blur filter which reduces noise and detail.",
                "Median Blur": "Applies a median filter that reduces noise while preserving edges.",
                "Bilateral Filter": "Reduces noise while preserving edges by considering both spatial and intensity differences.",
                "Rotation": "Rotates the image by a specified angle in degrees.",
                "Flip": "Mirrors the image horizontally, vertically, or both.",
                "Resize": "Changes the dimensions of the image.",
                "Invert Image": "Creates a negative of the image by inverting all pixel values.",
                "Sharpening": "Enhances edges and fine details in the image.",
                "Histogram Equalization": "Improves contrast by stretching out the intensity range."
            }
            
            st.markdown(f'<div class="info-box"><p>{operation_descriptions.get(basic_op, "")}</p></div>', unsafe_allow_html=True)
            
            # Parameters for operations
            result = None
            
            if basic_op == "Brightness & Contrast":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                alpha = st.slider("Contrast", 0.5, 3.0, 1.0, help="1.0 = original, >1.0 = more contrast, <1.0 = less contrast")
                beta = st.slider("Brightness", -100, 100, 0, help="0 = original, positive = brighter, negative = darker")
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
                
            elif basic_op == "Rotation":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                angle = st.slider("Rotation Angle", -180, 180, 0, help="Positive = clockwise, Negative = counter-clockwise")
                st.markdown('</div>', unsafe_allow_html=True)
                h, w = img_np.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                result = cv2.warpAffine(img_np, M, (w, h))
                
            elif basic_op == "Flip":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                flip_mode = st.radio("Flip Direction", ["Horizontal", "Vertical", "Both"], help="Choose how to mirror the image")
                st.markdown('</div>', unsafe_allow_html=True)
                if flip_mode == "Horizontal":
                    result = cv2.flip(img_np, 1)
                elif flip_mode == "Vertical":
                    result = cv2.flip(img_np, 0)
                else:
                    result = cv2.flip(img_np, -1)
                    
            elif basic_op == "Resize":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                maintain_aspect = st.checkbox("Maintain aspect ratio", True, help="Keep original width/height proportions")
                if maintain_aspect:
                    scale = st.slider("Scale factor", 0.1, 2.0, 1.0, 0.1, help="1.0 = original size")
                    new_width = int(img_np.shape[1] * scale)
                    new_height = int(img_np.shape[0] * scale)
                else:
                    new_width = st.slider("Width", 50, img_np.shape[1]*2, img_np.shape[1], help="Target width in pixels")
                    new_height = st.slider("Height", 50, img_np.shape[0]*2, img_np.shape[0], help="Target height in pixels")
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.resize(img_np, (new_width, new_height))
                
            elif basic_op == "Grayscale":
                result = gray
                
            elif basic_op == "Gaussian Blur":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                kernel_size = st.slider("Blur Amount", 1, 25, 5, step=2, help="Higher = more blur. Must be odd number.")
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
                
            elif basic_op == "Median Blur":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                kernel_size = st.slider("Blur Amount", 1, 25, 5, step=2, help="Higher = more blur. Good for removing salt-and-pepper noise.")
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.medianBlur(img_np, kernel_size)
                
            elif basic_op == "Bilateral Filter":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                d = st.slider("Filter Size", 5, 15, 9)
                sigma_color = st.slider("Sigma Color", 10, 150, 75)
                sigma_space = st.slider("Sigma Space", 10, 150, 75)
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
                
            elif basic_op == "Invert Image":
                result = cv2.bitwise_not(img_np)
                
            elif basic_op == "Sharpening":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                amount = st.slider("Sharpening Amount", 1, 10, 5)
                st.markdown('</div>', unsafe_allow_html=True)
                kernel_sharp = np.array([[0,-1,0], [-1,1+(2*amount),-1], [0,-1,0]])
                result = cv2.filter2D(img_np, -1, kernel_sharp)
                
            elif basic_op == "Histogram Equalization":
                if len(img_np.shape) > 2 and img_np.shape[2] == 3:
                    # For color images
                    img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
                    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                    result = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
                else:
                    result = cv2.equalizeHist(gray)
        
        with col2:
            if result is not None:
                with st.spinner('🎨 Processing image...'):
                    st.image(result, caption=f"Result: {basic_op}", use_container_width=True)
                    
                    # Action buttons with progress feedback
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("✅ Apply This Result", key=f"apply_basic_{basic_op}", use_container_width=True, help="Save this result to continue editing"):
                            st.session_state.processed_image = result
                            add_to_history(f"Basic: {basic_op}", result.copy(), {"operation": basic_op})
                            st.success(f"✓ {basic_op} applied!")
                            st.balloons()
                    with btn_col2:
                        if st.button("🔄 Reset to Original", key=f"reset_basic_{basic_op}", use_container_width=True, help="Discard changes and start over"):
                            st.session_state.processed_image = None
                            st.session_state.operation_history = []
                            st.session_state.history_index = -1
                            st.info("Reset to original image")
                            st.rerun()
    
    # Tab 2: Color Transformations
    with tab2:
        st.markdown('<h2 class="sub-header">🎨 Color Transformations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            color_op = st.selectbox("Choose a color operation", [
                "RGB to HSV",
                "HSV to RGB",
                "RGB to LAB",
                "LAB to RGB",
                "RGB to YCrCb",
                "Color Mask (HSV)",
                "Color Balance",
                "Channel Mixing",
                "Thresholding",
                "Adaptive Thresholding"
            ])
            
            # Show operation description
            color_descriptions = {
                "RGB to HSV": "Converts from RGB (Red, Green, Blue) to HSV (Hue, Saturation, Value) color space.",
                "HSV to RGB": "Converts from HSV back to RGB color space.",
                "RGB to LAB": "Converts from RGB to LAB (Lightness, a, b) color space, which is perceptually uniform.",
                "LAB to RGB": "Converts from LAB back to RGB color space.",
                "RGB to YCrCb": "Converts from RGB to YCrCb (Luminance, Red-difference, Blue-difference).",
                "Color Mask (HSV)": "Creates a mask to isolate specific colors using HSV ranges.",
                "Color Balance": "Adjusts the balance of color channels in the image.",
                "Channel Mixing": "View and modify individual color channels.",
                "Thresholding": "Converts grayscale to binary using a single threshold value.",
                "Adaptive Thresholding": "Local thresholding that adapts to different image regions."
            }
            
            st.markdown(f'<div class="info-box"><p>{color_descriptions.get(color_op, "")}</p></div>', unsafe_allow_html=True)
            
            # Parameters for operations
            result = None
            
            if color_op == "RGB to HSV":
                result = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                
            elif color_op == "HSV to RGB":
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
            elif color_op == "RGB to LAB":
                result = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                
            elif color_op == "LAB to RGB":
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
            elif color_op == "RGB to YCrCb":
                result = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
                
            elif color_op == "Color Mask (HSV)":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                
                # Create two columns for the HSV sliders
                hue_col, satval_col = st.columns(2)
                
                with hue_col:
                    lower_h = st.slider("Lower Hue", 0, 179, 30)
                    upper_h = st.slider("Upper Hue", 0, 179, 90)
                
                with satval_col:
                    lower_s = st.slider("Lower Saturation", 0, 255, 50)
                    upper_s = st.slider("Upper Saturation", 0, 255, 255)
                    
                    lower_v = st.slider("Lower Value", 0, 255, 50)
                    upper_v = st.slider("Upper Value", 0, 255, 255)
                
                lower = np.array([lower_h, lower_s, lower_v])
                upper = np.array([upper_h, upper_s, upper_v])
                
                # Preview the mask
                show_mask = st.checkbox("Show mask only", False)
                st.markdown('</div>', unsafe_allow_html=True)
                
                mask = cv2.inRange(hsv, lower, upper)
                if show_mask:
                    result = mask
                else:
                    result = cv2.bitwise_and(img_np, img_np, mask=mask)
                
            elif color_op == "Color Balance":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                b, g, r = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                
                r_gain = st.slider("Red Channel", 0.0, 2.0, 1.0, 0.1)
                g_gain = st.slider("Green Channel", 0.0, 2.0, 1.0, 0.1)
                b_gain = st.slider("Blue Channel", 0.0, 2.0, 1.0, 0.1)
                
                r = cv2.convertScaleAbs(r, alpha=r_gain)
                g = cv2.convertScaleAbs(g, alpha=g_gain)
                b = cv2.convertScaleAbs(b, alpha=b_gain)
                
                st.markdown('</div>', unsafe_allow_html=True)
                balanced = cv2.merge([b, g, r])
                result = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
                
            elif color_op == "Channel Mixing":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                channel = st.radio("Select Channel", ["Red", "Green", "Blue", "All"])
                st.markdown('</div>', unsafe_allow_html=True)
                
                r, g, b = cv2.split(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                
                if channel == "Red":
                    result = cv2.cvtColor(cv2.merge([np.zeros_like(b), np.zeros_like(g), r]), cv2.COLOR_BGR2RGB)
                elif channel == "Green":
                    result = cv2.cvtColor(cv2.merge([np.zeros_like(b), g, np.zeros_like(r)]), cv2.COLOR_BGR2RGB)
                elif channel == "Blue":
                    result = cv2.cvtColor(cv2.merge([b, np.zeros_like(g), np.zeros_like(r)]), cv2.COLOR_BGR2RGB)
                else:
                    result = img_np
                    
            elif color_op == "Thresholding":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                thresh_val = st.slider("Threshold Value", 0, 255, 127)
                thresh_type = st.selectbox("Threshold Type", [
                    "Binary", "Binary Inverted", "Truncate", 
                    "To Zero", "To Zero Inverted"
                ])
                
                thresh_types = {
                    "Binary": cv2.THRESH_BINARY,
                    "Binary Inverted": cv2.THRESH_BINARY_INV,
                    "Truncate": cv2.THRESH_TRUNC,
                    "To Zero": cv2.THRESH_TOZERO,
                    "To Zero Inverted": cv2.THRESH_TOZERO_INV
                }
                st.markdown('</div>', unsafe_allow_html=True)
                
                _, result = cv2.threshold(gray, thresh_val, 255, thresh_types[thresh_type])
                
            elif color_op == "Adaptive Thresholding":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                method = st.radio("Method", ["Gaussian", "Mean"])
                block_size = st.slider("Block Size", 3, 99, 11, step=2)
                c_value = st.slider("C Value", -10, 30, 2)
                
                adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
                st.markdown('</div>', unsafe_allow_html=True)
                
                result = cv2.adaptiveThreshold(gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c_value)
        
        with col2:
            if result is not None:
                with st.spinner('🎨 Processing image...'):
                    st.image(result, caption=f"Result: {color_op}", use_container_width=True)
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("✅ Apply This Result", key=f"apply_color_{color_op}", use_container_width=True, help="Save this result to continue editing"):
                            st.session_state.processed_image = result
                            add_to_history(f"Color: {color_op}", result.copy(), {"operation": color_op})
                            st.success(f"✓ {color_op} applied!")
                            st.balloons()
                    with btn_col2:
                        if st.button("🔄 Reset to Original", key=f"reset_color_{color_op}", use_container_width=True, help="Discard changes and start over"):
                            st.session_state.processed_image = None
                            st.session_state.operation_history = []
                            st.session_state.history_index = -1
                            st.info("Reset to original image")
                            st.rerun()
    
    # Tab 3: Edge Detection
    with tab3:
        st.markdown('<h2 class="sub-header">🔍 Edge Detection</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            col_select, col_fav = st.columns([5, 1])
            with col_select:
                edge_op = st.selectbox(
                    "Choose an edge detection method", 
                    [
                        "Sobel Edge Detection",
                        "Manual Sobel Edge Detection",
                        "Prewitt Edge Detection",
                        "Laplacian Edge Detection",
                        "Canny Edge Detection",
                        "Overlay Edges on Original"
                    ],
                    help="Select an edge detection algorithm"
                )
            with col_fav:
                is_favorite = edge_op in st.session_state.favorites
                if st.button("⭐" if is_favorite else "☆", key=f"fav_edge_{edge_op}", help="Add to favorites"):
                    toggle_favorite(edge_op)
                    st.rerun()
            
            # Show operation description
            edge_descriptions = {
                "Sobel Edge Detection": "Computes the gradient using the Sobel operator in x and y directions.",
                "Manual Sobel Edge Detection": "Custom implementation of the Sobel operator showing how it works.",
                "Prewitt Edge Detection": "Uses the Prewitt operator to detect edges (similar to Sobel but with uniform coefficients).",
                "Laplacian Edge Detection": "Uses the Laplacian operator which computes the second derivative of the image.",
                "Canny Edge Detection": "Multi-stage algorithm that detects edges with noise suppression.",
                "Overlay Edges on Original": "Overlays detected edges on the original image."
            }
            
            st.markdown(f'<div class="info-box"><p>{edge_descriptions.get(edge_op, "")}</p></div>', unsafe_allow_html=True)
            
            # Parameters for operations
            result = None
            
            if edge_op == "Sobel Edge Detection":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                ksize = st.slider("Kernel Size", 1, 7, 3, step=2)
                scale = st.slider("Scale", 1, 10, 1)
                delta = st.slider("Delta", 0, 10, 0)
                st.markdown('</div>', unsafe_allow_html=True)
                
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize, scale=scale, delta=delta)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize, scale=scale, delta=delta)
                result = cv2.magnitude(sobelx, sobely)
                result = cv2.convertScaleAbs(result)
                
            elif edge_op == "Manual Sobel Edge Detection":
                Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                Gx = cv2.filter2D(gray, -1, Kx)
                Gy = cv2.filter2D(gray, -1, Ky)
                result = np.sqrt(Gx**2 + Gy**2).astype(np.uint8)
                
            elif edge_op == "Prewitt Edge Detection":
                Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                Ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
                Gx = cv2.filter2D(gray, -1, Kx)
                Gy = cv2.filter2D(gray, -1, Ky)
                result = np.sqrt(Gx**2 + Gy**2).astype(np.uint8)
                
            elif edge_op == "Laplacian Edge Detection":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                ksize = st.slider("Kernel Size", 1, 7, 3, step=2)
                st.markdown('</div>', unsafe_allow_html=True)
                
                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
                result = cv2.convertScaleAbs(laplacian)
                
            elif edge_op == "Canny Edge Detection":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                low_threshold = st.slider("Low Threshold", 0, 255, 100)
                high_threshold = st.slider("High Threshold", 0, 255, 200)
                st.markdown('</div>', unsafe_allow_html=True)
                
                result = cv2.Canny(gray, low_threshold, high_threshold)
                
            elif edge_op == "Overlay Edges on Original":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                low_threshold = st.slider("Low Threshold", 0, 255, 100)
                high_threshold = st.slider("High Threshold", 0, 255, 200)
                edge_color = st.color_picker("Edge Color", "#00FF00")
                opacity = st.slider("Edge Opacity", 0.0, 1.0, 0.2)
                
                # Convert hex color to RGB
                edge_color = edge_color.lstrip('#')
                edge_color_rgb = tuple(int(edge_color[i:i+2], 16) for i in (0, 2, 4))
                st.markdown('</div>', unsafe_allow_html=True)
                
                edges = cv2.Canny(gray, low_threshold, high_threshold)
                
                # Create color mask with selected color
                color_mask = np.zeros_like(img_np)
                for i in range(3):
                    color_mask[:, :, i] = edge_color_rgb[i]
                
                # Apply mask where edges are detected
                mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                edges_colored = np.where(mask > 0, color_mask, 0)
                
                # Blend with original
                result = cv2.addWeighted(img_np, 1.0, edges_colored, opacity, 0)
        
        with col2:
            if result is not None:
                with st.spinner('🔍 Processing image...'):
                    st.image(result, caption=f"Result: {edge_op}", use_container_width=True)
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("✅ Apply This Result", key=f"apply_edge_{edge_op}", use_container_width=True, help="Save this result to continue editing"):
                            st.session_state.processed_image = result
                            add_to_history(f"Edge: {edge_op}", result.copy(), {"operation": edge_op})
                            st.success(f"✓ {edge_op} applied!")
                            st.balloons()
                    with btn_col2:
                        if st.button("🔄 Reset to Original", key=f"reset_edge_{edge_op}", use_container_width=True, help="Discard changes and start over"):
                            st.session_state.processed_image = None
                            st.session_state.operation_history = []
                            st.session_state.history_index = -1
                            st.info("Reset to original image")
                            st.rerun()
    
    # Tab 4: Morphological Operations
    with tab4:
        st.markdown('<h2 class="sub-header">📐 Morphological Operations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            morph_op = st.selectbox("Choose a morphological operation", [
                "Dilation",
                "Erosion",
                "Opening",
                "Closing",
                "Gradient",
                "Top Hat",
                "Black Hat",
                "Skeletonization",
                "Thinning",
                "Thickening",
                "Convex Hull",
                "Hole Filling",
                "Connected Components",
                "Contour Detection"
            ])
            
            # Show operation description
            morph_descriptions = {
                "Dilation": "Expands the white regions in the image, useful for joining broken parts.",
                "Erosion": "Shrinks the white regions in the image, useful for removing small noise.",
                "Opening": "Erosion followed by dilation, good for removing noise while preserving shape.",
                "Closing": "Dilation followed by erosion, good for closing small holes in objects.",
                "Gradient": "Difference between dilation and erosion, highlighting object boundaries.",
                "Top Hat": "Difference between original and opening, highlighting bright details.",
                "Black Hat": "Difference between closing and original, highlighting dark details.",
                "Skeletonization": "Reduces shapes to single-pixel-wide lines representing their structure.",
                "Thinning": "Similar to skeletonization but preserves endpoints better.",
                "Thickening": "Expands thin lines to make them more visible.",
                "Convex Hull": "Creates the smallest convex polygon that contains all white pixels.",
                "Hole Filling": "Fills holes inside objects, creating solid shapes.",
                "Connected Components": "Labels different connected regions with unique colors.",
                "Contour Detection": "Finds and draws outlines of objects in the image."
            }
            
            st.markdown(f'<div class="info-box"><p>{morph_descriptions.get(morph_op, "")}</p></div>', unsafe_allow_html=True)
            
            # Common parameters
            st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
            
            # Only show threshold for operations that need binary images
            if morph_op in ["Skeletonization", "Thinning", "Thickening", "Convex Hull", 
                           "Hole Filling", "Connected Components", "Contour Detection"]:
                threshold = st.slider("Threshold", 0, 255, 127)
                binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Show kernel size for operations that use it
            if morph_op in ["Dilation", "Erosion", "Opening", "Closing", "Gradient", "Top Hat", "Black Hat"]:
                kernel_size = st.slider("Kernel Size", 1, 15, 3, step=2)
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                iterations = st.slider("Iterations", 1, 10, 1)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Parameters for operations
            result = None
            
            if morph_op == "Dilation":
                result = cv2.dilate(gray, kernel, iterations=iterations)
                
            elif morph_op == "Erosion":
                result = cv2.erode(gray, kernel, iterations=iterations)
                
            elif morph_op == "Opening":
                result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
                
            elif morph_op == "Closing":
                result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                
            
            elif morph_op == "Gradient":
                result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
                
            elif morph_op == "Top Hat":
                result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
                
            elif morph_op == "Black Hat":
                result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
                
            elif morph_op == "Skeletonization":
                result = binary.copy()
                for _ in range(10):  # Multiple iterations for better results
                    skeleton = thin(result > 0)
                    result = (skeleton * 255).astype(np.uint8)
                
            elif morph_op == "Thinning":
                result = binary.copy()
                skeleton = thin(result > 0)
                result = (skeleton * 255).astype(np.uint8)
                
            elif morph_op == "Thickening":
                result = binary.copy()
                thickened = dilation(result > 0, disk(1))
                result = (thickened * 255).astype(np.uint8)
                
            elif morph_op == "Convex Hull":
                # Find contours and draw convex hull
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result = np.zeros_like(binary)
                
                for contour in contours:
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(result, [hull], 0, 255, -1)
                
            elif morph_op == "Hole Filling":
                # Find contours and fill them
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result = np.zeros_like(binary)
                cv2.drawContours(result, contours, -1, 255, -1)
                
            elif morph_op == "Connected Components":
                # Label connected components
                num_labels, labels = cv2.connectedComponents(binary)
                
                # Create a color map for visualization
                label_hue = np.uint8(179 * labels / np.max(labels))
                blank_ch = 255 * np.ones_like(label_hue)
                labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
                
                # Convert to BGR
                labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)
                
                # Set background to black
                labeled_img[labels == 0] = 0
                result = labeled_img
                
            elif morph_op == "Contour Detection":
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                result = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
                cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        with col2:
            if result is not None:
                with st.spinner('📐 Processing image...'):
                    st.image(result, caption=f"Result: {morph_op}", use_container_width=True)
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("✅ Apply This Result", key=f"apply_morph_{morph_op}", use_container_width=True, help="Save this result to continue editing"):
                            st.session_state.processed_image = result
                            add_to_history(f"Morph: {morph_op}", result.copy(), {"operation": morph_op})
                            st.success(f"✓ {morph_op} applied!")
                            st.balloons()
                    with btn_col2:
                        if st.button("🔄 Reset to Original", key=f"reset_morph_{morph_op}", use_container_width=True, help="Discard changes and start over"):
                            st.session_state.processed_image = None
                            st.session_state.operation_history = []
                            st.session_state.history_index = -1
                            st.info("Reset to original image")
                            st.rerun()
    
    # Tab 5: Special Effects
    with tab5:
        st.markdown('<h2 class="sub-header">✨ Special Effects</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            col_select, col_fav = st.columns([5, 1])
            with col_select:
                effect_op = st.selectbox(
                    "Choose a special effect", 
                    [
                        "Sepia Tone",
                        "Pencil Sketch",
                        "Cartoon Effect",
                        "Emboss Effect",
                        "Vintage Effect",
                        "Vignette Effect",
                        "Watercolor Effect",
                        "Oil Painting Effect",
                        "Pixelate",
                        "Posterize"
                    ],
                    help="Select a special effect to apply"
                )
            with col_fav:
                is_favorite = effect_op in st.session_state.favorites
                if st.button("⭐" if is_favorite else "☆", key=f"fav_effect_{effect_op}", help="Add to favorites"):
                    toggle_favorite(effect_op)
                    st.rerun()
            
            # Show operation description
            effect_descriptions = {
                "Sepia Tone": "Applies a warm brownish tone typical of old photographs.",
                "Pencil Sketch": "Creates a pencil-like drawing effect.",
                "Cartoon Effect": "Simplifies the image to look like a cartoon drawing.",
                "Emboss Effect": "Creates a 3D embossed effect.",
                "Vintage Effect": "Creates a faded, nostalgic look typical of vintage photographs.",
                "Vignette Effect": "Darkens the corners of the image for a classic photography look.",
                "Watercolor Effect": "Simplifies and softens the image like a watercolor painting.",
                "Oil Painting Effect": "Creates a painterly effect similar to oil painting.",
                "Pixelate": "Reduces resolution to create a blocky, pixelated look.",
                "Posterize": "Reduces the number of colors to create a poster-like effect."
            }
            
            st.markdown(f'<div class="info-box"><p>{effect_descriptions.get(effect_op, "")}</p></div>', unsafe_allow_html=True)
            
            # Parameters for operations
            result = None
            
            if effect_op == "Sepia Tone":
                # Create sepia effect
                img_sepia = np.array(img_np, dtype=np.float64)
                img_sepia = cv2.transform(img_sepia, np.matrix([
                    [0.272, 0.534, 0.131],
                    [0.349, 0.686, 0.168],
                    [0.393, 0.769, 0.189]
                ]))
                
                # Normalize values
                img_sepia[img_sepia > 255] = 255
                result = np.array(img_sepia, dtype=np.uint8)
                
            elif effect_op == "Pencil Sketch":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                ksize = st.slider("Blur Kernel Size", 1, 21, 5, step=2)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Convert to grayscale
                gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                
                # Invert the grayscale image
                inv_gray = 255 - gray_img
                
                # Apply Gaussian blur
                blur = cv2.GaussianBlur(inv_gray, (ksize, ksize), 0)
                
                # Blend using color dodge
                result = cv2.divide(gray_img, 255 - blur, scale=256)
                
                # You can also provide color version as an option
                # sketch_color = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
                # result = cv2.bitwise_and(img_np, sketch_color)
                
            elif effect_op == "Cartoon Effect":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                num_down = st.slider("Downsampling", 1, 3, 2)
                num_bilateral = st.slider("Bilateral Filtering", 1, 10, 7)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply bilateral filter for cartoon effect
                img_color = img_np
                for _ in range(num_down):
                    img_color = cv2.pyrDown(img_color)
                    
                for _ in range(num_bilateral):
                    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
                    
                for _ in range(num_down):
                    img_color = cv2.pyrUp(img_color)
                    
                # Resize to match original dimensions
                img_color = cv2.resize(img_color, (img_np.shape[1], img_np.shape[0]))
                
                # Convert to grayscale and apply median blur
                img_edges = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.medianBlur(img_edges, 7)
                img_edges = cv2.adaptiveThreshold(img_edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
                
                # Convert back to color for masking
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
                
                # Combine edges and color
                result = cv2.bitwise_and(img_color, img_edges)
                
            elif effect_op == "Emboss Effect":
                kernel = np.array([[0, -1, -1],
                                   [1, 0, -1],
                                   [1, 1, 0]])
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                result = cv2.filter2D(gray, -1, kernel) + 128
                
            elif effect_op == "Vintage Effect":
                # Apply vintage effect
                result = img_np.copy()
                
                # Reduce intensity and add warm tint
                result = cv2.convertScaleAbs(result, alpha=0.8, beta=30)
                
                # Reduce saturation
                hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
                hsv[:,:,1] = hsv[:,:,1] * 0.6
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
                # Add vignette
                rows, cols = result.shape[:2]
                kernel_x = cv2.getGaussianKernel(cols, cols/2)
                kernel_y = cv2.getGaussianKernel(rows, rows/2)
                kernel = kernel_y * kernel_x.T
                mask = 255 * kernel / np.linalg.norm(kernel)
                
                for i in range(3):
                    result[:,:,i] = result[:,:,i] * mask
                
            elif effect_op == "Vignette Effect":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                vignette_strength = st.slider("Vignette Strength", 0.0, 1.0, 0.5)
                st.markdown('</div>', unsafe_allow_html=True)
                
                rows, cols = img_np.shape[:2]
                
                # Generate vignette mask using Gaussian kernels
                kernel_x = cv2.getGaussianKernel(cols, cols/2)
                kernel_y = cv2.getGaussianKernel(rows, rows/2)
                kernel = kernel_y * kernel_x.T
                mask = 255 * kernel / np.linalg.norm(kernel)
                
                # Adjust strength
                mask = mask * (1 - vignette_strength) + vignette_strength * 255
                mask = np.clip(mask, 0, 255)
                
                # Apply mask to each channel
                result = img_np.copy()
                for i in range(3):
                    result[:,:,i] = result[:,:,i] * mask / 255
                
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            elif effect_op == "Watercolor Effect":
                # Apply bilateral filter for smoothing while preserving edges
                img_filtered = cv2.bilateralFilter(img_np, 9, 75, 75)
                
                # Apply median blur for painterly effect
                img_filtered = cv2.medianBlur(img_filtered, 5)
                
                # Increase saturation
                hsv = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2HSV)
                hsv[:,:,1] = hsv[:,:,1] * 1.2
                hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
            elif effect_op == "Oil Painting Effect":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                kernel_size = st.slider("Kernel Size", 1, 15, 7, step=2)
                levels = st.slider("Levels", 1, 20, 10)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply median blur for oil painting effect
                result = cv2.medianBlur(img_np, kernel_size)
                
                # Quantize colors for more pronounced oil painting look
                for i in range(3):
                    channel = result[:,:,i]
                    indices = np.arange(0, 256)
                    divider = np.linspace(0, 255, levels+1)[1]
                    quantiz = np.int0(np.linspace(0, 255, levels))
                    
                    for j in range(levels):
                        channel[np.logical_and(divider*j <= channel, channel < divider*(j+1))] = quantiz[j]
 
            elif effect_op == "Pixelate":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                pixel_size = st.slider("Pixel Size", 1, 50, 10)
                st.markdown('</div>', unsafe_allow_html=True)
                
                height, width = img_np.shape[:2]
                temp = cv2.resize(img_np, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
                result = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
                
            elif effect_op == "Posterize":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                levels = st.slider("Color Levels", 2, 8, 4)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create a posterization lookup table
                indices = np.arange(0, 256)
                divider = np.linspace(0, 255, levels+1)[1]
                quantiz = np.int0(np.linspace(0, 255, levels))
                
                # Apply the lookup table
                result = img_np.copy()
                for i in range(3):
                    ch = result[:,:,i]
                    for j in range(levels):
                        ch[np.logical_and(divider*j <= ch, ch < divider*(j+1))] = quantiz[j]
        
        with col2:
            if result is not None:
                with st.spinner('✨ Processing image...'):
                    st.image(result, caption=f"Result: {effect_op}", use_container_width=True)
                    
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        if st.button("✅ Apply This Result", key=f"apply_effect_{effect_op}", use_container_width=True, help="Save this result to continue editing"):
                            st.session_state.processed_image = result
                            add_to_history(f"Effect: {effect_op}", result.copy(), {"operation": effect_op})
                            st.success(f"✓ {effect_op} applied!")
                            st.balloons()
                    with btn_col2:
                        if st.button("🔄 Reset to Original", key=f"reset_effect_{effect_op}", use_container_width=True, help="Discard changes and start over"):
                            st.session_state.processed_image = None
                            st.session_state.operation_history = []
                            st.session_state.history_index = -1
                            st.info("Reset to original image")
                            st.rerun()

# Add a download button for the processed image
if 'result' in locals() and result is not None:
    st.markdown("---")
    st.markdown('<h3 class="sub-header">💾 Download Your Result</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Convert result to PIL Image for saving
    if len(result.shape) == 2:  # If grayscale
        pil_img = Image.fromarray(result)
    else:  # If color (RGB)
        pil_img = Image.fromarray(result.astype('uint8'))
    
    # Create download buttons for different formats
    import io
    from datetime import datetime
    
    with col1:
        buf_png = io.BytesIO()
        pil_img.save(buf_png, format="PNG")
        st.download_button(
            label="📥 Download as PNG",
            data=buf_png.getvalue(),
            file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col2:
        buf_jpg = io.BytesIO()
        if len(result.shape) == 2:
            pil_img_jpg = Image.fromarray(result).convert('RGB')
        else:
            pil_img_jpg = pil_img
        pil_img_jpg.save(buf_jpg, format="JPEG", quality=95)
        st.download_button(
            label="📥 Download as JPG",
            data=buf_jpg.getvalue(),
            file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    
    with col3:
        if st.button("🗑️ Clear Result", use_container_width=True):
            st.session_state.processed_image = None
            st.rerun()
else:
    # Welcome screen when no image is uploaded
    if uploaded_file is None:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <h2 style="color: #667eea; font-size: 2.5rem; margin-bottom: 20px;">👋 Welcome to Image Enhancer Pro!</h2>
            <p style="font-size: 1.2rem; color: #666; margin-bottom: 30px;">
                Upload an image to get started with professional-grade image processing
            </p>
            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 40px; flex-wrap: wrap;">
                <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 250px;">
                    <div style="font-size: 3rem; margin-bottom: 15px;">💫</div>
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">Basic Operations</h4>
                    <p style="color: #666; font-size: 0.9rem;">Adjust brightness, contrast, blur, and more</p>
                </div>
                <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 250px;">
                    <div style="font-size: 3rem; margin-bottom: 15px;">🎨</div>
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">Color Transform</h4>
                    <p style="color: #666; font-size: 0.9rem;">Apply color masks, balance, and conversions</p>
                </div>
                <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); max-width: 250px;">
                    <div style="font-size: 3rem; margin-bottom: 15px;">✨</div>
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">Special Effects</h4>
                    <p style="color: #666; font-size: 0.9rem;">Sepia, sketch, cartoon, and artistic filters</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7c7c7c; padding: 2rem 0; margin-top: 3rem; border-top: 1px solid #e9ecef;">
    <p style="font-weight: 600; margin-bottom: 10px; font-size: 1rem;">🖼️ Image Enhancer Pro</p>
    <p style="margin-bottom: 10px;">Built with ❤️ using Streamlit, OpenCV, and scikit-image</p>
    <p style="font-size: 0.75rem; margin-top: 10px; color: #999;">Version 2.0.0 | © 2025 | For educational and professional use</p>
</div>
""", unsafe_allow_html=True)

