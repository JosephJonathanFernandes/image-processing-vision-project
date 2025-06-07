import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import thin, dilation, disk

# Page configuration with custom theme
st.set_page_config(
    page_title="🖼️ Image Enhancer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        color: #1E88E5;
    }
    .category-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        color: #0D47A1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4F8BF9;
        color: white;
    }
    .info-box {
        background-color: #e8f0fe;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4F8BF9;
        margin-bottom: 10px; 
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
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stButton button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# App header with logo
st.markdown('<h1 class="main-header">🖼️ Image Enhancer Pro</h1>', unsafe_allow_html=True)

# Create sidebar for file upload and general settings
with st.sidebar:
    st.markdown("## 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.success("Image uploaded successfully!")
        
        # Display image information
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        st.markdown("### 📊 Image Information")
        st.markdown(f"**Dimensions:** {img_np.shape[1]} x {img_np.shape[0]} pixels")
        st.markdown(f"**Channels:** {img_np.shape[2] if len(img_np.shape) > 2 else 1}")
        
        # Preview thumbnail
        st.markdown("### 👁️ Preview")
        st.image(img_np, width=250)

# Main content area
if uploaded_file is not None:
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
            basic_op = st.selectbox("Choose a basic operation", [
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
            ])
            
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
                alpha = st.slider("Contrast", 0.5, 3.0, 1.0)
                beta = st.slider("Brightness", -100, 100, 0)
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
                
            elif basic_op == "Rotation":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                angle = st.slider("Rotation Angle", -180, 180, 0)
                st.markdown('</div>', unsafe_allow_html=True)
                h, w = img_np.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                result = cv2.warpAffine(img_np, M, (w, h))
                
            elif basic_op == "Flip":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                flip_mode = st.radio("Flip Direction", ["Horizontal", "Vertical", "Both"])
                st.markdown('</div>', unsafe_allow_html=True)
                if flip_mode == "Horizontal":
                    result = cv2.flip(img_np, 1)
                elif flip_mode == "Vertical":
                    result = cv2.flip(img_np, 0)
                else:
                    result = cv2.flip(img_np, -1)
                    
            elif basic_op == "Resize":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                maintain_aspect = st.checkbox("Maintain aspect ratio", True)
                if maintain_aspect:
                    scale = st.slider("Scale factor", 0.1, 2.0, 1.0, 0.1)
                    new_width = int(img_np.shape[1] * scale)
                    new_height = int(img_np.shape[0] * scale)
                else:
                    new_width = st.slider("Width", 50, img_np.shape[1]*2, img_np.shape[1])
                    new_height = st.slider("Height", 50, img_np.shape[0]*2, img_np.shape[0])
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.resize(img_np, (new_width, new_height))
                
            elif basic_op == "Grayscale":
                result = gray
                
            elif basic_op == "Gaussian Blur":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                kernel_size = st.slider("Blur Amount", 1, 25, 5, step=2)
                st.markdown('</div>', unsafe_allow_html=True)
                result = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
                
            elif basic_op == "Median Blur":
                st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
                kernel_size = st.slider("Blur Amount", 1, 25, 5, step=2)
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
                st.image(result, caption=f"Result: {basic_op}", use_container_width=True)
    
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
                st.image(result, caption=f"Result: {color_op}", use_container_width=True)
    
    # Tab 3: Edge Detection
    with tab3:
        st.markdown('<h2 class="sub-header">🔍 Edge Detection</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            edge_op = st.selectbox("Choose an edge detection method", [
                "Sobel Edge Detection",
                "Manual Sobel Edge Detection",
                "Prewitt Edge Detection",
                "Laplacian Edge Detection",
                "Canny Edge Detection",
                "Overlay Edges on Original"
            ])
            
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
                st.image(result, caption=f"Result: {edge_op}", use_container_width=True)
    
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
                st.image(result, caption=f"Result: {morph_op}", use_container_width=True)
    
    # Tab 5: Special Effects
    with tab5:
        st.markdown('<h2 class="sub-header">✨ Special Effects</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            effect_op = st.selectbox("Choose a special effect", [
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
            ])
            
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
                st.image(result, caption=f"Result: {effect_op}", use_container_width=True)

# Add a download button for the processed image
if 'result' in locals():
    st.markdown("---")
    st.markdown('<h3 class="sub-header">💾 Save Your Image</h3>', unsafe_allow_html=True)
    
    # Convert result to PIL Image for saving
    if len(result.shape) == 2:  # If grayscale
        pil_img = Image.fromarray(result)
    else:  # If color (RGB)
        pil_img = Image.fromarray(result)
    
    # Create a BytesIO object for the image
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    
    # Provide the download button
    st.download_button(
        label="Download Processed Image",
        data=buf.getvalue(),
        file_name="processed_image.png",
        mime="image/png"
    )

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7c7c7c; font-size: 0.8rem;">
    <p>Image Enhancer Pro | Built with Streamlit, OpenCV, and scikit-image</p>
    <p>© 2023 - For educational purposes only</p>
</div>
""", unsafe_allow_html=True)

