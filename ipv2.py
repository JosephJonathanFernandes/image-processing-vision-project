import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Best Image Enhancer", layout="wide")
st.title("🌟 Best Image Enhancer (Beginner IPV Project)")
st.markdown("""
<div style='background-color:#f0f8ff; padding:20px; border-radius:10px'>
<h3 style='color:#2e86de'>Upload an image and apply powerful image processing tools!</h3>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    with col1:
        st.image(img_np, caption="🖼️ Original Image", use_container_width=True)

    # Convert to Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    with col2:
        st.image(gray, caption="🖤 Grayscale Image", use_container_width=True, channels="GRAY")

    st.markdown("---")
    st.subheader("✨ Apply Filters and Morphological Operations")

    kernel_size = st.slider("🔧 Kernel Size (odd only)", min_value=3, max_value=11, step=2, value=5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    col3, col4 = st.columns(2)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # Median Blur
    median = cv2.medianBlur(gray, kernel_size)

    with col3:
        st.image(blur, caption="🌫️ Gaussian Blur", use_container_width=True, channels="GRAY")
        st.image(median, caption="🎯 Median Blur", use_container_width=True, channels="GRAY")

    # Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    # Canny
    canny = cv2.Canny(gray, 100, 200)

    with col4:
        st.image(np.uint8(sobel_combined), caption="⚡ Sobel Edge Detection", use_container_width=True, channels="GRAY")
        st.image(canny, caption="🧠 Canny Edge Detection", use_container_width=True, channels="GRAY")

    st.markdown("---")
    st.subheader("🧱 Morphological Operations")

    col5, col6 = st.columns(2)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    erosion = cv2.erode(gray, kernel, iterations=1)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    with col5:
        st.image(dilation, caption="🚀 Dilation", use_container_width=True, channels="GRAY")
        st.image(erosion, caption="⛏️ Erosion", use_container_width=True, channels="GRAY")

    with col6:
        st.image(opening, caption="🔓 Opening", use_container_width=True, channels="GRAY")
        st.image(closing, caption="🔒 Closing", use_container_width=True, channels="GRAY")