import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Best Image Enhancer", layout="wide")
st.title("🌟 Best Image Enhancer (Beginner IPV Project)")

st.markdown("""
<div style='background-color:#f0f8ff; padding:20px; border-radius:10px'>
<h3 style='color:#2e86de'>Upload an image and choose what operation you'd like to apply!</h3>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    st.image(img_np, caption="🖼️ Original Image", use_container_width=True)

    # Convert to Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Select Operation
    st.markdown("---")
    operation = st.selectbox("🔍 Choose an Operation", [
        "Grayscale",
        "Gaussian Blur",
        "Median Blur",
        "Sobel Edge Detection",
        "Canny Edge Detection",
        "Dilation",
        "Erosion",
        "Opening",
        "Closing"
    ])

    kernel_size = st.slider("🔧 Kernel Size (odd only)", min_value=3, max_value=11, step=2, value=5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if operation == "Grayscale":
        st.image(gray, caption="🖤 Grayscale Image", use_container_width=True, channels="GRAY")

    elif operation == "Gaussian Blur":
        blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        st.image(blur, caption="🌫️ Gaussian Blur", use_container_width=True, channels="GRAY")

    elif operation == "Median Blur":
        median = cv2.medianBlur(gray, kernel_size)
        st.image(median, caption="🎯 Median Blur", use_container_width=True, channels="GRAY")

    elif operation == "Sobel Edge Detection":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)
        st.image(np.uint8(sobel_combined), caption="⚡ Sobel Edge Detection", use_container_width=True, channels="GRAY")

    elif operation == "Canny Edge Detection":
        canny = cv2.Canny(gray, 100, 200)
        st.image(canny, caption="🧠 Canny Edge Detection", use_container_width=True, channels="GRAY")

    elif operation == "Dilation":
        dilation = cv2.dilate(gray, kernel, iterations=1)
        st.image(dilation, caption="🚀 Dilation", use_container_width=True, channels="GRAY")

    elif operation == "Erosion":
        erosion = cv2.erode(gray, kernel, iterations=1)
        st.image(erosion, caption="⛏️ Erosion", use_container_width=True, channels="GRAY")

    elif operation == "Opening":
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        st.image(opening, caption="🔓 Opening", use_container_width=True, channels="GRAY")

    elif operation == "Closing":
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        st.image(closing, caption="🔒 Closing", use_container_width=True, channels="GRAY")

    st.success(f"✅ Operation Applied: {operation}")
