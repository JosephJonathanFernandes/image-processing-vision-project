import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Ultimate Image Enhancer", layout="wide")
st.title("🎨 Ultimate Image Enhancer – Beginner Friendly Image Processing App")

st.markdown("""
<div style='background-color:#f0f8ff; padding:20px; border-radius:10px'>
<h3 style='color:#2e86de'>Upload an image and apply a wide range of filters and operations!</h3>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload your image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    st.image(img_np, caption="🖼️ Original Image", use_container_width=True)

    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    operation = st.selectbox("🛠️ Choose an Operation", [
        "Grayscale",
        "Gaussian Blur",
        "Median Blur",
        "Bilateral Filter",
        "Sobel Edge Detection",
        "Canny Edge Detection",
        "Histogram Equalization",
        "Thresholding",
        "Adaptive Thresholding",
        "Brightness & Contrast",
        "Sharpening",
        "Rotation",
        "Flip",
        "Dilation",
        "Erosion",
        "Opening",
        "Closing",
        "Gradient",
        "Top Hat",
        "Black Hat"
    ])

    kernel_size = st.slider("🧩 Kernel Size (odd only)", 3, 11, step=2, value=5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    result = None

    if operation == "Grayscale":
        result = gray

    elif operation == "Gaussian Blur":
        result = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)

    elif operation == "Median Blur":
        result = cv2.medianBlur(img_np, kernel_size)

    elif operation == "Bilateral Filter":
        result = cv2.bilateralFilter(img_np, 9, 75, 75)

    elif operation == "Sobel Edge Detection":
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        sobel = cv2.magnitude(sobel_x, sobel_y)
        result = np.uint8(sobel)

    elif operation == "Canny Edge Detection":
        result = cv2.Canny(gray, 100, 200)

    elif operation == "Histogram Equalization":
        result = cv2.equalizeHist(gray)

    elif operation == "Thresholding":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    elif operation == "Adaptive Thresholding":
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, kernel_size, 5)

    elif operation == "Brightness & Contrast":
        brightness = st.slider("🔆 Brightness", -100, 100, 0)
        contrast = st.slider("🎚️ Contrast", -100, 100, 0)
        result = cv2.convertScaleAbs(img_np, alpha=1 + contrast / 100.0, beta=brightness)

    elif operation == "Sharpening":
        sharp_kernel = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        result = cv2.filter2D(img_np, -1, sharp_kernel)

    elif operation == "Rotation":
        angle = st.slider("🔁 Rotate Angle", -180, 180, 0)
        (h, w) = img_np.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(img_np, M, (w, h))

    elif operation == "Flip":
        flip_code = st.radio("↔️ Flip Mode", ["Horizontal", "Vertical"])
        result = cv2.flip(img_np, 1 if flip_code == "Horizontal" else 0)

    elif operation == "Dilation":
        result = cv2.dilate(gray, kernel, iterations=1)

    elif operation == "Erosion":
        result = cv2.erode(gray, kernel, iterations=1)

    elif operation == "Opening":
        result = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    elif operation == "Closing":
        result = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    elif operation == "Gradient":
        result = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    elif operation == "Top Hat":
        result = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    elif operation == "Black Hat":
        result = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    if result is not None:
        st.markdown("---")
        st.image(result, caption=f"✨ Result: {operation}", use_container_width=True,
                 channels="GRAY" if len(result.shape) == 2 else "BGR")
        st.success("✅ Operation Applied Successfully!")
