import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ Basic Image Enhancer (IPV Project for Beginners)")

# Upload the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    st.image(img_np, caption="Original Image")

    # Convert to Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale Image", channels="GRAY")

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    st.image(blur, caption="Gaussian Blur", channels="GRAY")

    # Apply Median Blur
    median = cv2.medianBlur(gray, 5)
    st.image(median, caption="Median Blur", channels="GRAY")

    # Sobel Edge Detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    st.image(np.uint8(sobel_combined), caption="Sobel Edge Detection", channels="GRAY")

    # Canny Edge Detection
    canny = cv2.Canny(gray, 100, 200)
    st.image(canny, caption="Canny Edge Detection", channels="GRAY")

    # Morphological Operations
    kernel = np.ones((5, 5), np.uint8)
    
    dilation = cv2.dilate(gray, kernel, iterations=1)
    erosion = cv2.erode(gray, kernel, iterations=1)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    st.image(dilation, caption="Dilation", channels="GRAY")
    st.image(erosion, caption="Erosion", channels="GRAY")
    st.image(opening, caption="Opening", channels="GRAY")
    st.image(closing, caption="Closing", channels="GRAY")
