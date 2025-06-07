import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import thin, dilation, disk

st.set_page_config(page_title="🖼️ Image Enhancer", layout="wide")
st.title("🖼️ Image Enhancer - OpenCV & Streamlit")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((3, 3), np.uint8)

    st.image(img_np, caption="📷 Original Image", use_container_width=True)

    operation = st.selectbox("🛠️ Choose an Operation", [
        "Grayscale",
        "RGB to HSV",
        "HSV to RGB",
        "RGB to LAB",
        "LAB to RGB",
        "RGB to YCrCb",
        "Color Mask (HSV)",
        "Convex Hull",
        "Hole Filling",
        "Connected Components",
        "Skeletonization",
        "Thinning",
        "Thickening",
        "Gaussian Blur",
        "Median Blur",
        "Bilateral Filter",
        "Sobel Edge Detection",
        "Manual Sobel Edge Detection",
        "Prewitt Edge Detection",
        "Laplacian Edge Detection",
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

    result = None

    if operation == "Grayscale":
        result = gray

    elif operation == "RGB to HSV":
        result = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    elif operation == "HSV to RGB":
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    elif operation == "RGB to LAB":
        result = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    elif operation == "LAB to RGB":
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif operation == "RGB to YCrCb":
        result = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)

    elif operation == "Color Mask (HSV)":
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        lower_h = st.slider("Lower Hue", 0, 179, 30)
        upper_h = st.slider("Upper Hue", 0, 179, 90)
        lower = np.array([lower_h, 50, 50])
        upper = np.array([upper_h, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img_np, img_np, mask=mask)

    elif operation == "Convex Hull":
        gray_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
        contours, _ = cv2.findContours(gray_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_img = np.zeros_like(gray_bin)
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(hull_img, [hull], 0, 255, -1)
        result = hull_img

    elif operation == "Hole Filling":
        inv = cv2.bitwise_not(gray)
        h, w = gray.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        flood = inv.copy()
        cv2.floodFill(flood, mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        result = inv | flood_inv

    elif operation == "Connected Components":
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary)
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        result = cv2.merge([label_hue, blank_ch, blank_ch])
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        result[label_hue == 0] = 0

    elif operation == "Skeletonization":
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_not(binary)
        skel = np.zeros(binary.shape, np.uint8)
        while True:
            eroded = cv2.erode(binary, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(binary, temp)
            skel = cv2.bitwise_or(skel, temp)
            binary = eroded.copy()
            if cv2.countNonZero(binary) == 0:
                break
        result = skel

    elif operation == "Thinning":
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary = binary // 255
        thinned = thin(binary).astype(np.uint8) * 255
        result = thinned

    elif operation == "Thickening":
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        binary = binary // 255
        thick = dilation(binary, disk(1)).astype(np.uint8) * 255
        result = thick

    elif operation == "Gaussian Blur":
        result = cv2.GaussianBlur(img_np, (5, 5), 0)

    elif operation == "Median Blur":
        result = cv2.medianBlur(img_np, 5)

    elif operation == "Bilateral Filter":
        result = cv2.bilateralFilter(img_np, 9, 75, 75)

    elif operation == "Sobel Edge Detection":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.magnitude(sobelx, sobely)

    elif operation == "Manual Sobel Edge Detection":
        Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        Gx = cv2.filter2D(gray, -1, Kx)
        Gy = cv2.filter2D(gray, -1, Ky)
        result = np.sqrt(Gx**2 + Gy**2).astype(np.uint8)

    elif operation == "Prewitt Edge Detection":
        Kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        Ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        Gx = cv2.filter2D(gray, -1, Kx)
        Gy = cv2.filter2D(gray, -1, Ky)
        result = np.sqrt(Gx**2 + Gy**2).astype(np.uint8)

    elif operation == "Laplacian Edge Detection":
    # Apply Laplacian edge detection
      laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Convert back to uint8 for proper visualization
      result = cv2.convertScaleAbs(laplacian)


    elif operation == "Canny Edge Detection":
        result = cv2.Canny(gray, 100, 200)

    elif operation == "Histogram Equalization":
        result = cv2.equalizeHist(gray)

    elif operation == "Thresholding":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    elif operation == "Adaptive Thresholding":
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

    elif operation == "Brightness & Contrast":
        alpha = st.slider("Contrast", 0.5, 3.0, 1.0)
        beta = st.slider("Brightness", -100, 100, 0)
        result = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)

    elif operation == "Sharpening":
        kernel_sharp = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        result = cv2.filter2D(img_np, -1, kernel_sharp)

    elif operation == "Rotation":
        angle = st.slider("Rotation Angle", -180, 180, 0)
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        result = cv2.warpAffine(img_np, M, (w, h))

    elif operation == "Flip":
        flip_mode = st.selectbox("Flip Mode", ["Horizontal", "Vertical", "Both"])
        if flip_mode == "Horizontal":
            result = cv2.flip(img_np, 1)
        elif flip_mode == "Vertical":
            result = cv2.flip(img_np, 0)
        else:
            result = cv2.flip(img_np, -1)

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
        st.image(result, caption=f"🔧 {operation} Result", use_container_width=True)
