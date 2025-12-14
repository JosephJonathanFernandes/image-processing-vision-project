"""
Utility functions for image processing operations
"""
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import thin, dilation, disk
import io


def get_image_download_link(img_array, format="PNG"):
    """Generate download data for the processed image"""
    if len(img_array.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(img_array)
    else:  # Color
        pil_img = Image.fromarray(img_array.astype('uint8'))
    
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


def apply_sepia(img):
    """Apply sepia tone effect"""
    img_sepia = np.array(img, dtype=np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([
        [0.272, 0.534, 0.131],
        [0.349, 0.686, 0.168],
        [0.393, 0.769, 0.189]
    ]))
    img_sepia[img_sepia > 255] = 255
    return np.array(img_sepia, dtype=np.uint8)


def apply_pencil_sketch(img, ksize=5):
    """Apply pencil sketch effect"""
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv_gray = 255 - gray_img
    blur = cv2.GaussianBlur(inv_gray, (ksize, ksize), 0)
    result = cv2.divide(gray_img, 255 - blur, scale=256)
    return result


def apply_cartoon_effect(img, num_down=2, num_bilateral=7):
    """Apply cartoon effect"""
    img_color = img
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)
        
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)
        
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
        
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))
    
    img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.medianBlur(img_edges, 7)
    img_edges = cv2.adaptiveThreshold(img_edges, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
    result = cv2.bitwise_and(img_color, img_edges)
    return result


def apply_emboss(img):
    """Apply emboss effect"""
    kernel = np.array([[0, -1, -1],
                       [1, 0, -1],
                       [1, 1, 0]])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = cv2.filter2D(gray, -1, kernel) + 128
    return result


def apply_vignette(img, strength=0.5):
    """Apply vignette effect"""
    rows, cols = img.shape[:2]
    kernel_x = cv2.getGaussianKernel(cols, cols/2)
    kernel_y = cv2.getGaussianKernel(rows, rows/2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    
    mask = mask * (1 - strength) + strength * 255
    mask = np.clip(mask, 0, 255)
    
    result = img.copy()
    for i in range(3):
        result[:,:,i] = result[:,:,i] * mask / 255
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def apply_watercolor(img):
    """Apply watercolor effect"""
    img_filtered = cv2.bilateralFilter(img, 9, 75, 75)
    img_filtered = cv2.medianBlur(img_filtered, 5)
    
    hsv = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2HSV)
    hsv[:,:,1] = hsv[:,:,1] * 1.2
    hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return result


def apply_oil_painting(img, kernel_size=7, levels=10):
    """Apply oil painting effect"""
    result = cv2.medianBlur(img, kernel_size)
    
    for i in range(3):
        channel = result[:,:,i]
        indices = np.arange(0, 256)
        divider = np.linspace(0, 255, levels+1)[1]
        quantiz = np.int0(np.linspace(0, 255, levels))
        
        for j in range(levels):
            channel[np.logical_and(divider*j <= channel, channel < divider*(j+1))] = quantiz[j]
    
    return result


def apply_pixelate(img, pixel_size=10):
    """Apply pixelate effect"""
    height, width = img.shape[:2]
    temp = cv2.resize(img, (width // pixel_size, height // pixel_size), interpolation=cv2.INTER_LINEAR)
    result = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return result


def apply_posterize(img, levels=4):
    """Apply posterize effect"""
    indices = np.arange(0, 256)
    divider = np.linspace(0, 255, levels+1)[1]
    quantiz = np.int0(np.linspace(0, 255, levels))
    
    result = img.copy()
    for i in range(3):
        ch = result[:,:,i]
        for j in range(levels):
            ch[np.logical_and(divider*j <= ch, ch < divider*(j+1))] = quantiz[j]
    
    return result


def apply_hdr_effect(img):
    """Apply HDR effect"""
    # Convert to float
    img_float = img.astype(np.float32) / 255.0
    
    # Apply tone mapping
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    
    return hdr


def apply_cross_process(img):
    """Apply cross process effect"""
    # Create curves for each channel
    result = img.copy()
    
    # Red channel curve
    red_curve = np.array([min(255, i * 1.1) for i in range(256)]).astype(np.uint8)
    # Green channel curve  
    green_curve = np.array([min(255, i * 0.9) for i in range(256)]).astype(np.uint8)
    # Blue channel curve
    blue_curve = np.array([min(255, i * 1.15) for i in range(256)]).astype(np.uint8)
    
    result[:,:,0] = cv2.LUT(result[:,:,0], red_curve)
    result[:,:,1] = cv2.LUT(result[:,:,1], green_curve)
    result[:,:,2] = cv2.LUT(result[:,:,2], blue_curve)
    
    return result


def apply_unsharp_mask(img, amount=1.5, radius=5):
    """Apply unsharp mask for sharpening"""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_motion_blur(img, size=15, angle=45):
    """Apply motion blur effect"""
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    
    # Rotate the kernel
    M = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
    kernel_motion_blur = cv2.warpAffine(kernel_motion_blur, M, (size, size))
    
    result = cv2.filter2D(img, -1, kernel_motion_blur)
    return result


def apply_noise_reduction(img, h=10):
    """Apply noise reduction"""
    result = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return result


def apply_detail_enhancement(img, sigma_s=60, sigma_r=0.07):
    """Enhance image details"""
    result = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)
    return result
