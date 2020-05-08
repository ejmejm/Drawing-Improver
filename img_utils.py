import cv2
import glob
import numpy as np
import os

def get_outline_v1(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if len(c) > 3]
    contours_img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    outline = cv2.drawContours(np.ones(img.shape) * 255, contours, -1, (0, 0, 0), 1)
    return outline

def get_outline_v2(img):
    return 255 - cv2.Canny(img, 200, 400)

def get_outline_v3(img):
    img = np.int32(img)
    
    thresh = 30
    outline = np.ones(img.shape[:2]) * 255
    diff1 = np.sum(np.abs(img[1:] - img[:-1]), axis=-1) - thresh
    diff1 = np.round(np.clip(diff1, 0, 1))
    diff2 = np.sum(np.abs(img[:, 1:] - img[:, :-1]), axis=-1) - thresh
    diff2 = np.round(np.clip(diff2, 0, 1))
    
    blank = np.zeros(img.shape[:2])
    blank[1:] += diff1
    blank[:, 1:] += diff2
    outline = np.clip(blank, 0, 1)
    
#     outline = diff1 | diff2
    outline = outline * 255
    outline = np.int16(np.stack([outline] * 3, axis=-1))
    
    kernel = np.ones((2, 2))
    
    outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, kernel, iterations=2)
    outline = cv2.erode(outline, kernel, iterations=1)
    outline = cv2.dilate(outline, kernel, iterations=1)
    outline = cv2.morphologyEx(outline, cv2.MORPH_OPEN, kernel, iterations=1)
    
    smooth_size = 2
    kernel = np.ones((smooth_size, smooth_size),np.float32) / smooth_size**2
    outline = cv2.filter2D(outline, -1, kernel)
            
    return 255 - outline

get_outline = get_outline_v3

def get_frac_white(img, max_val=255.):
    if len(img.shape) == 3:
        return (img[:, :, 0]/max_val).round().sum() / (img.shape[0] * img.shape[1])
    elif len(img.shape) == 2:
        return (img/max_val).round().sum() / (img.shape[0] * img.shape[1])
    raise ValueError('`img` must be an image with a shape of length 2 or 3!')

def has_content_threshold(outline, threshold=0.96, max_val=255.):
    if get_frac_white(outline, max_val) < threshold:
        return True
    return False