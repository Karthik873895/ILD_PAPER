import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.filters import median
from skimage.morphology import disk

def compute_lbp(image, P=8, R=1):
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    lbp = local_binary_pattern(image_uint8, P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=P*(P-1)+3, range=(0, P*(P-1)+2))
    return hist

def compute_elbp(image):
    image_uint8 = (image * 255).astype(np.uint8)
    helbp = local_binary_pattern(image_uint8, P=8, R=2, method="uniform")
    velbp = local_binary_pattern(image_uint8, P=8, R=1, method="uniform")
    hist_helbp, _ = np.histogram(helbp, bins=8*(8-1)+3, range=(0, 8*(8-1)+2))
    hist_velbp, _ = np.histogram(velbp, bins=8*(8-1)+3, range=(0, 8*(8-1)+2))
    return np.concatenate([hist_helbp, hist_velbp])

def compute_mbp(image):
    image_uint8 = (image * 255).astype(np.uint8)
    median_filtered = median(image_uint8, disk(3))
    mbp = np.where(image_uint8 >= median_filtered, 1, 0)
    hist, _ = np.histogram(mbp, bins=2, range=(0, 1))
    return hist

def compute_lpq(image):
    image_uint8 = (image * 255).astype(np.uint8)
    return np.histogram(image_uint8, bins=256, range=(0, 256))[0]

def extract_features(image):
    lbp_hist = compute_lbp(image)
    elbp_hist = compute_elbp(image)
    mbp_hist = compute_mbp(image)
    lpq_hist = compute_lpq(image)
    return np.concatenate([lbp_hist, elbp_hist, mbp_hist, lpq_hist])