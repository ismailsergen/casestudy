import cv2 as cv
import numpy as np

from common import calculate_kernel_size


def adaptive_gaussian(gray_img):
    """Apply an adaptive Gaussian blur to smooth the grayscale image."""
    ksize = calculate_kernel_size(gray_img.shape)
    sigma = 0.3 * (((ksize - 1) * 0.5) - 1) + 0.8
    return cv.GaussianBlur(gray_img, (ksize, ksize), sigma)


def preprocess(gray_img):
    """Normalize and threshold the image to produce a binary ROI candidate mask."""
    normalized = cv.normalize(gray_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    gray_blurred = adaptive_gaussian(normalized)
    _otsu_val, binary_otsu = cv.threshold(
        gray_blurred,
        0,
        255,
        cv.THRESH_BINARY_INV | cv.THRESH_OTSU,
    )
    return binary_otsu


def refine_roi_mask(binary_mask: np.ndarray):
    """Remove thin mask connections with a small morphological opening."""
    # Remove thin bridges that incorrectly connect the panel to nearby structure.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    return cv.morphologyEx(binary_mask, cv.MORPH_OPEN, kernel, iterations=1)


def find_largest_component_bbox(binary_mask: np.ndarray):
    """Return the bounding box of the largest connected foreground component."""
    n_labels, _labels, stats, _centroids = cv.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
        ltype=cv.CV_32S,
    )

    best_label = -1
    best_area = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = i

    if best_label < 0 or best_area <= 0:
        return None

    x = int(stats[best_label, cv.CC_STAT_LEFT])
    y = int(stats[best_label, cv.CC_STAT_TOP])
    w = int(stats[best_label, cv.CC_STAT_WIDTH])
    h = int(stats[best_label, cv.CC_STAT_HEIGHT])
    return (x, y, w, h), best_area


def detect_roi(img: np.ndarray):
    """Extract and normalize the largest connected ROI region from the input image."""
    binary_img = preprocess(img)
    binary_img = refine_roi_mask(binary_img)
    roi_data = find_largest_component_bbox(binary_img)
    (x, y, w, h), _ = roi_data
    out_roi = img[y:y + h, x:x + w]
    out_roi = cv.normalize(out_roi, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return out_roi, (x, y)
