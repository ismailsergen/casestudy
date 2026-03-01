import math

import cv2 as cv
import numpy as np

MIN_AR = 0.4
MAX_AR = 2.0
MIN_FILL = 0.50
MIN_AREA_RATIO = 0.00075
MAX_AREA_RATIO = 0.01


def calculate_kernel_size(img_shape):
    """Compute an odd kernel size scaled from the image area."""
    height, width = img_shape[:2]
    dvec = math.sqrt(math.sqrt(width * height))
    ksize = max(1, int(dvec))
    return ksize if ksize % 2 == 1 else ksize + 1


def visualize(
    roi_img: np.ndarray,
    rect_list: list[tuple[int, int, int, int, float, float, float]],
    group_bbox: tuple[int, int, int, int] | None = None,
):
    """Render rectangle IDs, centers, and optional group box on the ROI image."""
    vis = cv.cvtColor(roi_img, cv.COLOR_GRAY2BGR)
    for idx, (x, y, w, h, _area, _ar, _fill) in enumerate(rect_list, start=1):
        cv.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cx = x + (w // 2)
        cy = y + (h // 2)
        cv.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        label = f"id={idx}"
        cv.putText(
            vis,
            label,
            (x, max(12, y - 4)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv.LINE_AA,
        )
    if group_bbox is not None:
        gx, gy, gw, gh = group_bbox
        pad = 20
        h, w = roi_img.shape[:2]
        x1 = max(0, gx - pad)
        y1 = max(0, gy - pad)
        x2 = min(w - 1, gx + gw + pad)
        y2 = min(h - 1, gy + gh + pad)
        cv.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return vis
