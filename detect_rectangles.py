import cv2 as cv
import numpy as np

from common import (
    MAX_AR,
    MAX_AREA_RATIO,
    MIN_AR,
    MIN_AREA_RATIO,
    MIN_FILL,
    calculate_kernel_size,
)

MIN_CLUSTER_SIZE = 6
SIZE_SIMILARITY = 0.55
MAX_DX_FACTOR = 2.8
MAX_DY_FACTOR = 1.8


def apply_sharpen(src: np.ndarray):
    """Sharpen ROI before gradient extraction."""
    base = src.astype(np.float32)
    conv_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    sharpened = cv.filter2D(base, cv.CV_32F, conv_kernel)
    out = cv.addWeighted(base, 0.5, sharpened, 0.5, 0.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_sobel_mag(src: np.ndarray):
    """Compute Sobel magnitude map from sharpened ROI."""
    ksize = min(31, calculate_kernel_size(src.shape))
    src_f = src.astype(np.float32) / 255.0
    rx = cv.Sobel(src_f, cv.CV_32F, 1, 0, ksize=ksize)
    ry = cv.Sobel(src_f, cv.CV_32F, 0, 1, ksize=ksize)
    resp = cv.magnitude(rx, ry)
    disp = np.abs(resp)
    dmin = float(disp.min())
    dmax = float(disp.max())
    return ((disp - dmin) / (dmax - dmin + 1e-6) * 255.0).astype(np.uint8)


def apply_morphology(sobel_roi: np.ndarray):
    """Suppress small edge noise with morphological opening on Sobel output."""
    morph_kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    return cv.morphologyEx(sobel_roi, cv.MORPH_OPEN, morph_kernel, iterations=1)


def preprocess(roi: np.ndarray):
    """Run sharpen, Sobel magnitude, and morphology to prepare contour extraction."""
    sharpen_roi = apply_sharpen(roi)
    sobel_roi = apply_sobel_mag(sharpen_roi)
    morph_roi = apply_morphology(sobel_roi)
    return morph_roi


def extract_contours(
    thresholded: np.ndarray,
    retrieval_mode: int = cv.RETR_TREE,
    chain_approximation: int = cv.CHAIN_APPROX_SIMPLE,
    min_area: float = 0.0,
    max_area: float = 0.0,
    min_ar: float = MIN_AR,
    max_ar: float = MAX_AR,
    min_fill: float = MIN_FILL,
):
    """Find contours and keep rectangle candidates using area, aspect ratio, and fill filters."""
    contours, _hier = cv.findContours(
        thresholded,
        retrieval_mode,
        chain_approximation,
    )

    kept: list[tuple[int, int, int, int, float, float, float]] = []
    for cnt in contours:
        area = float(cv.contourArea(cnt))
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue

        x, y, w, h = cv.boundingRect(cnt)
        if h == 0 or w == 0:
            continue
        ar = float(w) / float(h)
        fill = area / float(w * h)
        if ar < min_ar:
            continue
        if max_ar > 0 and ar > max_ar:
            continue
        if fill < min_fill:
            continue

        kept.append((x, y, w, h, area, ar, fill))

    return kept


def _cluster_score(
    component: list[int],
    rects: list[tuple[int, int, int, int, float, float, float]],
):
    """Score a rectangle cluster by size and spatial density within its bounding box."""
    xs = [rects[i][0] for i in component]
    ys = [rects[i][1] for i in component]
    x2 = [rects[i][0] + rects[i][2] for i in component]
    y2 = [rects[i][1] + rects[i][3] for i in component]
    bw = max(x2) - min(xs)
    bh = max(y2) - min(ys)
    bbox_area = float(max(1, bw * bh))
    density = len(component) / bbox_area
    return len(component), density


def group_rectangles(
    rects: list[tuple[int, int, int, int, float, float, float]],
    min_cluster_size: int = MIN_CLUSTER_SIZE,
):
    """Group spatially and geometrically similar rectangles, then keep the best cluster."""
    if len(rects) < min_cluster_size:
        return rects, None

    n = len(rects)
    neighbors = [set() for _ in range(n)]
    centers = []
    for (x, y, w, h, *_rest) in rects:
        centers.append((x + 0.5 * w, y + 0.5 * h, w, h))

    for i in range(n):
        cxi, cyi, wi, hi = centers[i]
        for j in range(i + 1, n):
            cxj, cyj, wj, hj = centers[j]
            w_ref = max(wi, wj)
            h_ref = max(hi, hj)
            w_sim = min(wi, wj) / float(w_ref + 1e-6)
            h_sim = min(hi, hj) / float(h_ref + 1e-6)
            if w_sim < SIZE_SIMILARITY or h_sim < SIZE_SIMILARITY:
                continue

            dx = abs(cxi - cxj)
            dy = abs(cyi - cyj)
            if dx <= MAX_DX_FACTOR * w_ref and dy <= MAX_DY_FACTOR * h_ref:
                neighbors[i].add(j)
                neighbors[j].add(i)

    visited = [False] * n
    components: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in neighbors[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(comp)

    valid_components = [c for c in components if len(c) >= min_cluster_size]
    if not valid_components:
        return rects, None

    best = max(valid_components, key=lambda c: _cluster_score(c, rects))
    grouped = [rects[i] for i in best]

    xs = [r[0] for r in grouped]
    ys = [r[1] for r in grouped]
    x2 = [r[0] + r[2] for r in grouped]
    y2 = [r[1] + r[3] for r in grouped]
    group_bbox = (
        min(xs),
        min(ys),
        max(x2) - min(xs),
        max(y2) - min(ys),
    )
    return grouped, group_bbox


def detect_rectangles(roi: np.ndarray):
    """Detect, filter, and cluster rectangle candidates inside the normalized ROI image."""
    roi_area = float(roi.shape[0] * roi.shape[1])
    min_area = roi_area * MIN_AREA_RATIO
    max_area = roi_area * MAX_AREA_RATIO

    morph_roi = preprocess(roi)
    raw_boxes = extract_contours(
        morph_roi,
        min_area=min_area,
        max_area=max_area,
    )
    grouped_boxes, group_bbox = group_rectangles(raw_boxes)
    return grouped_boxes, morph_roi, min_area, max_area, group_bbox, len(raw_boxes)
