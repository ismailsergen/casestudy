import argparse
from pathlib import Path

import cv2 as cv

from common import visualize
from detect_rectangles import detect_rectangles
from detect_roi import detect_roi


def build_argparser():
    """Create CLI arguments for input and output paths."""
    p = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline: extract ROI first, then detect rectangles "
            "with built-in parameters."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("Test Data"),
        help="Input image or directory",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Results"),
        help="Output directory path",
    )
    return p


def main():
    """Run ROI extraction and rectangle detection for each input TIFF image."""
    args = build_argparser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tiff_files = (
        [args.input]
        if args.input.is_file()
        else sorted(list(args.input.rglob("*.tif")) + list(args.input.rglob("*.tiff")))
    )

    for tiff_file in tiff_files:
        gray_img = cv.imread(str(tiff_file), cv.IMREAD_GRAYSCALE)
        if gray_img is None:
            print(f"Could not read: {tiff_file}")
            continue

        roi_img, (rx, ry) = detect_roi(gray_img)
        rect_list, morph_roi, min_area, max_area, group_bbox, raw_count = (
            detect_rectangles(roi_img)
        )
        vis = visualize(roi_img, rect_list, group_bbox=group_bbox)

        stem = tiff_file.stem
        cv.imwrite(str(args.output / f"{stem}_bbox.png"), vis)
        print(
            f"{tiff_file.name}: raw={raw_count} grouped={len(rect_list)} roi_offset=({rx},{ry}) "
            f"min_area={min_area:.1f} max_area={max_area:.1f}"
        )


if __name__ == "__main__":
    main()
