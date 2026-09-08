from pathlib import Path
import cv2
import numpy as np


def apply_clahe_to_rgb(rgb_image: np.ndarray, clipLimit: float = 2.0,
                       tileGridSize: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an RGB image.

    Args:
        rgb_image (np.ndarray): Input image as a NumPy array. Can be single-channel (grayscale)
            or multi-channel (RGB).
        clipLimit (float): Threshold for contrast limiting. Default is 2.0.
        tileGridSize (tuple): Size of the grid for histogram equalization. Default is (8, 8).
    Returns:
        np.ndarray: The image after CLAHE enhancement, with the same shape as the input.
    Raises:
        ValueError: If the input image does not have a valid shape or number of channels.

    """
    # handle grayscale (2D) or color (H,W,3)
    channels = 1 if rgb_image.ndim == 2 else rgb_image.shape[2]
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    if channels == 1:
        # ensure 8-bit single channel
        img_gray = rgb_image if rgb_image.dtype == np.uint8 else (
            rgb_image.astype(np.uint8))
        clahe_enhanced_rgb = clahe.apply(img_gray)
    else:
        r, g, b = cv2.split(rgb_image)
        cr = clahe.apply(r)
        cg = clahe.apply(g)
        cb = clahe.apply(b)
        clahe_enhanced_rgb = cv2.merge([cr, cg, cb])

    return clahe_enhanced_rgb


def process_folder(input_dir: str = "corrosion_best_frames_croped",
                   output_dir: str = "clahe_frames_croped",
                   clipLimit: float = 2.0,
                   tileGridSize: tuple = (8, 8)):
    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(src.glob("*.png")) + sorted(src.glob("*.jpg")) + sorted(src.glob("*.jpeg")):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        enhanced = apply_clahe_to_rgb(
            img, clipLimit=clipLimit, tileGridSize=tileGridSize)
        out_path = dst / img_path.name
        cv2.imwrite(str(out_path), enhanced)


if __name__ == "__main__":
    process_folder()
