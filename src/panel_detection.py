import cv2
import numpy as np


def detect_panel_array_bbox(img, modality="rgb"):
    """
    Detect bounding box of the full PV panel array.

    Works for both RGB and IR images.

    Returns:
        bbox = (x, y, w, h)
    """

    # --------------------------------------------------
    # 1. Preprocessing
    # --------------------------------------------------
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --------------------------------------------------
    # 2. Modality-specific enhancement
    # --------------------------------------------------
    if modality == "rgb":
        # RGB: strong edges
        edges = cv2.Canny(gray, 50, 150)

    else:
        # IR: low contrast → enhance first
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # thermal blobs → gradient works better than Canny
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        edges = cv2.convertScaleAbs(sobelx) + cv2.convertScaleAbs(sobely)

        _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    # --------------------------------------------------
    # 3. Morphological cleanup
    # --------------------------------------------------
    kernel = np.ones((7, 7), np.uint8)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)

    # --------------------------------------------------
    # 4. Find contours
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # --------------------------------------------------
    # 5. Select largest "panel-like" region
    # --------------------------------------------------
    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)

    # --------------------------------------------------
    # 6. Optional sanity filtering
    # --------------------------------------------------
    area = w * h
    img_area = img.shape[0] * img.shape[1]

    # reject tiny detections
    if area < 0.05 * img_area:
        return None

    return (x, y, w, h)