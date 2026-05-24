import numpy as np
import cv2

from preprocess import enhance_ir_frame
#  align frames spatially using metadata
#  homography
#  focal length correction]


def gps_to_translation(rgb_lat, rgb_lon, ir_lat, ir_lon, altitude):
    """Convert GPS differences to pixel translation (simplified)."""
    # Approximate meters per degree at given latitude
    lat_m_per_deg = 111320 * np.cos(np.radians(rgb_lat))
    lon_m_per_deg = 111320

    # Convert GPS differences to meters
    delta_lat = (ir_lat - rgb_lat) * lat_m_per_deg
    delta_lon = (ir_lon - rgb_lon) * lon_m_per_deg

    # Convert meters to pixels (assuming 1m = ~10 pixels at altitude=10m)
    pixels_per_meter = 10  # Adjust based on altitude
    tx = delta_lon * pixels_per_meter
    ty = delta_lat * pixels_per_meter
    return tx, ty


def compute_homography(rgb_metadata, ir_metadata):
    """Estimate homography using GPS, altitude, and gimbal angles."""
    # Convert GPS to local coordinates (simplified)
    lat_diff = ir_metadata['latitude'] - rgb_metadata['latitude']
    lon_diff = ir_metadata['longitude'] - rgb_metadata['longitude']
    alt_diff = ir_metadata['rel_alt'] - rgb_metadata['rel_alt']

    # Scale factor based on focal length (IR: 40mm, RGB: 24mm)
    scale_factor = ir_metadata['focal_len'] / rgb_metadata['focal_len']  # 40/24 = 1.67 [1][2]

    # Homography matrix (simplified; use OpenCV's findHomography for real data)
    H = np.array([
        [scale_factor, 0, lon_diff * 1e5],  # Scale + translation
        [0, scale_factor, lat_diff * 1e5],
        [0, 0, 1]
    ])
    return H

def detect_nadir_panel(frame, panel_dimensions, focal_len, rel_alt):
    """
    Detect the PV panel closest to the drone's nadir point using edge detection and contour analysis.

    Args:
        frame: Input RGB frame (numpy array)
        panel_dimensions: Panel width & height in meters
        focal_len: Camera focal length in mm 
        rel_alt: Relative altitude in meters

    Returns:
        panel_corners: Detected panel corners as numpy array (4x2)
        or None if no suitable panel found
    """
    # 1. Estimate expected panel size in pixels
    # Calculate ground sampling distance (GSD) in meters/pixel
    gsd = rel_alt / focal_len  # Simplified calculation [1]

    # Expected panel size in pixels
    expected_width_px = panel_dimensions['width'] / gsd
    expected_height_px = panel_dimensions['height'] / gsd
    expected_ratio = panel_dimensions['width'] / panel_dimensions['height']

    # Get frame dimensions
    h, w = frame.shape[:2]
    nadir_x, nadir_y = w // 2, h // 2  # Assume drone is centered

    # 2. Preprocess the image - try multiple preprocessing methods
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try adaptive thresholding first (often better than Canny for varying lighting)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Method 1: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Method 2: Canny edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Method 3: Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Try all methods and collect candidates
    all_contours = []
    for method_img, method_name in [(thresh, "adaptive"), (edges, "canny"), (otsu, "otsu")]:
        contours, _ = cv2.findContours(method_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend([(cnt, method_name) for cnt in contours])

    # 3. Filter contours based on expected panel characteristics
    panel_contours = []
    
    for cnt, method_name in all_contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Check if quadrilateral
        if len(approx) != 4:
            continue

        # Get bounding rectangle
        x, y, bbox_w, bbox_h = cv2.boundingRect(approx)

        # Check size ratio (allow 30% tolerance for robustness)
        size_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
        ratio_diff = abs(size_ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else float('inf')

        # Check area (allow 50% tolerance to handle various altitudes)
        area = cv2.contourArea(approx)
        expected_area = expected_width_px * expected_height_px
        area_diff = abs(area - expected_area) / expected_area if expected_area > 0 else float('inf')

        # Check convexity (panels should be mostly convex)
        convexity = cv2.contourArea(approx) / cv2.contourArea(cv2.convexHull(approx)) if cv2.contourArea(cv2.convexHull(approx)) > 0 else 0

        # Check if contour is in the center region (within 70% of frame)
        center_x = x + bbox_w / 2
        center_y = y + bbox_h / 2
        center_dist = np.sqrt((center_x - nadir_x)**2 + (center_y - nadir_y)**2)
        max_center_dist = min(w, h) * 0.7

        # Accept contour if it meets criteria
        if (ratio_diff < 0.3 and 
            area_diff < 0.5 and 
            convexity > 0.85 and
            center_dist < max_center_dist):
            panel_contours.append((approx, area_diff + ratio_diff, method_name))

    if not panel_contours:
        return None

    # 4. Sort by score (lower is better) and select best
    panel_contours.sort(key=lambda x: x[1])
    best_contour = panel_contours[0][0]

    # 5. Order corners (top-left, top-right, bottom-right, bottom-left)
    # Reshape contour to 4x2 array
    corners = best_contour.reshape(4, 2)

    # Calculate center
    center = np.mean(corners, axis=0)

    # Sort corners based on angle from center
    angles = np.arctan2(corners[:,1] - center[1], corners[:,0] - center[0])
    sorted_indices = np.argsort(angles)
    ordered_corners = corners[sorted_indices]

    return ordered_corners

def find_panel_in_ir(ir_frame, projected_corners, search_margin=30):
    """
    Find panel in IR frame using multiple strategies.

    Args:
        ir_frame: IR frame
        projected_corners: Projected corners from RGB frame
        search_margin: Margin around projected location to search

    Returns:
        Detected panel corners in IR frame or None if not found
    """
    # Strategy 1: Try direct detection in IR frame
    ir_panel = detect_panel_directly(ir_frame)
    if ir_panel is not None:
        return ir_panel

    # Strategy 2: Search around projected location
    ir_panel = search_around_projection(ir_frame, projected_corners, search_margin)
    if ir_panel is not None:
        return ir_panel

    # Strategy 3: Use projected corners as fallback
    print("Warning: Using projected corners as IR panel fallback")
    return projected_corners.reshape(-1, 2)


def detect_panel_directly(ir_frame):
    """Try to detect panel directly in IR frame using multiple approaches."""
    # Approach 1: Adaptive thresholding
    thresh = cv2.adaptiveThreshold(ir_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest quadrilateral
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2)

    # Approach 2: Edge detection with Hough lines
    edges = cv2.Canny(ir_frame, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                           minLineLength=50, maxLineGap=10)

    if lines is not None:
        # Find intersections to form quadrilateral
        # (Implementation depends on your specific needs)
        pass

    return None

def search_around_projection(ir_frame, projected_corners, margin):
    """Search for panel around projected location."""
    # Get bounding box around projected corners
    x, y, w, h = cv2.boundingRect(projected_corners)

    # Add margin and ensure we stay within frame bounds
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(ir_frame.shape[1], x + w + margin)
    y2 = min(ir_frame.shape[0], y + h + margin)

    # Extract ROI
    roi = ir_frame[y1:y2, x1:x2]

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest quadrilateral in ROI
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            # Convert back to original coordinates
            approx[:, :, 0] += x1
            approx[:, :, 1] += y1
            return approx.reshape(-4, 2)

    return None


def detect_panel_by_color(frame, panel_dim, focal_len, rel_alt):
    """
    Detect panel based on color characteristics (for RGB frames).
    PV panels typically have dark (black/blue) color with uniform texture.

    Args:
        frame: RGB frame
        panel_dim: Panel dimensions dictionary
        focal_len: Camera focal length
        rel_alt: Relative altitude

    Returns:
        Detected panel corners or None
    """
    h, w = frame.shape[:2]
    
    # Calculate expected panel size in pixels
    gsd = rel_alt / focal_len
    expected_width_px = panel_dim['width'] / gsd
    expected_height_px = panel_dim['height'] / gsd
    expected_ratio = panel_dim['width'] / panel_dim['height']
    
    # Nadir point (center)
    nadir_x, nadir_y = w // 2, h // 2
    
    # Define ROI around center (70% of frame)
    roi_size = min(w, h) * 0.7
    x1 = max(0, nadir_x - int(roi_size / 2))
    y1 = max(0, nadir_y - int(roi_size / 2))
    x2 = min(w, nadir_x + int(roi_size / 2))
    y2 = min(h, nadir_y + int(roi_size / 2))
    
    roi = frame[int(y1):int(y2), int(x1):int(x2)]
    
    # Convert to HSV color space - better for color segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # PV panels are typically dark and have low saturation
    # Define color thresholds for dark regions
    # Low brightness and low saturation
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 50, 80])  # Adjust based on your panel color
    
    mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    best_contour = None
    best_score = float('inf')
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) != 4:
            continue
        
        x, y, bbox_w, bbox_h = cv2.boundingRect(approx)
        
        # Check size ratio
        size_ratio = bbox_w / bbox_h if bbox_h > 0 else 0
        ratio_diff = abs(size_ratio - expected_ratio) / expected_ratio if expected_ratio > 0 else float('inf')
        
        # Check area
        area = cv2.contourArea(approx)
        expected_area = expected_width_px * expected_height_px
        area_diff = abs(area - expected_area) / expected_area if expected_area > 0 else float('inf')
        
        # Check aspect ratio tolerance
        if ratio_diff < 0.35 and area_diff < 0.6:
            score = ratio_diff + area_diff
            if score < best_score:
                best_score = score
                best_contour = approx
    
    if best_contour is None:
        return None
    
    # Convert back to full frame coordinates
    corners = best_contour.reshape(4, 2)
    corners[:, 0] += x1
    corners[:, 1] += y1
    
    # Order corners
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:,1] - center[1], corners[:,0] - center[0])
    sorted_indices = np.argsort(angles)
    ordered_corners = corners[sorted_indices]
    
    return ordered_corners


def detect_panel_with_fallbacks(frame, frame_metadata, panel_dim, is_rgb=True):
    """
    Detect panel with multiple fallback strategies.

    Args:
        frame: Input frame (RGB or IR)
        frame_metadata: Frame metadata
        panel_dim: Panel dimensions dictionary
        is_rgb: Whether the frame is RGB (True) or IR (False)

    Returns:
        Detected panel corners or None if all strategies fail
    """
    focal_len = frame_metadata['focal_len']
    rel_alt = frame_metadata['rel_alt']
    
    # Strategy 1: Color-based detection (for RGB frames only - often works well for PV panels)
    if is_rgb:
        panel = detect_panel_by_color(frame, panel_dim, focal_len, rel_alt)
        if panel is not None:
            print("[DETECT] Panel detected using color-based method")
            return panel
    
    # Strategy 2: Direct edge/contour detection (works for both RGB and IR)
    panel = detect_nadir_panel(frame, panel_dim, focal_len, rel_alt)
    if panel is not None:
        print("[DETECT] Panel detected using edge/contour method")
        return panel

    # Strategy 3: Grid-based detection (for RGB frames with visible grid lines)
    if is_rgb:
        panel = detect_panel_grid(frame, panel_dim)
        if panel is not None:
            print("[DETECT] Panel detected using grid method")
            return panel

    # Strategy 4: Metadata-based estimation (uses expected size and position)
    panel = estimate_panel_from_metadata(frame, frame_metadata, panel_dim)
    if panel is not None:
        print("[DETECT] Panel estimated from metadata")
        return panel

    # Strategy 5: Fallback to center of frame
    print("[WARN] All panel detection strategies failed, using frame center as fallback")
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Calculate expected panel size in pixels
    gsd = rel_alt / focal_len
    panel_width_px = int(panel_dim['width'] / gsd)
    panel_height_px = int(panel_dim['height'] / gsd)

    return np.array([
        [center_x - panel_width_px//2, center_y - panel_height_px//2],
        [center_x + panel_width_px//2, center_y - panel_height_px//2],
        [center_x + panel_width_px//2, center_y + panel_height_px//2],
        [center_x - panel_width_px//2, center_y + panel_height_px//2]
    ], dtype=np.float32)


def detect_panel_grid(frame, panel_dim):
    """
    Detect panel based on grid pattern in RGB frames.

    Args:
        frame: RGB frame
        panel_width_m: Panel width in meters
        panel_height_m: Panel height in meters

    Returns:
        Detected panel corners or None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find lines - increase threshold to get fewer, stronger lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=150,
                           minLineLength=100, maxLineGap=10)

    if lines is None:
        return None

    # Limit number of lines to prevent O(N^2) explosion
    # Keep only the strongest lines (sort by line length)
    max_lines = 100
    if len(lines) > max_lines:
        # Calculate line lengths and sort
        line_lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_lengths.append(length)
        # Get indices of longest lines
        top_indices = np.argsort(line_lengths)[-max_lines:]
        lines = lines[top_indices]

    # Cluster lines by angle
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)

    # Find dominant angles (horizontal and vertical)
    angles = np.array(angles)
    hist, bin_edges = np.histogram(angles, bins=180, range=(-90, 90))
    # Get indices of top 2 bins, ensuring we have at least 2
    sorted_indices = np.argsort(hist)
    if len(sorted_indices) >= 2:
        dominant_angles = bin_edges[sorted_indices[-2:]]
    else:
        return None

    # Separate lines into horizontal and vertical groups based on dominant angles
    horizontal_lines = []
    vertical_lines = []
    angle_tolerance = 10  # degrees
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if abs(angle - dominant_angles[0]) < angle_tolerance:
            horizontal_lines.append(np.array([x1, y1, x2, y2], dtype=np.float64))
        elif abs(angle - dominant_angles[1]) < angle_tolerance:
            vertical_lines.append(np.array([x1, y1, x2, y2], dtype=np.float64))

    # Find intersections - only between horizontal and vertical lines (O(N*M) not O(N^2))
    intersections = []
    h, w = frame.shape[:2]
    max_coord = max(w, h) * 2
    
    for h_line in horizontal_lines:
        hx1, hy1, hx2, hy2 = h_line
        # Skip lines with extreme coordinates
        if (abs(hx1) > max_coord or abs(hy1) > max_coord or 
            abs(hx2) > max_coord or abs(hy2) > max_coord):
            continue
        
        for v_line in vertical_lines:
            vx1, vy1, vx2, vy2 = v_line
            # Skip lines with extreme coordinates
            if (abs(vx1) > max_coord or abs(vy1) > max_coord or 
                abs(vx2) > max_coord or abs(vy2) > max_coord):
                continue
            
            # Calculate intersection
            try:
                denom = (hx1 - hx2) * (vy1 - vy2) - (hy1 - hy2) * (vx1 - vx2)
                if abs(denom) > 1e-10:
                    x = ((hx1*hy2 - hy1*hx2) * (vx1 - vx2) - (hx1 - hx2) * (vx1*vy2 - vy1*vx2)) / denom
                    y = ((hx1*hy2 - hy1*hx2) * (vy1 - vy2) - (hy1 - hy2) * (vx1*vy2 - vy1*vx2)) / denom
                    # Check for valid finite numbers within reasonable bounds
                    if (np.isfinite(x) and np.isfinite(y) and
                        0 <= x <= w * 1.5 and 0 <= y <= h * 1.5):
                        intersections.append((x, y))
            except (OverflowError, RuntimeWarning):
                continue

    if len(intersections) < 4:
        return None

    # Find convex hull of intersections
    hull = cv2.convexHull(np.array(intersections, dtype=np.float32))

    # Approximate to quadrilateral
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(-1, 2)

    return None


def estimate_panel_from_metadata(frame, frame_metadata, panel_dim):
    """
    Estimate panel position based on metadata.

    Args:
        frame: Input frame
        frame_metadata: Frame metadata
        panel_width_m: Panel width in meters
        panel_height_m: Panel height in meters

    Returns:
        Estimated panel corners or None
    """
    h, w = frame.shape[:2]

    # Calculate expected panel size in pixels
    pixel_size_m = frame_metadata['rel_alt'] / frame_metadata['focal_len']
    panel_width_px = int(panel_dim['width'] / pixel_size_m)
    panel_height_px = int(panel_dim['height'] / pixel_size_m)

    # Estimate panel position (center of frame)
    center_x, center_y = w // 2, h // 2

    # Create panel corners
    panel = np.array([
        [center_x - panel_width_px//2, center_y - panel_height_px//2],
        [center_x + panel_width_px//2, center_y - panel_height_px//2],
        [center_x + panel_width_px//2, center_y + panel_height_px//2],
        [center_x - panel_width_px//2, center_y + panel_height_px//2]
    ], dtype=np.float32)

    # Verify panel is within frame bounds
    if (np.all(panel[:, 0] >= 0) and np.all(panel[:, 0] < w) and
        np.all(panel[:, 1] >= 0) and np.all(panel[:, 1] < h)):
        return panel

    return None



# Cross-Modal Panel Matching
def refine_homography(rgb_frame, rgb_meta, ir_frame, H_initial, panel_dimensions):
    # 1. Detect panels in RGB
    rgb_panel = detect_nadir_panel(rgb_frame, panel_dimensions, rgb_meta['focal_len'], rgb_meta['rel_alt'], rgb_meta['latitude'], rgb_meta['longitude'])
    if rgb_panel is None:
        return H_initial

    # 2. Project RGB panel to IR space
    # Reshape to (N, 1, 2) format required by perspectiveTransform
    rgb_panel_reshaped = rgb_panel.reshape(-1, 1, 2).astype(np.float32)
    ir_panel_projected = cv2.perspectiveTransform(rgb_panel_reshaped, H_initial)

    # 3. Search for panel in IR around projected location
    x,y,w,h = cv2.boundingRect(ir_panel_projected)
    roi = ir_frame[y-20:y+h+20, x-20:x+w+20]  # Add margin

    # 4. Use template matching in IR ROI
    res = cv2.matchTemplate(enhance_ir_frame(roi),
                          enhance_ir_frame(ir_frame[y:y+h, x:x+w]),
                          cv2.TM_CCOEFF_NORMED)
    _, _, min_loc, max_loc = cv2.minMaxLoc(res)

    # 5. Calculate final homography
    offset_x, offset_y = max_loc
    H_final = H_initial.copy()
    H_final[0,2] += (offset_x - 20)  # Adjust for ROI margin
    H_final[1,2] += (offset_y - 20)

    return H_final


def preprocess_for_ecc(img):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    img = clahe.apply(img)

    # Gradient-based preprocessing works well cross-modal
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    grad = cv2.magnitude(gx, gy)

    grad = cv2.normalize(
        grad,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    return grad


def refine_homography_ecc(
        rgb_frame,
        rgb_meta,
        ir_frame,
        H_initial,
        panel_dimensions,
        warp_mode=cv2.MOTION_AFFINE,
        debug=False):

    h, w = ir_frame.shape[:2]

    # ---------------------------------------------------
    # 0. Detect panel in RGB
    # ---------------------------------------------------

    rgb_panel = detect_nadir_panel(
        rgb_frame,
        panel_dimensions,
        rgb_meta['focal_len'],
        rgb_meta['rel_alt']
    )

    if rgb_panel is None:
        if debug:
            print("[ECC] Panel detection failed")
        return H_initial

    rgb_panel = np.asarray(rgb_panel, dtype=np.float32)

    if rgb_panel.shape[0] < 3:
        if debug:
            print("[ECC] Invalid panel geometry")
        return H_initial

    # ---------------------------------------------------
    # 1. Project panel into IR coordinates
    # ---------------------------------------------------

    rgb_panel_reshaped = rgb_panel.reshape(-1, 1, 2)

    ir_panel = cv2.perspectiveTransform(rgb_panel_reshaped, H_initial)
    ir_panel = ir_panel.reshape(-1, 2)

    # Check validity
    if not np.isfinite(ir_panel).all():
        if debug:
            print("[ECC] NaN in projected panel")
        return H_initial

    # ---------------------------------------------------
    # 2. Compute bounding box & validate overlap
    # ---------------------------------------------------

    x, y, bw, bh = cv2.boundingRect(ir_panel.astype(np.int32))

    if bw <= 5 or bh <= 5:
        if debug:
            print("[ECC] Panel too small after projection")
        return H_initial

    # Check if completely outside frame
    if x >= w or y >= h or (x + bw) <= 0 or (y + bh) <= 0:
        if debug:
            print("[ECC] Panel outside IR frame")
        return H_initial

    # ---------------------------------------------------
    # 3. Build mask safely (clamped)
    # ---------------------------------------------------

    mask = np.zeros((h, w), dtype=np.uint8)

    try:
        cv2.fillConvexPoly(mask, ir_panel.astype(np.int32), 255)
    except cv2.error:
        if debug:
            print("[ECC] Failed to build mask")
        return H_initial

    mask_area = np.count_nonzero(mask)

    if mask_area < 500:
        if debug:
            print(f"[ECC] Mask too small: {mask_area}")
        return H_initial

    # Dilate slightly
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))

    # ---------------------------------------------------
    # 4. Crop region around panel (IMPORTANT)
    # ---------------------------------------------------

    pad = 50

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)

    rgb_crop = cv2.warpPerspective(rgb_frame, H_initial, (w, h))[y1:y2, x1:x2]
    ir_crop = ir_frame[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    if rgb_crop.size == 0 or ir_crop.size == 0:
        if debug:
            print("[ECC] Empty crop region")
        return H_initial

    # ---------------------------------------------------
    # 5. Preprocess (stable version)
    # ---------------------------------------------------

    def preprocess(img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        clahe = cv2.createCLAHE(2.0, (8, 8))
        img = clahe.apply(img)

        img = cv2.GaussianBlur(img, (5, 5), 0)

        return img

    rgb_proc = preprocess(rgb_crop)
    ir_proc = preprocess(ir_crop)

    # ---------------------------------------------------
    # 6. ECC initialization
    # ---------------------------------------------------

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
        return H_initial  # avoid unstable homography ECC by default
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5
    )

    # ---------------------------------------------------
    # 7. Run ECC
    # ---------------------------------------------------

    try:
        cc, warp_matrix = cv2.findTransformECC(
            templateImage=ir_proc,
            inputImage=rgb_proc,
            warpMatrix=warp_matrix,
            motionType=warp_mode,
            criteria=criteria,
            inputMask=mask_crop,
            gaussFiltSize=3
        )

        if debug:
            print(f"[ECC] Convergence score: {cc:.4f}")

        # Reject bad solutions
        if cc < 0.75:
            if debug:
                print("[ECC] Low convergence, rejecting result")
            return H_initial

        # ---------------------------------------------------
        # 8. Convert transform
        # ---------------------------------------------------

        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            warp_3x3 = np.eye(3, dtype=np.float32)
            warp_3x3[:2, :] = warp_matrix
        else:
            warp_3x3 = warp_matrix

        # ---------------------------------------------------
        # 9. Compose transforms
        # ---------------------------------------------------

        H_refined = warp_3x3 @ H_initial

        return H_refined

    except cv2.error as e:
        if debug:
            print("[ECC] Failed:", e)
        return H_initial
    


def align_frames_spatially(frame1, frame2, H):
    """Warp IR frame to RGB using homography."""
    h, w = frame1.shape[:2]
    return cv2.warpPerspective(frame2, H, (w, h))