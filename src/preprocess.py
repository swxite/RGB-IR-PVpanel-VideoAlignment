import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

#  correct lens distortion
#  resize frames

# time shifting
# FOV matching
# Frame Interp.


def correct_lens_distortion_polynomial(image, k1, focus=1.0):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            dx = x - center[0]
            dy = y - center[1]
            r = np.sqrt(dx**2 + dy**2) / focus

            # Polynomial distortion model
            if r > 0:
                r_undistorted = r * (1 + k1 * r**2)
            else:
                r_undistorted = 0

            map_x[y, x] = center[0] + r_undistorted * dx / r if r > 0 else center[0]
            map_y[y, x] = center[1] + r_undistorted * dy / r if r > 0 else center[1]

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)



# ==========================================================================================


def resize_frame_FOV(frame, frame_metadata, goal_frame, goal_metadata):
    """ """
    # Extract focal lengths with error checking
    focal_len = frame_metadata.get('focal_len')
    goal_focal_len = goal_metadata.get('focal_len')

    # Verify focal lengths are valid
    if focal_len is None or goal_focal_len is None:
        raise ValueError("Focal length information missing in metadata. "
                        f"Input frame focal_len: {focal_len}, "
                        f"Goal frame focal_len: {goal_focal_len}")

    if focal_len == 0 or goal_focal_len == 0:
        raise ValueError("Focal length cannot be zero")
    scale_factor = focal_len / goal_focal_len 
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Get the dimensions of the first frame
    height1, width1 = goal_frame.shape[:2]

    # Resize the second frame to match the dimensions of the first frame
    # resized_frame = cv2.resize(frame, (width1, height1))
    # return resized_frame
    return frame

def resize_frame_simple(ir_frame, rgb_frame):
     # Resize IR to match RGB height (simpler approach)
    height1, width1 = rgb_frame.shape[:2]
    ir_h, ir_w = ir_frame.shape[:2]
    if ir_h != height1:
        scale = height1 / ir_h
        ir_resized = cv2.resize(ir_frame, (int(ir_w * scale), height1))
    else:
        ir_resized = ir_frame
    
    return ir_resized

def resize_frame(frame, frame_metadata, goal_metadata):
    """
    Resize a frame to match the field of view of another camera based on focal lengths,
    while also ensuring the output has the same dimensions as the goal frame.

    Args:
        frame: Input frame to resize
        frame_metadata: Metadata of the input frame (must contain 'focal_len')
        goal_metadata: Metadata of the target frame (must contain 'focal_len' and frame dimensions)

    Returns:
        Resized and aligned frame matching the goal frame's dimensions
    """
    # Extract focal lengths with error checking
    focal_len = frame_metadata.get('focal_len')
    goal_focal_len = goal_metadata.get('focal_len')

    # Verify focal lengths are valid
    if focal_len is None or goal_focal_len is None:
        raise ValueError("Focal length information missing in metadata. "
                        f"Input frame focal_len: {focal_len}, "
                        f"Goal frame focal_len: {goal_focal_len}")

    if focal_len == 0 or goal_focal_len == 0:
        raise ValueError("Focal length cannot be zero")

    # Calculate scale factor based on focal lengths [1][2]
    scale_factor = focal_len / goal_focal_len 

    # Resize frame to match field of view
    resized_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Get goal frame dimensions (assuming they're available in metadata)
    # If not, you'll need to pass them as additional parameters
    goal_width = goal_metadata.get('width')
    goal_height = goal_metadata.get('height')

    # If dimensions aren't in metadata, use the resized frame as reference
    if goal_width is None or goal_height is None:
        return resized_frame

    # Calculate padding needed to match goal dimensions
    resized_h, resized_w = resized_frame.shape[:2]
    pad_x = (goal_width - resized_w) // 2
    pad_y = (goal_height - resized_h) // 2

    # Create new image with goal dimensions and place resized content
    if len(resized_frame.shape) == 2:  # Grayscale
        aligned_frame = np.zeros((goal_height, goal_width), dtype=np.uint8)
    else:  # Color
        aligned_frame = np.zeros((goal_height, goal_width, 3), dtype=np.uint8)

    # Place resized content in the center
    aligned_frame[pad_y:pad_y+resized_h, pad_x:pad_x+resized_w] = resized_frame

    # Optional: Apply feathering to blend edges
    if pad_x > 0 or pad_y > 0:
        aligned_frame = apply_feathering(aligned_frame, pad_x, pad_y)

    return aligned_frame

def apply_feathering(image, pad_x, pad_y):
    """
    Apply feathering to blend edges of the padded image.

    Args:
        image: Image with padding
        pad_x: Horizontal padding
        pad_y: Vertical padding

    Returns:
        Image with feathered edges
    """
    h, w = image.shape[:2]

    # Create masks for feathering
    if pad_x > 0:
        left_mask = np.linspace(0, 1, pad_x)
        right_mask = np.linspace(1, 0, pad_x)
        for i in range(pad_x):
            image[:, i] = image[:, i] * left_mask[i]
            image[:, w-1-i] = image[:, w-1-i] * right_mask[i]

    if pad_y > 0:
        top_mask = np.linspace(0, 1, pad_y)
        bottom_mask = np.linspace(1, 0, pad_y)
        for i in range(pad_y):
            image[i, :] = image[i, :] * top_mask[i]
            image[h-1-i, :] = image[h-1-i, :] * bottom_mask[i]

    return image

def resize_ir_to_exact_rgb(ir_frame, ir_focal_len, rgb_focal_len, rgb_dimensions):
    """
    Resize IR frame to exactly match RGB dimensions while preserving content.

    Args:
        ir_frame: Input IR frame
        ir_focal_len: IR camera focal length (40.00) [2]
        rgb_focal_len: RGB camera focal length (24.00) [1]
        rgb_dimensions: (width, height) of RGB frame

    Returns:
        IR frame exactly matching RGB dimensions
    """
    # 1. Calculate scale factor to match field of view
    scale_factor = ir_focal_len / rgb_focal_len  # 40/24 ≈ 1.67

    # 2. Resize IR frame to match RGB field of view
    ir_resized = cv2.resize(ir_frame, None, fx=scale_factor, fy=scale_factor)

    # 3. Get dimensions
    rgb_w, rgb_h = rgb_dimensions
    ir_h, ir_w = ir_resized.shape[:2]

    # 4. If IR is larger than RGB, crop to match dimensions
    if ir_w > rgb_w or ir_h > rgb_h:
        # Calculate crop coordinates (center crop)
        x = (ir_w - rgb_w) // 2
        y = (ir_h - rgb_h) // 2
        ir_cropped = ir_resized[y:y+rgb_h, x:x+rgb_w]
        return ir_cropped

    # 5. If IR is smaller than RGB, scale up to match height
    # This ensures no black borders while maintaining aspect ratio
    elif ir_h < rgb_h:
        # Calculate new scale factor to match height
        height_scale = rgb_h / ir_h
        ir_scaled = cv2.resize(ir_resized, None, fx=height_scale, fy=height_scale)

        # If width is now larger than RGB, crop to match
        if ir_scaled.shape[1] > rgb_w:
            x = (ir_scaled.shape[1] - rgb_w) // 2
            ir_final = ir_scaled[:, x:x+rgb_w]
            return ir_final
        else:
            # If still smaller, pad with content (not black)
            # This case shouldn't happen with your focal lengths
            return ir_scaled

    # 6. If dimensions match exactly, return as is
    return ir_resized

def resize_ir_to_match_rgb_height(ir_frame, ir_focal_len, rgb_focal_len, rgb_height):
    """
    Resize IR frame to exactly match RGB height while preserving aspect ratio.

    Args:
        ir_frame: Input IR frame
        ir_focal_len: IR camera focal length (40.00) [2]
        rgb_focal_len: RGB camera focal length (24.00) [1]
        rgb_height: Height of RGB frame

    Returns:
        IR frame with height exactly matching RGB, width adjusted proportionally
    """
    # 1. Calculate scale factor to match field of view
    fov_scale_factor = ir_focal_len / rgb_focal_len  # 40/24 ≈ 1.67

    # 2. Resize IR frame to match RGB field of view
    ir_resized = cv2.resize(ir_frame, None, fx=fov_scale_factor, fy=fov_scale_factor)

    # 3. Calculate height scale factor to match RGB height
    height_scale_factor = rgb_height / ir_resized.shape[0]

    # 4. Resize to match RGB height while preserving aspect ratio
    ir_final = cv2.resize(ir_resized, None, fx=height_scale_factor, fy=height_scale_factor)

    return ir_final

def resize_and_pad_to_match(frame, frame_metadata, goal_frame, goal_metadata):
    import cv2
    import numpy as np

    focal_len = frame_metadata['focal_len']
    goal_focal_len = goal_metadata['focal_len']

    scale = focal_len / goal_focal_len
    frame = cv2.resize(frame, None, fx=scale, fy=scale)

    gh, gw = goal_frame.shape[:2]
    h, w = frame.shape[:2]

    # create empty canvas
    canvas = np.zeros((gh, gw, 3), dtype=frame.dtype)

    # center placement
    y_offset = (gh - h) // 2
    x_offset = (gw - w) // 2

    # if frame is larger, crop BEFORE placing
    y1 = max(0, -y_offset)
    x1 = max(0, -x_offset)

    frame_crop = frame[y1:y1 + min(h, gh), x1:x1 + min(w, gw)]

    y_offset = max(0, y_offset)
    x_offset = max(0, x_offset)

    canvas[y_offset:y_offset+frame_crop.shape[0],
           x_offset:x_offset+frame_crop.shape[1]] = frame_crop

    return canvas


def enhance_ir_frame(ir_frame):
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(ir_frame)
    print(ir_frame.shape, ir_frame.dtype)
    return enhanced