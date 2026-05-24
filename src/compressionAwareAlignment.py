from mvextractor.videocap import VideoCap
from spatialAlignment import detect_nadir_panel


# =================================================================================================================================

def extract_motion_vectors(video_path):

    cap = VideoCap()
    cap.open(video_path)

    # (optional) skip decoding frames
    cap.set_decode_frames(False)

    while True:
        ret, frame, motion_vectors, frame_type = cap.read()
        if not ret:
            break
        # print(f"Num. motion vectors: {len(motion_vectors)}")
        # print(f"Frame type: {frame_type}")
        if frame is not None:
            print(f"Frame size: {frame.shape}")

    cap.release()

    return motion_vectors


# =================================================================================================================================

def refine_homography_with_motion_vectors(rgb_frame, ir_frame, H_initial, pair_motion_vectors,
                                         rgb_frame_metadata, ir_frame_metadata, panel_dimensions):
    """
    Refine homography using motion vectors from video compression that are already mapped to the frame pair.

    Args:
        rgb_frame: RGB frame
        ir_frame: IR frame
        H_initial: Initial homography matrix
        pair_motion_vectors: Dictionary containing motion vectors for both frames
        rgb_frame_metadata: Metadata for RGB frame
        ir_frame_metadata: Metadata for IR frame
        panel_dimensions: Tuple of (width, height) in meters

    Returns:
        Refined homography matrix
    """
    # 1. Detect panel in RGB frame
    rgb_panel = detect_nadir_panel(rgb_frame, panel_dimensions,
                                 focal_len=rgb_frame_metadata['focal_len'],
                                 rel_alt=rgb_frame_metadata['rel_alt'])

    if rgb_panel is None:
        print("Warning: No panel detected in RGB frame, using initial homography")
        return H_initial

    # 2. Get motion vectors from the synchronized pair
    rgb_mvs = pair_motion_vectors['rgb']
    ir_mvs = pair_motion_vectors['ir']

    if not rgb_mvs or not ir_mvs:
        print("Warning: Motion vectors not available for this frame pair, falling back to initial homography")
        return H_initial

    # 3. Calculate average motion vector difference
    avg_mv_diff = calculate_average_motion_vector_difference(rgb_mvs, ir_mvs)

    # 4. Project RGB panel to IR space using initial homography
    try:
        rgb_panel_reshaped = rgb_panel.reshape(-1, 1, 2).astype(np.float32)
        ir_panel_projected = cv2.perspectiveTransform(rgb_panel_reshaped, H_initial)
    except Exception as e:
        print(f"Error projecting panel to IR space: {str(e)}")
        return H_initial

    # 5. Adjust homography based on motion vector difference
    H_refined = H_initial.copy()
    H_refined[0, 2] += avg_mv_diff[0]  # Adjust x-translation
    H_refined[1, 2] += avg_mv_diff[1]  # Adjust y-translation

    # 6. Further refine using panel detection in IR frame
    ir_panel = find_panel_in_ir(ir_frame, ir_panel_projected)

    if ir_panel is not None:
        try:
            H_final, _ = cv2.findHomography(ir_panel_projected.reshape(-1, 2),
                                          ir_panel.reshape(-1, 2),
                                          cv2.RANSAC, 5.0)
            return H_final if H_final is not None else H_refined
        except Exception as e:
            print(f"Error refining homography: {str(e)}")
            return H_refined

    return H_refined

def calculate_average_motion_vector_difference(rgb_mvs, ir_mvs):
    """
    Calculate the average difference between RGB and IR motion vectors.

    Args:
        rgb_mvs: Motion vectors from RGB frame
        ir_mvs: Motion vectors from IR frame

    Returns:
        Average difference as (dx, dy)
    """
    if not rgb_mvs or not ir_mvs:
        return (0, 0)

    # Calculate average motion vector for RGB
    try:
        rgb_dx = np.mean([mv['target'][0] - mv['source'][0] for mv in rgb_mvs])
        rgb_dy = np.mean([mv['target'][1] - mv['source'][1] for mv in rgb_mvs])
    except (KeyError, TypeError):
        # Handle cases where motion vector format is different
        print("Warning: Unexpected motion vector format, using zero difference")
        return (0, 0)

    # Calculate average motion vector for IR
    try:
        ir_dx = np.mean([mv['target'][0] - mv['source'][0] for mv in ir_mvs])
        ir_dy = np.mean([mv['target'][1] - mv['source'][1] for mv in ir_mvs])
    except (KeyError, TypeError):
        # Handle cases where motion vector format is different
        print("Warning: Unexpected motion vector format, using zero difference")
        return (0, 0)

    # Return the difference
    return (ir_dx - rgb_dx, ir_dy - rgb_dy)