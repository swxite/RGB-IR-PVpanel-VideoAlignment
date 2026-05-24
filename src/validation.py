import cv2, time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from spatialAlignment import detect_nadir_panel, find_panel_in_ir,  detect_panel_with_fallbacks
from matplotlib.gridspec import GridSpec
# The results will be evaluated using multiple metrics such as reprojection error and intersection over union (IoU) for quantitative assessment, 
# supplemented by structural similarity measures and visual inspection for qualitative validation


# validation per frame pair
def calc_reprojection_error(frame1_points, frame2_points, homography):
    """
    Calculate the reprojection error between corresponding points in RGB and IR frames.

    Args:
        frame1_points: Nx2 array of points in the RGB frame (detected panel corners).
        frame2_points: Nx2 array of points in the IR frame (projected from RGB using homography).
        homography: 3x3 homography matrix transforming IR to RGB space.

    Returns:
        Mean reprojection error in pixels.
    """
    # Check if either input is None
    if frame1_points is None or frame2_points is None:
        print("Warning: One or both input point sets are None")
        return None

    # Check if points have the correct shape
    if len(frame1_points.shape) != 2 or len(frame2_points.shape) != 2:
        print(f"Warning: Invalid point shape. frame1_points shape: {frame1_points.shape}, frame2_points shape: {frame2_points.shape}")
        return None

    # Check if both point sets have the same number of points
    if len(frame1_points) != len(frame2_points):
        print(f"Warning: Different number of points. frame1_points: {len(frame1_points)}, frame2_points: {len(frame2_points)}")
        return None

    try:
        # Convert points to homogeneous coordinates
        rgb_homogeneous = np.column_stack((frame1_points, np.ones(len(frame1_points))))

        # Project IR points to RGB space using the homography
        # Note: frame2_points needs to be reshaped to (N, 1, 2) for perspectiveTransform
        projected_points = cv2.perspectiveTransform(frame2_points.reshape(-1, 1, 2).astype(np.float32), homography)
        projected_points = projected_points.reshape(-1, 2)

        # Calculate Euclidean distance between actual and projected points
        errors = np.linalg.norm(frame1_points - projected_points, axis=1)
        mean_error = np.mean(errors)

        return mean_error
    
    except Exception as e:
        print(f"Error calculating reprojection error: {str(e)}")
        return None


def calc_IoU(frame1, frame2, panel1, panel2):
    if panel1 is None or panel2 is None:
        return 0.0

    # Create masks
    mask1 = np.zeros_like(frame1[:,:,0])
    mask2 = np.zeros_like(frame1[:,:,0])
    
    # Ensure panel corners are in correct format for cv2.drawContours
    # (N, 1, 2) with integer type
    panel1_contour = panel1.reshape(-1, 1, 2).astype(np.int32)
    panel2_contour = panel2.reshape(-1, 1, 2).astype(np.int32)
    
    cv2.drawContours(mask1, [panel1_contour], -1, 255, -1)
    cv2.drawContours(mask2, [panel2_contour], -1, 255, -1)

    # Calculate IoU
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    return iou

#  structural similarity
def calc_SSIM(frame1, frame2):
    """
    Calculate the Structural Similarity Index (SSIM) between RGB and aligned IR frames.

    Args:
        frame1: Input RGB frame (BGR format).
        frame2: Aligned IR frame (grayscale or BGR).

    Returns:
        SSIM score (range: -1 to 1, where 1 indicates perfect similarity).
    """
    # Convert frames to grayscale if needed
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1

    if len(frame2.shape) == 3:
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        frame2_gray = frame2

    # Ensure frames are the same size
    if frame1_gray.shape != frame2_gray.shape:
        frame2_gray = cv2.resize(frame2_gray, (frame1_gray.shape[1], frame1_gray.shape[0]))

    # Calculate SSIM
    score, _ = ssim(frame1_gray, frame2_gray, full=True)
    return score

def visulize_panel_alignment(frame1, panel_corners1, frame2, panel_corners2, homography=None):
    """
    Visualize the alignment of the nadir panel between RGB and IR frames.

    Args:
        frame1: Input RGB frame (BGR format)
        rgb_corners: Detected panel corners in RGB frame (4x2 numpy array)
        ir_frame: Input IR frame (grayscale or BGR)
        ir_corners: Detected or projected panel corners in IR frame (4x2 numpy array)
        homography: Optional homography matrix to show alignment transformation

    Returns:
        Combined visualization of both frames with panel overlays
    """
    if panel_corners1 is None:
        print("[WARN] rgb panel missing — skipping visualization")
        return
    if panel_corners2 is None:
        print("[WARN] ir panel missing — skipping visualization")
        return
    # Get frame dimensions for visualization
    rgb_h, rgb_w = frame1.shape[:2]
    ir_h, ir_w = frame2.shape[:2]
    
    # Create copies to avoid modifying original frames
    frame1_vis = frame1.copy()
    
    # Prepare IR frame: resize to match RGB height, then pad vertically to center
    # Scale factor: make IR height match RGB height
    scale = rgb_h / ir_h
    ir_resized = cv2.resize(frame2, (int(ir_w * scale), rgb_h))
    
    # Create a canvas with RGB height and IR resized width
    # For vertical centering, we need both frames to have same height
    # Since we resized IR to match RGB height, they now have the same height
    frame2_vis = ir_resized
    
    # Scale IR panel corners to match resized frame
    if panel_corners2 is not None:
        panel_corners2 = panel_corners2 * scale
    
    # Get updated dimensions
    h, w = frame1_vis.shape[:2]
    ir_w_scaled = frame2_vis.shape[1]

    # Convert IR to 3-channel if needed for consistent visualization
    if len(frame2_vis.shape) == 2:
        frame2_vis = cv2.cvtColor(frame2_vis, cv2.COLOR_GRAY2BGR)

    # Draw RGB panel (green)
    cv2.drawContours(frame1_vis, [panel_corners1.reshape(-1, 1, 2).astype(np.int32)], -1, (255, 0, 255), 2)

    # Draw IR panel (red)
    if panel_corners2 is not None:
        cv2.drawContours(frame2_vis, [panel_corners2.reshape(-1, 1, 2).astype(np.int32)], -1, (255, 0, 0), 2)

    # Add corner labels for clarity
    if panel_corners1 is not None and panel_corners2 is not None:
        for i, (corner_rgb, corner_ir) in enumerate(zip(panel_corners1, panel_corners2)):
            cv2.putText(frame1_vis, str(i), tuple(corner_rgb.reshape(-1).astype(np.int32)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame2_vis, str(i), tuple(corner_ir.reshape(-1).astype(np.int32)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Create side-by-side visualization
    combined = np.hstack((frame1_vis, frame2_vis))

    # Add homography visualization if provided
    if homography is not None:
        # Create a grid to show transformation
        grid_size = 20
        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                # Transform grid points
                point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
                transformed = cv2.perspectiveTransform(point, homography)
                tx, ty = transformed[0,0]

                # Draw lines between original and transformed points
                if tx < ir_w_scaled and ty < h:  # Only draw if within resized IR frame bounds
                    cv2.line(combined,
                            (x, y),
                            (w + int(tx), int(ty)),
                            (255, 0, 255), 1)

    # Add labels and metadata info
    cv2.putText(combined, "RGB Frame", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(combined, "IR Frame", (w + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Add legend
    cv2.putText(combined, "Green: RGB Panel", (10, h - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(combined, "Red: IR Panel", (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if homography is not None:
        cv2.putText(combined, "Magenta: Homography Grid", (10, h - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.imshow("Panel Alignment", combined)
    cv2.waitKey(1)  # Required to update the window
    time.sleep(15)   # Keep window open for 3 seconds
    cv2.destroyAllWindows()

    return combined

def validate_frame_pair(frame1, meta1, frame2, meta2, panel_dimensions, H, visualize=False, ir_resized=None):
    # detect panels in both frames
    # rgb_panel = detect_nadir_panel(frame1, panel_dimensions, meta1['focal_len'], meta1['rel_alt'])
    rgb_panel = detect_panel_with_fallbacks(frame1, meta1, panel_dimensions, True)
    ir_panel = detect_panel_with_fallbacks(frame2, meta2, panel_dimensions, False) 
    # ir_panel = detect_nadir_panel(frame2, panel_dimensions, meta2['focal_len'], meta2['rel_alt'])  # Use same detector
    # ir_corners_projected = cv2.perspectiveTransform(rgb_panel.reshape(-1, 1, 2), H)
    # ir_panel = find_panel_in_ir(frame2, ir_corners_projected)

    if rgb_panel is None:
        print("[WARN] Skipping frame validation: missing RGB panel detection")
        return np.nan, np.nan, np.nan
    if ir_panel is None:
        print("[WARN] Skipping frame validation: missing IR panel detection")
        return np.nan, np.nan, np.nan

    reproj_error = calc_reprojection_error(rgb_panel, ir_panel, H)
    iou = calc_IoU(frame1, frame2, rgb_panel, ir_panel)
    ssim_score = calc_SSIM(frame1, frame2)

    if visualize:
        # Use ir_resized for visualization if provided (shows original IR, not warped)
        vis_frame2 = ir_resized if ir_resized is not None else frame2
        
        # If we're showing ir_resized but panels were detected on ir_aligned (frame2),
        # we need to transform the panel corners back to ir_resized coordinates
        vis_ir_panel = ir_panel
        if ir_resized is not None and ir_resized.shape != frame2.shape:
            # frame2 is the warped/aligned IR, ir_resized is the original resized IR
            # We need to transform panel corners from warped space back to resized space
            # This requires the inverse homography
            if H is not None:
                # Transform panel corners from warped (frame2) back to resized (ir_resized)
                # ir_resized was the input to align_frames_spatially which used H
                # So we need to apply inverse of H
                H_inv = np.linalg.inv(H)
                # Reshape panel corners for perspectiveTransform: (4, 1, 2)
                ir_panel_reshaped = ir_panel.reshape(-1, 1, 2).astype(np.float32)
                # Apply inverse transform
                vis_ir_panel_reshaped = cv2.perspectiveTransform(ir_panel_reshaped, H_inv)
                vis_ir_panel = vis_ir_panel_reshaped.reshape(-1, 2)
            else:
                # Without homography, just scale the corners
                scale_y = ir_resized.shape[0] / frame2.shape[0]
                scale_x = ir_resized.shape[1] / frame2.shape[1]
                vis_ir_panel = ir_panel * np.array([scale_x, scale_y])
        
        visulize_panel_alignment(frame1, rgb_panel, vis_frame2, vis_ir_panel)

    return reproj_error, iou, ssim_score


# =================================================================================================================
def visualize_validation_metrics(results_reproj_error, results_iou, results_ssim, save_dir=None):
    """
    Visualize the three validation metrics and their averages in a single graph.

    Args:
        results_reproj_error: List of reprojection error values
        results_iou: List of IoU values
        results_ssim: List of SSIM values
        save_path: Optional path to save the figure

    Returns:
        Dictionary containing average metrics
    """
    # Filter out NaN values
    valid_reproj = [r for r in results_reproj_error if not np.isnan(r)]
    valid_iou = [r for r in results_iou if not np.isnan(r)]
    valid_ssim = [r for r in results_ssim if not np.isnan(r)]

    # Calculate averages and other statistics
    avg_reproj = np.mean(valid_reproj) if valid_reproj else np.nan
    min_reproj = np.min(valid_reproj) if valid_reproj else np.nan
    max_reproj = np.max(valid_reproj) if valid_reproj else np.nan
    std_reproj = np.std(valid_reproj) if valid_reproj else np.nan

    avg_iou = np.mean(valid_iou) if valid_iou else np.nan
    min_iou = np.min(valid_iou) if valid_iou else np.nan
    max_iou = np.max(valid_iou) if valid_iou else np.nan
    std_iou = np.std(valid_iou) if valid_iou else np.nan

    avg_ssim = np.mean(valid_ssim) if valid_ssim else np.nan
    min_ssim = np.min(valid_ssim) if valid_ssim else np.nan
    max_ssim = np.max(valid_ssim) if valid_ssim else np.nan
    std_ssim = np.std(valid_ssim) if valid_ssim else np.nan

    # Create figure 1: Reprojection Error
    plt.figure(figsize=(10, 6))
    plt.plot(valid_reproj, 'o-', color='tab:blue', label='Reprojection Error')
    plt.axhline(y=avg_reproj, color='tab:red', linestyle='--', label=f'Average: {avg_reproj:.2f} px')
    plt.ylabel('Error (pixels)')
    plt.title('Reprojection Error Across Frame Pairs')
    plt.xlabel('Frame Pair Index')
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(f"{save_dir}/reprojection_error.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show(block=False)

    # Create figure 2: IoU
    plt.figure(figsize=(10, 6))
    plt.plot(valid_iou, 'o-', color='tab:green', label='IoU')
    plt.axhline(y=avg_iou, color='tab:red', linestyle='--', label=f'Average: {avg_iou:.3f}')
    plt.ylabel('IoU Score')
    plt.title('Intersection over Union (IoU) Across Frame Pairs')
    plt.xlabel('Frame Pair Index')
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(f"{save_dir}/iou.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show(block=False)

    # Create figure 3: SSIM
    plt.figure(figsize=(10, 6))
    plt.plot(valid_ssim, 'o-', color='tab:purple', label='SSIM')
    plt.axhline(y=avg_ssim, color='tab:red', linestyle='--', label=f'Average: {avg_ssim:.3f}')
    plt.ylabel('SSIM Score')
    plt.title('Structural Similarity Index (SSIM) Across Frame Pairs')
    plt.xlabel('Frame Pair Index')
    plt.legend()
    plt.grid(True)

    if save_dir:
        plt.savefig(f"{save_dir}/ssim.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show(block=False)

    # Create figure 4: Summary Table
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    plt.title('Validation Metrics Summary', fontsize=14, pad=20)

    # Create table data
    table_data = [
        ['Metric', 'Average', 'Minimum', 'Maximum', 'Std Dev', 'Valid Frames'],
        ['Reprojection Error (px)', f'{avg_reproj:.2f}', f'{min_reproj:.2f}',
         f'{max_reproj:.2f}', f'{std_reproj:.2f}', f'{len(valid_reproj)}'],
        ['IoU', f'{avg_iou:.3f}', f'{min_iou:.3f}', f'{max_iou:.3f}',
         f'{std_iou:.3f}', f'{len(valid_iou)}'],
        ['SSIM', f'{avg_ssim:.3f}', f'{min_ssim:.3f}', f'{max_ssim:.3f}',
         f'{std_ssim:.3f}', f'{len(valid_ssim)}']
    ]

    # Create table
    table = plt.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Add overall performance indicator
    if avg_iou > 0.7 and avg_ssim > 0.7 and avg_reproj < 5:
        performance = "GOOD"
        color = 'green'
    elif avg_iou > 0.5 and avg_ssim > 0.5 and avg_reproj < 10:
        performance = "MODERATE"
        color = 'orange'
    else:
        performance = "POOR"
        color = 'red'

    plt.figtext(0.5, 0.1, f"Overall Performance: {performance}",
                ha="center", fontsize=12, bbox={"facecolor":color, "alpha":0.5, "pad":5})

    if save_dir:
        plt.savefig(f"{save_dir}/summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show(block=False)

    # Create a DataFrame for easy export
    summary_df = pd.DataFrame({
        'Metric': ['Reprojection Error', 'IoU', 'SSIM'],
        'Average': [avg_reproj, avg_iou, avg_ssim],
        'Minimum': [min_reproj, min_iou, min_ssim],
        'Maximum': [max_reproj, max_iou, max_ssim],
        'Std Dev': [std_reproj, std_iou, std_ssim],
        'Valid Frames': [len(valid_reproj), len(valid_iou), len(valid_ssim)]
    })
    plt.show()

    if save_dir:
        summary_df.to_csv(f"{save_dir}/validation_summary.csv", index=False)

    return {
        "avg_reprojection_error": avg_reproj,
        "avg_iou": avg_iou,
        "avg_ssim": avg_ssim,
        "num_valid_frames": len(valid_reproj),
        "summary_table": summary_df
    }


def overall_validation(results_reproj_error, results_iou, results_ssim):
    avg_reproj_error = np.mean([r for r in results_reproj_error])
    avg_iou = np.mean([r for r in results_iou])
    avg_ssim = np.mean([r for r in results_ssim])
    return {
        "avg_reprojection_error": avg_reproj_error,
        "avg_iou": avg_iou,
        "avg_ssim": avg_ssim
    }