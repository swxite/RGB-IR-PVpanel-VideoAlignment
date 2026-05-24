import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from panel_detection import detect_panel_array_bbox

def test_and_visualize_detection(rgb_path, ir_path):
    """
    Test the detection function on RGB and IR frames and visualize results side by side.

    Args:
        rgb_path: Path to RGB image
        ir_path: Path to IR (thermal) image
        detection_func: The detection function to test (detect_panel_array_bbox)
    """
    # Load images
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path)

    if rgb_img is None or ir_img is None:
        print("Error: Could not load one or both images")
        return

    # Convert to RGB for matplotlib (OpenCV loads as BGR)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # Detect bounding boxes
    rgb_bbox = detect_panel_array_bbox(rgb_img, modality="rgb")
    ir_bbox = detect_panel_array_bbox(ir_img, modality="ir")

    # Draw bounding boxes
    def draw_bbox(img, bbox, color=(255, 0, 0), thickness=2):
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Create copies for visualization
    rgb_viz = rgb_img.copy()
    ir_viz = cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB)  # Convert IR to RGB for display

    draw_bbox(rgb_viz, rgb_bbox, (255, 0, 0))  # Blue for RGB
    draw_bbox(ir_viz, ir_bbox, (0, 0, 255))    # Red for IR

    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(rgb_viz)
    ax1.set_title('RGB Detection')
    ax1.axis('off')

    ax2.imshow(ir_viz)
    ax2.set_title('IR Detection')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # Print detection results
    print(f"RGB Detection: {rgb_bbox}")
    print(f"IR Detection: {ir_bbox}")

# Example usage:
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
data_path = os.path.join(parent_dir, "data", "frames")
rgb = os.path.join(data_path, "DJI_20240621133005_0019_T", "frames")
ir = os.path.join(data_path, "undistorted", "0019T_frame_1291.png_undistorted.png")
test_and_visualize_detection("rgb_image.jpg", "ir_image.jpg")