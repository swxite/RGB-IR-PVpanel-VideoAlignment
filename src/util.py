import pysrt
from datetime import datetime
import cv2, os, time
import numpy as np

# Extract and synchronize metadata from .srt files for both RGB and IR.


def parse_metadata_from_srt(srtFilePath):
    """Loads .srt file and parses it into structured metadata"""
    subs = pysrt.open(srtFilePath)
    metadata = []

    for sub in subs:
        text = sub.text.replace('\n', ' ')
        data = {
            'frame_id': sub.index,
            'timestamp': datetime.strptime(sub.start.to_time().strftime('%H:%M:%S.%f'), '%H:%M:%S.%f'),
            'iso': None, 'shutter': None, 'fnum': None, 'focal_len': None,
            'latitude': None, 'longitude': None, 'rel_alt': None, 'abs_alt': None,
            'gb_yaw': None, 'gb_pitch': None, 'gb_roll': None
        }

        # First remove the font tags if they exist
        if '<font' in text:
            text = text.split('>', 1)[1]
        if '</font>' in text:
                text = text.split('</font>', 1)[0]

        # Parse key-value pairs (e.g., [latitude: 55.695964])
        items = text.split('[')
        for item in items:
            if ':' in item:
                # Split into key and value, handling the closing bracket
                parts = item.split(':', 1)
                key = parts[0].strip()
                val = parts[1].split(']')[0].strip()  # Remove closing bracket
                # if key in data:
                #     data[key] = float(val.strip()) if '.' in val else val.strip()
                # Handle different metadata fields
                if key == 'focal_len':
                    data['focal_len'] = float(val)
                elif key == 'latitude':
                    data['latitude'] = float(val)
                elif key == 'longitude':
                    data['longitude'] = float(val)
                elif key == 'rel_alt':
                    data['rel_alt'] = float(val.split()[0])
                elif key == 'abs_alt':
                    data['abs_alt'] = float(val.split()[0])
                elif key == 'gb_yaw':
                    data['gb_yaw'] = float(val.split()[0])
                elif key == 'gb_pitch':
                    data['gb_pitch'] = float(val.split()[0])
                elif key == 'gb_roll':
                    data['gb_roll'] = float(val.split()[0])
                elif key == 'iso':
                    data['iso'] = val
                elif key == 'shutter':
                    data['shutter'] = val
                elif key == 'fnum':
                    data['fnum'] = val
        metadata.append(data)
        
    return metadata


def load_frame(video_path, frame_number=0):
    """ """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        return None

    cap.release()
    return frame

def visualize_aligned_frame(ir_aligned, video_name, frame_idx, title="Aligned IR Frame", save_dir=None):
    """
    Visualize a single aligned IR frame with optional color mapping.

    Args:
        ir_aligned: The aligned IR frame (can be grayscale or color)
        title: Title for the visualization window

    Returns:
        None (displays the visualization)
    """
    # Convert to 3-channel if grayscale
    if len(ir_aligned.shape) == 2:
        # Apply a colormap for better visualization of thermal data
        ir_vis = cv2.applyColorMap(ir_aligned, cv2.COLORMAP_JET)
    else:
        ir_vis = ir_aligned.copy()

    # Add a title to the frame
    cv2.putText(ir_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{video_name}_panelAlignment_frame{frame_idx}.png")

        if cv2.imwrite(output_path, ir_vis):
            print(f"✅ Alignment Image saved to: {output_path}")
        else:
            print("❌ Error: Could not save the alignemt image.")
    else:    
        # Display the frame
        cv2.imshow(title, ir_vis)
        cv2.waitKey(1)  # Required to update the window
        time.sleep(15)   # Keep window open for 3 seconds
        cv2.destroyAllWindows()


def create_video_from_frames(frames, output_path, fps, frame_size):
    """Create a video from a list of frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        # Ensure frame is in the correct format and size
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[1::-1] != frame_size:
            frame = cv2.resize(frame, frame_size)
        out.write(frame)

    out.release()
    parts = str.split(output_path, '/')
    print(f"Output video {parts[len(parts)-1]} created successfully.")



def create_side_by_side_video(rgb_frames, ir_frames, output_path, fps):
    """Create a side-by-side video for alignment visualization."""
    # Get frame dimensions
    rgb_height, rgb_width = rgb_frames[0].shape[:2]
    ir_height, ir_width = ir_frames[0].shape[:2]
    
    # Calculate IR width after resize to match RGB height
    if ir_height != rgb_height:
        scale = rgb_height / ir_height
        ir_width_scaled = int(ir_width * scale)
    else:
        ir_width_scaled = ir_width
    
    total_width = rgb_width + ir_width_scaled

    # Create VideoWriter with correct dimensions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, rgb_height))

    for rgb_frame, ir_frame in zip(rgb_frames, ir_frames):
        # Resize IR frame to match RGB height (maintaining aspect ratio)
        ir_height, ir_width = ir_frame.shape[:2]
        if ir_height != rgb_height:
            scale = rgb_height / ir_height
            ir_frame = cv2.resize(ir_frame, (int(ir_width * scale), rgb_height))
        
        # Convert IR frame to 3-channel if needed
        if len(ir_frame.shape) == 2:
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)

        # Create side-by-side frame
        side_by_side = np.hstack((rgb_frame, ir_frame))

        # Add text labels
        # cv2.putText(side_by_side, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(side_by_side, "IR Aligned", (rgb_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(side_by_side)

    out.release()
    parts = str.split(output_path, '/')
    print(f"Output video {parts[len(parts)-1]} created successfully.")