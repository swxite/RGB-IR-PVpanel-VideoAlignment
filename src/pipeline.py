import os, cv2
from util import parse_metadata_from_srt, load_frame, create_video_from_frames, create_side_by_side_video
from tempSynchronization import sync_frames, sync_frames_with_motion_vectors
from preprocess import resize_frame, resize_frame_FOV, resize_ir_to_exact_rgb, resize_ir_to_match_rgb_height, resize_and_pad_to_match, resize_frame_simple, correct_lens_distortion_polynomial
from spatialAlignment import compute_homography, align_frames_spatially, refine_homography_ecc
from validation import validate_frame_pair, overall_validation, visualize_validation_metrics
from compressionAwareAlignment import extract_motion_vectors, refine_homography_with_motion_vectors


def processing_pipeline(data_folder, name_video1, name_video2, panel_dimensions):
    # files
    rgb_video_path = os.path.join(data_folder, "input_videos", f"{name_video1}.MP4")
    ir_video_path = os.path.join(data_folder, "input_videos", f"{name_video2}.MP4")
    rgb_srt_path = os.path.join(data_folder, "input_videos", f"{name_video1}.SRT")
    ir_srt_path = os.path.join(data_folder, "input_videos", f"{name_video2}.SRT")
    if name_video1.endswith("_T") or name_video1.endswith("_V"):
        video_name = name_video1[:-2]

    # 1. Get neccessary data 
    # metadata
    rgb_metadata = parse_metadata_from_srt(rgb_srt_path)
    ir_metadata = parse_metadata_from_srt(ir_srt_path)
    # motion vectors
    rgb_motion_vectors = extract_motion_vectors(rgb_video_path)
    ir_motion_vectors = extract_motion_vectors(ir_video_path)
    # Open video files to get properties
    cap_rgb = cv2.VideoCapture(rgb_video_path)
    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_rgb.release()

    # 2. temporal synchronization
    # synch frsmes tempral & map motion vectors to it
    synched_data = sync_frames_with_motion_vectors(rgb_metadata, ir_metadata, rgb_motion_vectors, ir_motion_vectors)

    # loop through all frame pairs
    # Initialize lists to store aligned frames
    rgb_frames_aligned = []
    ir_frames_aligned = []
    ir_frames_resized = []  # For video output (shows full IR frame)
    # Initialize validation metrics
    validation_metrics = {
            'reprojection_error': [],
            'IoU': [],
            'SSIM': [],
            'avg_reprojection_error': None,
            'avg_IoU': None,
            'avg_SSIM': None
        }
    frame_pairs = []
    max_iterations = 5
    for i, pair in enumerate(synched_data):
        if i >= max_iterations:
            break
        #  load frames & metadata
        rgb_frame = load_frame(rgb_video_path, pair['rgb_frame'])
        ir_frame = load_frame(ir_video_path, pair['ir_frame'])
        rgb_frame_metadata = pair['metadata']['rgb']
        ir_frame_metadata = pair['metadata']['ir']

        # 3. Preprocess frame pair
        # lens distortion correction
        k1=-0.2
        focus=600
        ir_undistorted  = correct_lens_distortion_polynomial(ir_frame, k1, focus)

        # resize IR frame
        ir_resized = resize_frame_FOV(ir_undistorted, ir_frame_metadata, rgb_frame, rgb_frame_metadata)
        # ir_resized = resize_frame_simple(ir_undistorted, rgb_frame)

        # 4. Initial spatial alignment (using metadata)
        H = compute_homography(rgb_frame_metadata, ir_frame_metadata)
        H_refined = refine_homography_ecc(rgb_frame, rgb_frame_metadata, ir_resized, H, panel_dimensions)
        # H_refined = H

        # 5. Refinement alignment
        H_refined = refine_homography_with_motion_vectors(rgb_frame, ir_resized, H_refined, pair['motion_vectors'], rgb_frame_metadata, ir_frame_metadata, panel_dimensions)

        # Apply the final alignment
        ir_aligned = align_frames_spatially(rgb_frame, ir_resized, H_refined)
        # Store aligned frames for validation
        rgb_frames_aligned.append(rgb_frame)
        ir_frames_aligned.append(ir_aligned)
        
        # Store resized frames for video output (to show full IR frame)
        ir_frames_resized.append(ir_resized)

        # 6. Frame-pair validation
        visualize = False
        if i == 2:
            visualize = True
        frame_output_dir = os.path.join(data_folder, "panel_alignment")
        reproj_error, iou, ssim = validate_frame_pair(rgb_frame, rgb_frame_metadata, ir_aligned, ir_frame_metadata, panel_dimensions, H_refined, video_name, i, frame_output_dir, visualize, ir_resized)
        
        validation_metrics['reprojection_error'].append(reproj_error)
        validation_metrics['IoU'].append(iou)
        validation_metrics['SSIM'].append(ssim)


    # 6. Fuse frames to create aligned videos
    output_dir = os.path.join(data_folder, "output_videos")
    os.makedirs(output_dir, exist_ok=True)
    rgb_output_path = os.path.join(output_dir, f"{name_video1}.MP4")
    create_video_from_frames(rgb_frames_aligned, rgb_output_path, fps, (width, height))
    ir_output_path = os.path.join(output_dir, f"{name_video2}.MP4")
    # Use resized IR frames for video to show full frame (not cropped by alignment)
    # create_video_from_frames(ir_frames_resized, ir_output_path, fps, (ir_frames_resized[0].shape[1], ir_frames_resized[0].shape[0]))
    create_video_from_frames(ir_frames_aligned, ir_output_path, fps, (ir_frames_resized[0].shape[1], ir_frames_resized[0].shape[0]))

    #  7. overall validation
    averages = overall_validation(validation_metrics['reprojection_error'], validation_metrics['IoU'], validation_metrics['SSIM'])
    print(averages)

    #  visualization
    output_sideBySide = os.path.join(output_dir, f"{video_name}_SideBySide.MP4")
    # Use resized IR frames for side-by-side video to show full frame
    create_side_by_side_video(rgb_frames_aligned, ir_frames_aligned, output_sideBySide, fps)
    save_dir = os.path.join(data_folder, "metric")
    visualize_validation_metrics(validation_metrics['reprojection_error'], validation_metrics['IoU'], validation_metrics['SSIM'], video_name, save_dir)

    




if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    data_folder = "data"
    data_path = os.path.join(parent_dir, data_folder)
    rgb_video_name = "DJI_20240621133005_0019_V"
    ir_video_name =  "DJI_20240621133005_0019_T"

    panel_dim = {"width": 0.992, "height": 1.650}
    processing_pipeline(data_path, rgb_video_name, ir_video_name, panel_dim)