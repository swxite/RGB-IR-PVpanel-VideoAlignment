import numpy as np
from datetime import timedelta

#  synchronize frames temporal by timestamp

def sync_frames(rgb_metadata, ir_metadata, max_time_diff=timedelta(milliseconds=50)):
    """ """
    synced = []
    for rgb in rgb_metadata:
        # Find closest IR frame by timestamp
        closest_ir = min(
            ir_metadata,
            key=lambda x: abs((x['timestamp'] - rgb['timestamp']).total_seconds())
        )
        time_diff = abs((rgb['timestamp'] - closest_ir['timestamp']).total_seconds())
        if time_diff <= max_time_diff.total_seconds():
            synced.append({
                'rgb_frame': rgb['frame_id'],
                'ir_frame': closest_ir['frame_id'],
                'time_diff': time_diff,
                'metadata': {
                    'rgb': rgb,
                    'ir': closest_ir
                }
            })
    return synced



def sync_frames_with_motion_vectors(rgb_metadata, ir_metadata, rgb_motion_vectors=None, ir_motion_vectors=None, max_time_diff=0.05):
    """
    Synchronize RGB and IR frames based on timestamps and map corresponding motion vectors.

    Args:
        rgb_metadata: List of RGB frame metadata
        ir_metadata: List of IR frame metadata
        rgb_motion_vectors: Dictionary of RGB motion vectors (frame_index -> vectors)
        ir_motion_vectors: Dictionary of IR motion vectors (frame_index -> vectors)
        max_time_diff: Maximum allowed timestamp difference in seconds

    Returns:
        List of synchronized frame pairs with metadata and motion vectors
    """
    synced_data = []

    # Convert timestamps to seconds since start for easier comparison
    rgb_start_time = rgb_metadata[0]['timestamp']
    ir_start_time = ir_metadata[0]['timestamp']

    # Create lists with relative timestamps
    rgb_frames = []
    for meta in rgb_metadata:
        rel_time = (meta['timestamp'] - rgb_start_time).total_seconds()
        rgb_frames.append({
            'frame_id': meta['frame_id'],
            'timestamp': meta['timestamp'],
            'rel_time': rel_time,
            'metadata': meta,
            'motion_vectors': rgb_motion_vectors.get(meta['frame_id'], []) if rgb_motion_vectors.size > 0 else []
        })

    ir_frames = []
    for meta in ir_metadata:
        rel_time = (meta['timestamp'] - ir_start_time).total_seconds()
        ir_frames.append({
            'frame_id': meta['frame_id'],
            'timestamp': meta['timestamp'],
            'rel_time': rel_time,
            'metadata': meta,
            'motion_vectors': ir_motion_vectors.get(meta['frame_id'], []) if ir_motion_vectors.size > 0 else []
        })

    # Sort frames by relative time
    rgb_frames.sort(key=lambda x: x['rel_time'])
    ir_frames.sort(key=lambda x: x['rel_time'])

    # Synchronize frames
    rgb_idx = 0
    ir_idx = 0

    while rgb_idx < len(rgb_frames) and ir_idx < len(ir_frames):
        rgb_frame = rgb_frames[rgb_idx]
        ir_frame = ir_frames[ir_idx]

        # Calculate time difference
        time_diff = abs(rgb_frame['rel_time'] - ir_frame['rel_time'])

        if time_diff <= max_time_diff:
            # Found matching pair
            synced_data.append({
                'rgb_frame': rgb_frame['frame_id'],
                'ir_frame': ir_frame['frame_id'],
                'time_diff': time_diff,
                'metadata': {
                    'rgb': rgb_frame['metadata'],
                    'ir': ir_frame['metadata']
                },
                'motion_vectors': {
                    'rgb': rgb_frame['motion_vectors'],
                    'ir': ir_frame['motion_vectors']
                }
            })

            # Move to next frames
            rgb_idx += 1
            ir_idx += 1
        elif rgb_frame['rel_time'] < ir_frame['rel_time']:
            # RGB frame is earlier, move to next RGB frame
            rgb_idx += 1
        else:
            # IR frame is earlier, move to next IR frame
            ir_idx += 1

    return synced_data
