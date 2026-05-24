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
