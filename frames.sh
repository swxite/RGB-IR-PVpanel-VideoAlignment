#!/bin/zsh


# Printing all pixel formats in ffmpeg_listpixfmts.txt
ffmpeg -pix_fmts > ffmpeg_listpixfmts.txt
# Printing all options for libx264 in ffmpeg_helpx264.txt
ffmpeg -h encoder=libx264 > ffmpeg_helpx264.txt


#  extract Frames from the .mp4 File: CwdRstack_QP36_PSNR_CVVDP_sc.mp4 (compressed at QP 36)
# ffmpeg -i data/DJI_20240621133005_0019_T.MP4 data/frames/DJI_20240621133005_0019_T/0019T_frame_%04d.png
ffmpeg -i data/input_videos/DJI_20240829114550_0004_V.MP4 data/frames/DJI_20240829114550_0004_V/0004T_frame_%04d.png