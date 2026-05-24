#!/usr/bin/env python3
"""
cv_refined_from_videos.py

A more realistic traditional-CV baseline for thermal/RGB PV imagery.

Main differences from the previous script:
1) Uses the metadata-matched frame pairs you already created
2) Heuristically center-crops the visible image to account for focal-length mismatch
3) Resizes the cropped visible image to the thermal image size
4) Builds edge images (Canny) to reduce cross-modal appearance differences
5) Runs ECC alignment (affine or homography) on the edge images
6) Saves before/after overlays and a summary table

This is NOT guaranteed to solve the problem, but it is a more fair baseline than raw ORB/SIFT
on the full frames.

Example:
python cv_refined_from_videos.py ^
  --csv DJI_20240621133005_0019_matched.csv ^
  --thermal_video DJI_20240621133005_0019_T.MP4 ^
  --visible_video DJI_20240621133005_0019_V.MP4 ^
  --num_pairs 5 ^
  --visible_focal 24 ^
  --thermal_focal 40 ^
  --outdir refined_cv_results
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_csv_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def choose_rows(rows, num_pairs):
    if not rows:
        return []
    if num_pairs >= len(rows):
        return rows
    if num_pairs <= 1:
        return [rows[0]]
    idxs = np.linspace(0, len(rows) - 1, num_pairs, dtype=int)
    return [rows[i] for i in idxs]


def read_frame(video_path: Path, frame_index_1based: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    zero_based = max(0, frame_index_1based - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index_1based} from {video_path}")
    return frame


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def center_crop(img, crop_ratio):
    """crop_ratio=0.6 keeps 60% of width and height around the center."""
    h, w = img.shape[:2]
    cw = max(20, int(w * crop_ratio))
    ch = max(20, int(h * crop_ratio))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return img[y0:y0+ch, x0:x0+cw]


def normalize_gray(img):
    return cv2.equalizeHist(img)


def build_edges(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)


def overlay(base_gray, moving_gray):
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    mov_bgr = cv2.cvtColor(moving_gray, cv2.COLOR_GRAY2BGR)
    out = np.zeros_like(base_bgr)
    out[:, :, 1] = base_bgr[:, :, 1]
    out[:, :, 0] = mov_bgr[:, :, 0]
    out[:, :, 2] = mov_bgr[:, :, 2]
    return out


def run_ecc(template_img, input_img, motion_model_name="affine", iterations=300, eps=1e-5):
    """
    Align input_img to template_img.
    Returns (success, warp_matrix, cc, warped_img)
    """
    template_f = template_img.astype(np.float32) / 255.0
    input_f = input_img.astype(np.float32) / 255.0

    if motion_model_name == "translation":
        motion_model = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_model_name == "euclidean":
        motion_model = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_model_name == "affine":
        motion_model = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_model_name == "homography":
        motion_model = cv2.MOTION_HOMOGRAPHY
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        raise ValueError("Unknown motion model")

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        iterations,
        eps
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            template_f, input_f, warp_matrix, motion_model, criteria, None, 1
        )
    except cv2.error:
        return False, None, None, None

    if motion_model == cv2.MOTION_HOMOGRAPHY:
        warped = cv2.warpPerspective(
            input_img, warp_matrix,
            (template_img.shape[1], template_img.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )
    else:
        warped = cv2.warpAffine(
            input_img, warp_matrix,
            (template_img.shape[1], template_img.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

    return True, warp_matrix, float(cc), warped


def save_side_by_side(img1, img2, outpath: Path):
    h = max(img1.shape[0], img2.shape[0])

    def fit_h(img):
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    a = fit_h(img1)
    b = fit_h(img2)
    if len(a.shape) == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if len(b.shape) == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    canvas = np.hstack([a, b])
    cv2.imwrite(str(outpath), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--thermal_video", required=True)
    parser.add_argument("--visible_video", required=True)
    parser.add_argument("--num_pairs", type=int, default=5)
    parser.add_argument("--visible_focal", type=float, default=24.0)
    parser.add_argument("--thermal_focal", type=float, default=40.0)
    parser.add_argument("--motion_model", choices=["translation", "euclidean", "affine", "homography"], default="affine")
    parser.add_argument("--outdir", default="refined_cv_results")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    thermal_video = Path(args.thermal_video)
    visible_video = Path(args.visible_video)
    outdir = Path(args.outdir)
    extracted_dir = outdir / "extracted_pairs"
    prep_dir = outdir / "prepared_pairs"
    ecc_dir = outdir / "ecc"
    outdir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(exist_ok=True)
    prep_dir.mkdir(exist_ok=True)
    ecc_dir.mkdir(exist_ok=True)

    rows = read_csv_rows(csv_path)
    chosen = choose_rows(rows, args.num_pairs)
    if not chosen:
        raise RuntimeError("No rows found in matched CSV.")

    # Heuristic crop ratio based on focal lengths.
    # Since thermal focal is larger, thermal sees a narrower view.
    # Visible crop ratio ~ visible_focal / thermal_focal = 24/40 = 0.6
    crop_ratio = min(1.0, max(0.2, args.visible_focal / args.thermal_focal))

    summary_rows = []

    for i, row in enumerate(chosen, start=1):
        stem = f"pair_{i:02d}"
        t_frame = int(float(row["T_frame"]))
        v_frame = int(float(row["V_frame"]))

        frame_t = read_frame(thermal_video, t_frame)
        frame_v = read_frame(visible_video, v_frame)

        gray_t = normalize_gray(to_gray(frame_t))
        gray_v = normalize_gray(to_gray(frame_v))

        # Save raw extracted pair
        cv2.imwrite(str(extracted_dir / f"{stem}_T.png"), gray_t)
        cv2.imwrite(str(extracted_dir / f"{stem}_V_raw.png"), gray_v)
        save_side_by_side(gray_t, gray_v, extracted_dir / f"{stem}_side_by_side_raw.png")

        # Heuristic crop of visible image to reduce FOV mismatch
        crop_v = center_crop(gray_v, crop_ratio)
        crop_v_resized = cv2.resize(crop_v, (gray_t.shape[1], gray_t.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Edge images
        edges_t = build_edges(gray_t)
        edges_v = build_edges(crop_v_resized)

        cv2.imwrite(str(prep_dir / f"{stem}_V_cropped_resized.png"), crop_v_resized)
        cv2.imwrite(str(prep_dir / f"{stem}_T_edges.png"), edges_t)
        cv2.imwrite(str(prep_dir / f"{stem}_V_edges.png"), edges_v)
        save_side_by_side(gray_t, crop_v_resized, prep_dir / f"{stem}_side_by_side_prepared.png")
        save_side_by_side(edges_t, edges_v, prep_dir / f"{stem}_side_by_side_edges.png")

        # Before alignment overlay
        before_overlay = overlay(gray_t, crop_v_resized)
        cv2.imwrite(str(ecc_dir / f"{stem}_overlay_before.png"), before_overlay)

        # ECC on edge images, apply resulting warp to prepared visible grayscale
        success, warp_matrix, cc, warped_edges = run_ecc(edges_t, edges_v, motion_model_name=args.motion_model)

        if success:
            if args.motion_model == "homography":
                warped_visible = cv2.warpPerspective(
                    crop_v_resized, warp_matrix, (gray_t.shape[1], gray_t.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )
            else:
                warped_visible = cv2.warpAffine(
                    crop_v_resized, warp_matrix, (gray_t.shape[1], gray_t.shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
                )

            after_overlay = overlay(gray_t, warped_visible)
            cv2.imwrite(str(ecc_dir / f"{stem}_overlay_after.png"), after_overlay)
            cv2.imwrite(str(ecc_dir / f"{stem}_warped_visible.png"), warped_visible)
            cv2.imwrite(str(ecc_dir / f"{stem}_warped_edges.png"), warped_edges)
        else:
            warped_visible = None
            after_overlay = None

        summary_rows.append({
            "pair_id": stem,
            "T_frame": t_frame,
            "V_frame": v_frame,
            "crop_ratio_used": f"{crop_ratio:.3f}",
            "motion_model": args.motion_model,
            "ecc_success": "Yes" if success else "No",
            "ecc_correlation": f"{cc:.6f}" if cc is not None else "",
        })

    csv_out = outdir / "summary.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    txt_out = outdir / "summary.txt"
    with txt_out.open("w", encoding="utf-8") as f:
        f.write("Refined traditional CV from matched video frames\n")
        f.write("===============================================\n\n")
        f.write(f"Pairs processed: {len(chosen)}\n")
        f.write(f"Visible focal length: {args.visible_focal}\n")
        f.write(f"Thermal focal length: {args.thermal_focal}\n")
        f.write(f"Heuristic crop ratio used: {crop_ratio:.3f}\n")
        f.write(f"Motion model: {args.motion_model}\n\n")
        successes = sum(r["ecc_success"] == "Yes" for r in summary_rows)
        f.write(f"Successful ECC alignments: {successes}/{len(summary_rows)}\n")

    print("Done.")
    print(f"Pairs processed: {len(chosen)}")
    print(f"Extracted pairs: {extracted_dir}")
    print(f"Prepared pairs: {prep_dir}")
    print(f"ECC results: {ecc_dir}")
    print(f"CSV summary: {csv_out}")
    print(f"TXT summary: {txt_out}")


if __name__ == "__main__":
    main()
