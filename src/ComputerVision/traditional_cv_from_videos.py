#!/usr/bin/env python3
"""
traditional_cv_from_videos.py

Full pipeline for traditional CV on DJI thermal/visible video pairs.

What it does:
1) Reads a matched-frames CSV (from your metadata step)
2) Extracts matched thermal and visible frames from the videos
3) Runs ORB + RANSAC homography
4) Runs SIFT + RANSAC homography
5) Saves extracted frame pairs, match visualizations, overlays, and summary CSV/TXT

Expected inputs:
- matched CSV with at least these columns:
    T_frame, V_frame
- thermal video file
- visible video file

Example:
python traditional_cv_from_videos.py ^
  --csv matched_frames.csv ^
  --thermal_video DJI_20240621133005_0019_T.MP4 ^
  --visible_video DJI_20240621133005_0019_V.MP4 ^
  --num_pairs 5 ^
  --outdir traditional_cv_results
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

    # Spread picks across the sequence
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


def normalize_for_features(img_gray):
    return cv2.equalizeHist(img_gray)


def draw_overlay(base_gray, warp_gray):
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    warp_bgr = cv2.cvtColor(warp_gray, cv2.COLOR_GRAY2BGR)

    overlay = np.zeros_like(base_bgr)
    overlay[:, :, 1] = base_bgr[:, :, 1]   # green channel from base
    overlay[:, :, 0] = warp_bgr[:, :, 0]   # blue from warped
    overlay[:, :, 2] = warp_bgr[:, :, 2]   # red from warped
    return overlay


def compute_reprojection_error(pts_src, pts_dst, H, inlier_mask):
    if H is None or inlier_mask is None:
        return None
    inlier_mask = inlier_mask.ravel().astype(bool)
    if inlier_mask.sum() == 0:
        return None

    src_in = pts_src[inlier_mask].reshape(-1, 1, 2)
    dst_in = pts_dst[inlier_mask].reshape(-1, 2)
    proj = cv2.perspectiveTransform(src_in, H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst_in, axis=1)
    return float(np.mean(err))


def run_orb(img_t, img_v):
    orb = cv2.ORB_create(nfeatures=1500)
    kp_t, des_t = orb.detectAndCompute(img_t, None)
    kp_v, des_v = orb.detectAndCompute(img_v, None)

    result = {
        "method": "ORB",
        "kp_t": len(kp_t) if kp_t is not None else 0,
        "kp_v": len(kp_v) if kp_v is not None else 0,
        "raw_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "success": False,
        "reproj_error_px": None,
        "match_vis": None,
        "overlay": None,
    }

    if des_t is None or des_v is None or len(kp_t) < 4 or len(kp_v) < 4:
        return result

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_t, des_v)
    matches = sorted(matches, key=lambda m: m.distance)
    result["raw_matches"] = len(matches)

    if len(matches) < 4:
        return result

    keep_n = max(10, min(80, int(0.35 * len(matches))))
    good = matches[:keep_n]
    result["good_matches"] = len(good)

    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good])
    pts_v = np.float32([kp_v[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_v, pts_t, cv2.RANSAC, 5.0)

    result["match_vis"] = cv2.drawMatches(
        img_t, kp_t, img_v, kp_v, good[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if H is None or mask is None:
        return result

    inliers = int(mask.ravel().sum())
    result["inliers"] = inliers
    result["success"] = inliers >= 8
    result["reproj_error_px"] = compute_reprojection_error(pts_v, pts_t, H, mask)

    warped_v = cv2.warpPerspective(img_v, H, (img_t.shape[1], img_t.shape[0]))
    result["overlay"] = draw_overlay(img_t, warped_v)
    return result


def run_sift(img_t, img_v):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not include SIFT_create().")

    sift = cv2.SIFT_create()
    kp_t, des_t = sift.detectAndCompute(img_t, None)
    kp_v, des_v = sift.detectAndCompute(img_v, None)

    result = {
        "method": "SIFT",
        "kp_t": len(kp_t) if kp_t is not None else 0,
        "kp_v": len(kp_v) if kp_v is not None else 0,
        "raw_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "success": False,
        "reproj_error_px": None,
        "match_vis": None,
        "overlay": None,
    }

    if des_t is None or des_v is None or len(kp_t) < 4 or len(kp_v) < 4:
        return result

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_t, des_v, k=2)
    result["raw_matches"] = len(knn)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    result["good_matches"] = len(good)

    result["match_vis"] = cv2.drawMatches(
        img_t, kp_t, img_v, kp_v, good[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if len(good) < 4:
        return result

    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good])
    pts_v = np.float32([kp_v[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_v, pts_t, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return result

    inliers = int(mask.ravel().sum())
    result["inliers"] = inliers
    result["success"] = inliers >= 8
    result["reproj_error_px"] = compute_reprojection_error(pts_v, pts_t, H, mask)

    warped_v = cv2.warpPerspective(img_v, H, (img_t.shape[1], img_t.shape[0]))
    result["overlay"] = draw_overlay(img_t, warped_v)
    return result


def save_side_by_side(gray_t, gray_v, outpath: Path):
    h = max(gray_t.shape[0], gray_v.shape[0])
    if gray_t.shape[0] != h:
        scale = h / gray_t.shape[0]
        gray_t = cv2.resize(gray_t, (int(gray_t.shape[1] * scale), h))
    if gray_v.shape[0] != h:
        scale = h / gray_v.shape[0]
        gray_v = cv2.resize(gray_v, (int(gray_v.shape[1] * scale), h))
    canvas = np.hstack([cv2.cvtColor(gray_t, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(gray_v, cv2.COLOR_GRAY2BGR)])
    cv2.imwrite(str(outpath), canvas)


def save_result_images(method_dir: Path, stem: str, result):
    if result["match_vis"] is not None:
        cv2.imwrite(str(method_dir / f"{stem}_matches.png"), result["match_vis"])
    if result["overlay"] is not None:
        cv2.imwrite(str(method_dir / f"{stem}_overlay.png"), result["overlay"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Matched frames CSV from metadata step")
    parser.add_argument("--thermal_video", required=True)
    parser.add_argument("--visible_video", required=True)
    parser.add_argument("--num_pairs", type=int, default=5, help="How many matched pairs to process")
    parser.add_argument("--outdir", default="traditional_cv_results")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    thermal_video = Path(args.thermal_video)
    visible_video = Path(args.visible_video)
    outdir = Path(args.outdir)

    extracted_dir = outdir / "extracted_pairs"
    orb_dir = outdir / "orb"
    sift_dir = outdir / "sift"
    outdir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(exist_ok=True)
    orb_dir.mkdir(exist_ok=True)
    sift_dir.mkdir(exist_ok=True)

    rows = read_csv_rows(csv_path)
    chosen = choose_rows(rows, args.num_pairs)

    if not chosen:
        raise RuntimeError("No rows found in matched CSV.")

    summary_rows = []

    for i, row in enumerate(chosen, start=1):
        t_frame = int(float(row["T_frame"]))
        v_frame = int(float(row["V_frame"]))
        stem = f"pair_{i:02d}"

        frame_t = read_frame(thermal_video, t_frame)
        frame_v = read_frame(visible_video, v_frame)

        gray_t = normalize_for_features(to_gray(frame_t))
        gray_v = normalize_for_features(to_gray(frame_v))

        cv2.imwrite(str(extracted_dir / f"{stem}_T.png"), gray_t)
        cv2.imwrite(str(extracted_dir / f"{stem}_V.png"), gray_v)
        save_side_by_side(gray_t, gray_v, extracted_dir / f"{stem}_side_by_side.png")

        orb_res = run_orb(gray_t, gray_v)
        sift_res = run_sift(gray_t, gray_v)

        save_result_images(orb_dir, stem, orb_res)
        save_result_images(sift_dir, stem, sift_res)

        for res in (orb_res, sift_res):
            summary_rows.append({
                "pair_id": stem,
                "T_frame": t_frame,
                "V_frame": v_frame,
                "method": res["method"],
                "kp_thermal": res["kp_t"],
                "kp_visible": res["kp_v"],
                "raw_matches": res["raw_matches"],
                "good_matches": res["good_matches"],
                "inliers": res["inliers"],
                "homography_success": "Yes" if res["success"] else "No",
                "mean_reprojection_error_px": f"{res['reproj_error_px']:.3f}" if res["reproj_error_px"] is not None else "",
            })

    csv_out = outdir / "summary.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    txt_out = outdir / "summary.txt"
    with txt_out.open("w", encoding="utf-8") as f:
        f.write("Traditional CV from matched video frames\n")
        f.write("=======================================\n\n")
        f.write(f"Pairs processed: {len(chosen)}\n")
        for method in ("ORB", "SIFT"):
            rows_m = [r for r in summary_rows if r["method"] == method]
            successes = sum(r["homography_success"] == "Yes" for r in rows_m)
            f.write(f"\n{method}:\n")
            f.write(f"  Successful homographies: {successes}/{len(rows_m)}\n")
            if rows_m:
                avg_good = sum(int(r["good_matches"]) for r in rows_m) / len(rows_m)
                avg_inliers = sum(int(r["inliers"]) for r in rows_m) / len(rows_m)
                f.write(f"  Mean good matches: {avg_good:.2f}\n")
                f.write(f"  Mean inliers: {avg_inliers:.2f}\n")

    print("Done.")
    print(f"Pairs processed: {len(chosen)}")
    print(f"Extracted images: {extracted_dir}")
    print(f"CSV summary: {csv_out}")
    print(f"TXT summary: {txt_out}")
    print(f"ORB results: {orb_dir}")
    print(f"SIFT results: {sift_dir}")


if __name__ == "__main__":
    main()
