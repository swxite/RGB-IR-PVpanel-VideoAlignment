#!/usr/bin/env python3
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


def choose_rows(rows, n):
    if not rows:
        return []
    if n >= len(rows):
        return rows
    if n <= 1:
        return [rows[0]]
    idxs = np.linspace(0, len(rows) - 1, n, dtype=int)
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
    h, w = img.shape[:2]
    cw = max(20, int(w * crop_ratio))
    ch = max(20, int(h * crop_ratio))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return img[y0:y0+ch, x0:x0+cw]


def preprocess_gray(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def draw_overlay(base_gray, warp_gray):
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    warp_bgr = cv2.cvtColor(warp_gray, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(base_bgr)
    overlay[:, :, 1] = base_bgr[:, :, 1]
    overlay[:, :, 0] = warp_bgr[:, :, 0]
    overlay[:, :, 2] = warp_bgr[:, :, 2]
    return overlay


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


def draw_matches_big(img1, kp1, img2, kp2, matches, inlier_mask=None, max_draw=40):
    draw_matches = matches[:max_draw]
    if inlier_mask is not None:
        filtered = []
        for m, keep in zip(matches, inlier_mask.ravel().tolist()):
            if keep:
                filtered.append(m)
        if filtered:
            draw_matches = filtered[:max_draw]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)

    if len(img1.shape) == 2:
        canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        canvas[:h1, :w1] = img1
    if len(img2.shape) == 2:
        canvas[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        canvas[:h2, w1:w1+w2] = img2

    for i, m in enumerate(draw_matches):
        p1 = tuple(int(v) for v in kp1[m.queryIdx].pt)
        p2_local = tuple(int(v) for v in kp2[m.trainIdx].pt)
        p2 = (p2_local[0] + w1, p2_local[1])
        color = (
            int((37 * i) % 255),
            int((97 * i + 50) % 255),
            int((173 * i + 100) % 255),
        )
        cv2.line(canvas, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 6, color, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, 6, color, thickness=2, lineType=cv2.LINE_AA)
    return canvas


def run_sift(img_t, img_v):
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("This OpenCV build does not include SIFT_create().")

    sift = cv2.SIFT_create(nfeatures=1500)
    kp_t, des_t = sift.detectAndCompute(img_t, None)
    kp_v, des_v = sift.detectAndCompute(img_v, None)

    result = {
        "kp_t": len(kp_t) if kp_t is not None else 0,
        "kp_v": len(kp_v) if kp_v is not None else 0,
        "raw_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "success": False,
        "reproj_error_px": None,
        "match_vis": None,
        "overlay": None,
        "warped_visible": None,
    }

    if des_t is None or des_v is None or len(kp_t) < 4 or len(kp_v) < 4:
        result["match_vis"] = draw_matches_big(img_t, kp_t or [], img_v, kp_v or [], [])
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

    if len(good) < 4:
        result["match_vis"] = draw_matches_big(img_t, kp_t, img_v, kp_v, good)
        return result

    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good])
    pts_v = np.float32([kp_v[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_v, pts_t, cv2.RANSAC, 5.0)
    result["match_vis"] = draw_matches_big(img_t, kp_t, img_v, kp_v, good, inlier_mask=mask)

    if H is None or mask is None:
        return result

    inliers = int(mask.ravel().sum())
    result["inliers"] = inliers
    result["success"] = inliers >= 8
    result["reproj_error_px"] = compute_reprojection_error(pts_v, pts_t, H, mask)

    warped_v = cv2.warpPerspective(img_v, H, (img_t.shape[1], img_t.shape[0]))
    result["warped_visible"] = warped_v
    result["overlay"] = draw_overlay(img_t, warped_v)
    return result


def find_recording_files(input_dir: Path, short_id: str):
    csv_candidates = sorted(input_dir.rglob(f"*_{short_id}_matched.csv"))
    t_candidates = sorted(input_dir.rglob(f"*_{short_id}_T.MP4"))
    v_candidates = sorted(input_dir.rglob(f"*_{short_id}_V.MP4"))
    if not csv_candidates:
        raise FileNotFoundError(f"Could not find matched CSV for recording {short_id}")
    if not t_candidates:
        raise FileNotFoundError(f"Could not find thermal video for recording {short_id}")
    if not v_candidates:
        raise FileNotFoundError(f"Could not find visible video for recording {short_id}")
    return csv_candidates[0], t_candidates[0], v_candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Project folder containing videos and *_matched.csv files")
    parser.add_argument("--recordings", nargs="+", required=True, help="Recording short IDs, e.g. 0019 0020 0002 0004")
    parser.add_argument("--pairs_per_recording", type=int, default=10)
    parser.add_argument("--visible_focal", type=float, default=24.0)
    parser.add_argument("--thermal_focal", type=float, default=40.0)
    parser.add_argument("--outdir", default="sift_batch_results")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    extracted_dir = outdir / "extracted_pairs"
    prepared_dir = outdir / "prepared_pairs"
    sift_dir = outdir / "sift"
    for d in (outdir, extracted_dir, prepared_dir, sift_dir):
        d.mkdir(parents=True, exist_ok=True)

    crop_ratio = min(1.0, max(0.2, args.visible_focal / args.thermal_focal))
    summary_rows = []

    for rec_id in args.recordings:
        csv_path, thermal_video, visible_video = find_recording_files(input_dir, rec_id)
        rows = read_csv_rows(csv_path)
        chosen = choose_rows(rows, args.pairs_per_recording)

        for i, row in enumerate(chosen, start=1):
            pair_name = f"{rec_id}_pair_{i:02d}"
            t_frame = int(float(row["T_frame"]))
            v_frame = int(float(row["V_frame"]))

            frame_t = read_frame(thermal_video, t_frame)
            frame_v = read_frame(visible_video, v_frame)

            gray_t_raw = to_gray(frame_t)
            gray_v_raw = to_gray(frame_v)

            cv2.imwrite(str(extracted_dir / f"{pair_name}_T_raw.png"), gray_t_raw)
            cv2.imwrite(str(extracted_dir / f"{pair_name}_V_raw.png"), gray_v_raw)
            save_side_by_side(gray_t_raw, gray_v_raw, extracted_dir / f"{pair_name}_side_by_side_raw.png")

            gray_t = preprocess_gray(gray_t_raw)
            gray_v_crop = center_crop(gray_v_raw, crop_ratio)
            gray_v_crop = cv2.resize(gray_v_crop, (gray_t.shape[1], gray_t.shape[0]), interpolation=cv2.INTER_LINEAR)
            gray_v = preprocess_gray(gray_v_crop)

            cv2.imwrite(str(prepared_dir / f"{pair_name}_T_prepared.png"), gray_t)
            cv2.imwrite(str(prepared_dir / f"{pair_name}_V_prepared.png"), gray_v)
            save_side_by_side(gray_t, gray_v, prepared_dir / f"{pair_name}_side_by_side_prepared.png")

            res = run_sift(gray_t, gray_v)

            if res["match_vis"] is not None:
                cv2.imwrite(str(sift_dir / f"{pair_name}_matches.png"), res["match_vis"])
            if res["overlay"] is not None:
                cv2.imwrite(str(sift_dir / f"{pair_name}_overlay.png"), res["overlay"])
            if res["warped_visible"] is not None:
                cv2.imwrite(str(sift_dir / f"{pair_name}_warped_visible.png"), res["warped_visible"])

            summary_rows.append({
                "recording_id": rec_id,
                "pair_id": pair_name,
                "T_frame": t_frame,
                "V_frame": v_frame,
                "crop_ratio_used": f"{crop_ratio:.3f}",
                "kp_thermal": res["kp_t"],
                "kp_visible": res["kp_v"],
                "raw_matches": res["raw_matches"],
                "good_matches": res["good_matches"],
                "inliers": res["inliers"],
                "homography_success": "Yes" if res["success"] else "No",
                "mean_reprojection_error_px": f"{res['reproj_error_px']:.3f}" if res["reproj_error_px"] is not None else "",
                "matched_csv": str(csv_path),
                "thermal_video": str(thermal_video),
                "visible_video": str(visible_video),
            })

    if not summary_rows:
        raise RuntimeError("No results were produced.")

    csv_out = outdir / "summary.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    txt_out = outdir / "summary.txt"
    with txt_out.open("w", encoding="utf-8") as f:
        f.write("Batch SIFT alignment from matched video frames\n")
        f.write("=============================================\n\n")
        f.write(f"Recordings: {' '.join(args.recordings)}\n")
        f.write(f"Pairs per recording requested: {args.pairs_per_recording}\n")
        f.write(f"Visible focal length: {args.visible_focal}\n")
        f.write(f"Thermal focal length: {args.thermal_focal}\n")
        f.write(f"Crop ratio used: {crop_ratio:.3f}\n")
        f.write(f"Total tries: {len(summary_rows)}\n\n")

        successes = sum(r["homography_success"] == "Yes" for r in summary_rows)
        avg_good = sum(int(r["good_matches"]) for r in summary_rows) / len(summary_rows)
        avg_inliers = sum(int(r["inliers"]) for r in summary_rows) / len(summary_rows)
        f.write("Overall SIFT results:\n")
        f.write(f"  Successful homographies: {successes}/{len(summary_rows)}\n")
        f.write(f"  Mean good matches: {avg_good:.2f}\n")
        f.write(f"  Mean inliers: {avg_inliers:.2f}\n\n")

        for rec_id in args.recordings:
            rows_rec = [r for r in summary_rows if r["recording_id"] == rec_id]
            if not rows_rec:
                continue
            succ = sum(r["homography_success"] == "Yes" for r in rows_rec)
            avg_good_rec = sum(int(r["good_matches"]) for r in rows_rec) / len(rows_rec)
            avg_in_rec = sum(int(r["inliers"]) for r in rows_rec) / len(rows_rec)
            f.write(f"{rec_id}:\n")
            f.write(f"  Successful homographies: {succ}/{len(rows_rec)}\n")
            f.write(f"  Mean good matches: {avg_good_rec:.2f}\n")
            f.write(f"  Mean inliers: {avg_in_rec:.2f}\n\n")

    print("Done.")
    print(f"Total tries: {len(summary_rows)}")
    print(f"CSV summary: {csv_out}")
    print(f"TXT summary: {txt_out}")
    print(f"Extracted pairs: {extracted_dir}")
    print(f"Prepared pairs: {prepared_dir}")
    print(f"SIFT results: {sift_dir}")


if __name__ == "__main__":
    main()
