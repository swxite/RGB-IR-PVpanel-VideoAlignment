#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

TESTS = [
    {
        "name": "1",
        "thermal_image": "1.jpeg",
        "recording_id": "0004",
        "visible_video": "DJI_20240829114550_0004_V.MP4",
        "thermal_frame": 527,
        "visible_frame_guess": 527,
        "thermal_time": "00:00:17,542 --> 00:00:17,580",
    },
    {
        "name": "2",
        "thermal_image": "2.jpeg",
        "recording_id": "0019",
        "visible_video": "DJI_20240621133005_0019_V.MP4",
        "thermal_frame": 276,
        "visible_frame_guess": 276,
        "thermal_time": "00:00:09,161 --> 00:00:09,194",
    },
    {
        "name": "3",
        "thermal_image": "3.jpeg",
        "recording_id": "0019",
        "visible_video": "DJI_20240621133005_0019_V.MP4",
        "thermal_frame": 1291,
        "visible_frame_guess": 1291,
        "thermal_time": "00:00:43,014 --> 00:00:43,048",
    },
]

def read_frame(video_path: Path, frame_index_1based: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index_1based - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index_1based} from {video_path}")
    return frame

def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

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

def draw_matches_big(img1, kp1, img2, kp2, matches, inlier_mask=None, max_draw=50):
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
    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for i, m in enumerate(draw_matches):
        p1 = tuple(int(v) for v in kp1[m.queryIdx].pt)
        p2_local = tuple(int(v) for v in kp2[m.trainIdx].pt)
        p2 = (p2_local[0] + w1, p2_local[1])
        color = (int((37*i) % 255), int((97*i + 50) % 255), int((173*i + 100) % 255))
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
    result = {"method":"SIFT","kp_t":len(kp_t) if kp_t is not None else 0,"kp_v":len(kp_v) if kp_v is not None else 0,
              "raw_matches":0,"good_matches":0,"inliers":0,"success":False,"reproj_error_px":None,"match_vis":None,"overlay":None}
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
    result["inliers"] = int(mask.ravel().sum())
    result["success"] = result["inliers"] >= 8
    result["reproj_error_px"] = compute_reprojection_error(pts_v, pts_t, H, mask)
    warped_v = cv2.warpPerspective(img_v, H, (img_t.shape[1], img_t.shape[0]))
    result["overlay"] = draw_overlay(img_t, warped_v)
    return result

def run_orb(img_t, img_v):
    orb = cv2.ORB_create(nfeatures=2000, fastThreshold=10)
    kp_t, des_t = orb.detectAndCompute(img_t, None)
    kp_v, des_v = orb.detectAndCompute(img_v, None)
    result = {"method":"ORB","kp_t":len(kp_t) if kp_t is not None else 0,"kp_v":len(kp_v) if kp_v is not None else 0,
              "raw_matches":0,"good_matches":0,"inliers":0,"success":False,"reproj_error_px":None,"match_vis":None,"overlay":None}
    if des_t is None or des_v is None or len(kp_t) < 4 or len(kp_v) < 4:
        result["match_vis"] = draw_matches_big(img_t, kp_t or [], img_v, kp_v or [], [])
        return result
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_t, des_v)
    matches = sorted(matches, key=lambda m: m.distance)
    result["raw_matches"] = len(matches)
    if len(matches) < 4:
        result["match_vis"] = draw_matches_big(img_t, kp_t, img_v, kp_v, matches)
        return result
    keep_n = max(15, min(100, int(0.35 * len(matches))))
    good = matches[:keep_n]
    result["good_matches"] = len(good)
    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good])
    pts_v = np.float32([kp_v[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pts_v, pts_t, cv2.RANSAC, 5.0)
    result["match_vis"] = draw_matches_big(img_t, kp_t, img_v, kp_v, good, inlier_mask=mask)
    if H is None or mask is None:
        return result
    result["inliers"] = int(mask.ravel().sum())
    result["success"] = result["inliers"] >= 8
    result["reproj_error_px"] = compute_reprojection_error(pts_v, pts_t, H, mask)
    warped_v = cv2.warpPerspective(img_v, H, (img_t.shape[1], img_t.shape[0]))
    result["overlay"] = draw_overlay(img_t, warped_v)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", required=True, help="Folder containing 1.jpeg 2.jpeg 3.jpeg and the videos")
    parser.add_argument("--visible_focal", type=float, default=24.0)
    parser.add_argument("--thermal_focal", type=float, default=40.0)
    parser.add_argument("--outdir", default="final_three_pair_results")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    outdir = Path(args.outdir)
    raw_dir = outdir / "raw_pairs"
    prep_dir = outdir / "prepared_pairs"
    orb_dir = outdir / "orb"
    sift_dir = outdir / "sift"
    for d in (outdir, raw_dir, prep_dir, orb_dir, sift_dir):
        d.mkdir(parents=True, exist_ok=True)

    crop_ratio = min(1.0, max(0.2, args.visible_focal / args.thermal_focal))
    summary_rows = []

    for item in TESTS:
        thermal_img_path = project_dir / item["thermal_image"]
        visible_video_path = project_dir / item["visible_video"]

        thermal_img_raw = load_gray(thermal_img_path)
        visible_frame_raw = read_frame(visible_video_path, item["visible_frame_guess"])
        visible_gray_raw = to_gray(visible_frame_raw)

        cv2.imwrite(str(raw_dir / f"{item['name']}_thermal_corrected_raw.png"), thermal_img_raw)
        cv2.imwrite(str(raw_dir / f"{item['name']}_visible_raw.png"), visible_gray_raw)
        save_side_by_side(thermal_img_raw, visible_gray_raw, raw_dir / f"{item['name']}_side_by_side_raw.png")

        thermal_img = preprocess_gray(thermal_img_raw)
        visible_crop = center_crop(visible_gray_raw, crop_ratio)
        visible_crop = cv2.resize(visible_crop, (thermal_img.shape[1], thermal_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        visible_img = preprocess_gray(visible_crop)

        cv2.imwrite(str(prep_dir / f"{item['name']}_thermal_prepared.png"), thermal_img)
        cv2.imwrite(str(prep_dir / f"{item['name']}_visible_prepared.png"), visible_img)
        save_side_by_side(thermal_img, visible_img, prep_dir / f"{item['name']}_side_by_side_prepared.png")

        orb_res = run_orb(thermal_img, visible_img)
        sift_res = run_sift(thermal_img, visible_img)

        if orb_res["match_vis"] is not None:
            cv2.imwrite(str(orb_dir / f"{item['name']}_matches.png"), orb_res["match_vis"])
        if orb_res["overlay"] is not None:
            cv2.imwrite(str(orb_dir / f"{item['name']}_overlay.png"), orb_res["overlay"])
        if sift_res["match_vis"] is not None:
            cv2.imwrite(str(sift_dir / f"{item['name']}_matches.png"), sift_res["match_vis"])
        if sift_res["overlay"] is not None:
            cv2.imwrite(str(sift_dir / f"{item['name']}_overlay.png"), sift_res["overlay"])

        for res in (orb_res, sift_res):
            summary_rows.append({
                "test_image": item["name"],
                "recording_id": item["recording_id"],
                "thermal_time": item["thermal_time"],
                "thermal_frame": item["thermal_frame"],
                "visible_frame_used": item["visible_frame_guess"],
                "method": res["method"],
                "crop_ratio_used": f"{crop_ratio:.3f}",
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
        f.write("Final three-pair ORB/SIFT test\n")
        f.write("==============================\n\n")
        f.write(f"Visible focal length: {args.visible_focal}\n")
        f.write(f"Thermal focal length: {args.thermal_focal}\n")
        f.write(f"Crop ratio used: {crop_ratio:.3f}\n\n")
        for method in ("ORB", "SIFT"):
            rows = [r for r in summary_rows if r["method"] == method]
            succ = sum(r["homography_success"] == "Yes" for r in rows)
            avg_good = sum(int(r["good_matches"]) for r in rows) / len(rows)
            avg_in = sum(int(r["inliers"]) for r in rows) / len(rows)
            f.write(f"{method}:\n")
            f.write(f"  Successful homographies: {succ}/{len(rows)}\n")
            f.write(f"  Mean good matches: {avg_good:.2f}\n")
            f.write(f"  Mean inliers: {avg_in:.2f}\n\n")

    print("Done.")
    print(f"CSV summary: {csv_out}")
    print(f"TXT summary: {txt_out}")
    print(f"Raw pairs: {raw_dir}")
    print(f"Prepared pairs: {prep_dir}")
    print(f"ORB results: {orb_dir}")
    print(f"SIFT results: {sift_dir}")

if __name__ == "__main__":
    main()
