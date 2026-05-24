#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def normalize_for_features(img):
    return cv2.equalizeHist(img)


def pair_files(input_dir: Path):
    t_files = sorted(input_dir.glob("*_T.png"))
    pairs = []
    for t_path in t_files:
        stem = t_path.stem[:-2]
        v_path = input_dir / f"{stem}_V.png"
        if v_path.exists():
            pairs.append((stem, t_path, v_path))
    return pairs


def draw_overlay(base_gray, warp_gray):
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    warp_bgr = cv2.cvtColor(warp_gray, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(base_bgr)
    overlay[:, :, 1] = base_bgr[:, :, 1]
    overlay[:, :, 0] = warp_bgr[:, :, 0]
    overlay[:, :, 2] = warp_bgr[:, :, 2]
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

    match_vis = cv2.drawMatches(
        img_t, kp_t, img_v, kp_v, good[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    result["match_vis"] = match_vis

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

    match_vis = cv2.drawMatches(
        img_t, kp_t, img_v, kp_v, good[:30], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    result["match_vis"] = match_vis

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


def save_result_images(method_dir: Path, stem: str, result):
    if result["match_vis"] is not None:
        cv2.imwrite(str(method_dir / f"{stem}_matches.png"), result["match_vis"])
    if result["overlay"] is not None:
        cv2.imwrite(str(method_dir / f"{stem}_overlay.png"), result["overlay"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing *_T.png and *_V.png pairs")
    parser.add_argument("--outdir", default="traditional_cv_results", help="Output folder")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    orb_dir = outdir / "orb"
    sift_dir = outdir / "sift"
    outdir.mkdir(parents=True, exist_ok=True)
    orb_dir.mkdir(exist_ok=True)
    sift_dir.mkdir(exist_ok=True)

    pairs = pair_files(input_dir)
    if not pairs:
        raise FileNotFoundError("No image pairs found. Expected files like pair_01_T.png and pair_01_V.png")

    summary_rows = []

    for stem, t_path, v_path in pairs:
        img_t = normalize_for_features(load_gray(t_path))
        img_v = normalize_for_features(load_gray(v_path))

        orb_res = run_orb(img_t, img_v)
        sift_res = run_sift(img_t, img_v)

        save_result_images(orb_dir, stem, orb_res)
        save_result_images(sift_dir, stem, sift_res)

        for res in (orb_res, sift_res):
            summary_rows.append({
                "pair_id": stem,
                "method": res["method"],
                "kp_thermal": res["kp_t"],
                "kp_visible": res["kp_v"],
                "raw_matches": res["raw_matches"],
                "good_matches": res["good_matches"],
                "inliers": res["inliers"],
                "homography_success": "Yes" if res["success"] else "No",
                "mean_reprojection_error_px": f"{res['reproj_error_px']:.3f}" if res["reproj_error_px"] is not None else "",
                "thermal_image": str(t_path),
                "visible_image": str(v_path),
            })

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    txt_path = outdir / "summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Traditional CV alignment summary\n")
        f.write("================================\n\n")
        f.write(f"Pairs processed: {len(pairs)}\n")
        for method in ("ORB", "SIFT"):
            rows = [r for r in summary_rows if r["method"] == method]
            successes = sum(r["homography_success"] == "Yes" for r in rows)
            f.write(f"\n{method}:\n")
            f.write(f"  Successful homographies: {successes}/{len(rows)}\n")
            if rows:
                avg_good = sum(int(r["good_matches"]) for r in rows) / len(rows)
                avg_inliers = sum(int(r["inliers"]) for r in rows) / len(rows)
                f.write(f"  Mean good matches: {avg_good:.2f}\n")
                f.write(f"  Mean inliers: {avg_inliers:.2f}\n")

    print("Done.")
    print(f"Pairs processed: {len(pairs)}")
    print(f"CSV summary: {csv_path}")
    print(f"TXT summary: {txt_path}")
    print(f"ORB results: {orb_dir}")
    print(f"SIFT results: {sift_dir}")


if __name__ == "__main__":
    main()
