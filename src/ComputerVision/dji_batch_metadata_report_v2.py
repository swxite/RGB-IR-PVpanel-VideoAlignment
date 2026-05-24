#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"FrameCnt:\s*(?P<frame>\d+),\s*DiffTime:\s*(?P<diff>\d+)ms.*?"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*?"
    r"\[focal_len:\s*(?P<focal>-?\d+(?:\.\d+)?)\].*?"
    r"\[dzoom_ratio:\s*(?P<zoom>-?\d+(?:\.\d+)?)\],\s*"
    r"\[latitude:\s*(?P<lat>-?\d+(?:\.\d+)?)\]\s*"
    r"\[longitude:\s*(?P<lon>-?\d+(?:\.\d+)?)\]\s*"
    r"\[rel_alt:\s*(?P<rel_alt>-?\d+(?:\.\d+)?)\s*abs_alt:\s*(?P<abs_alt>-?\d+(?:\.\d+)?)\]\s*"
    r"\[gb_yaw:\s*(?P<yaw>-?\d+(?:\.\d+)?)\s*gb_pitch:\s*(?P<pitch>-?\d+(?:\.\d+)?)\s*gb_roll:\s*(?P<roll>-?\d+(?:\.\d+)?)\]",
    re.DOTALL
)
NAME_RE = re.compile(r"^(DJI_\d{14}_\d{4})_([STV])(?:\.(?:SRT|srt))?$")

def parse_srt(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows = []
    for m in LINE_RE.finditer(text):
        rows.append({
            "frame": int(m.group("frame")),
            "diff_ms": int(m.group("diff")),
            "ts": datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S.%f"),
            "focal_len": float(m.group("focal")),
            "zoom": float(m.group("zoom")),
            "lat": float(m.group("lat")),
            "lon": float(m.group("lon")),
            "rel_alt": float(m.group("rel_alt")),
            "abs_alt": float(m.group("abs_alt")),
            "yaw": float(m.group("yaw")),
            "pitch": float(m.group("pitch")),
            "roll": float(m.group("roll")),
        })
    return rows

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def nearest_visible_matches(t_rows, v_rows):
    matches = []
    j = 0
    for tr in t_rows:
        while j + 1 < len(v_rows) and abs((v_rows[j + 1]["ts"] - tr["ts"]).total_seconds()) <= abs((v_rows[j]["ts"] - tr["ts"]).total_seconds()):
            j += 1
        vr = v_rows[j]
        dt_ms = abs((vr["ts"] - tr["ts"]).total_seconds()) * 1000.0
        gps_m = haversine_m(tr["lat"], tr["lon"], vr["lat"], vr["lon"])
        matches.append({
            "T_frame": tr["frame"],
            "T_time": tr["ts"].isoformat(sep=" "),
            "V_frame": vr["frame"],
            "V_time": vr["ts"].isoformat(sep=" "),
            "time_diff_ms": dt_ms,
            "gps_distance_m": gps_m,
            "yaw_diff_deg": abs(tr["yaw"] - vr["yaw"]),
            "pitch_diff_deg": abs(tr["pitch"] - vr["pitch"]),
            "roll_diff_deg": abs(tr["roll"] - vr["roll"]),
            "T_focal_len": tr["focal_len"],
            "V_focal_len": vr["focal_len"],
        })
    return matches

def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")

def find_srt_files(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".srt"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pair_dir = outdir / "pair_match_tables"
    pair_dir.mkdir(exist_ok=True)

    groups = defaultdict(dict)
    for p in find_srt_files(input_dir):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        rec_id, stream = m.group(1), m.group(2)
        groups[rec_id][stream] = p

    summary_rows = []
    for rec_id, streams in sorted(groups.items()):
        if "T" not in streams or "V" not in streams:
            continue
        t_rows = parse_srt(streams["T"])
        v_rows = parse_srt(streams["V"])
        if not t_rows or not v_rows:
            continue

        matches = nearest_visible_matches(t_rows, v_rows)
        if not matches:
            continue

        csv_path = pair_dir / f"{rec_id}_matched.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(matches[0].keys()))
            w.writeheader()
            w.writerows(matches)

        dt_list = [m["time_diff_ms"] for m in matches]
        gps_list = [m["gps_distance_m"] for m in matches]
        yaw_list = [m["yaw_diff_deg"] for m in matches]
        pitch_list = [m["pitch_diff_deg"] for m in matches]
        roll_list = [m["roll_diff_deg"] for m in matches]

        summary_rows.append({
            "recording_id": rec_id,
            "thermal_frames": len(t_rows),
            "visible_frames": len(v_rows),
            "thermal_start": t_rows[0]["ts"].isoformat(sep=" "),
            "visible_start": v_rows[0]["ts"].isoformat(sep=" "),
            "start_offset_ms": (t_rows[0]["ts"] - v_rows[0]["ts"]).total_seconds() * 1000.0,
            "thermal_duration_s": (t_rows[-1]["ts"] - t_rows[0]["ts"]).total_seconds(),
            "visible_duration_s": (v_rows[-1]["ts"] - v_rows[0]["ts"]).total_seconds(),
            "thermal_focal_len": t_rows[0]["focal_len"],
            "visible_focal_len": v_rows[0]["focal_len"],
            "mean_time_diff_ms": mean(dt_list),
            "max_time_diff_ms": max(dt_list),
            "mean_gps_distance_m": mean(gps_list),
            "mean_yaw_diff_deg": mean(yaw_list),
            "mean_pitch_diff_deg": mean(pitch_list),
            "mean_roll_diff_deg": mean(roll_list),
            "thermal_srt": str(streams["T"]),
            "visible_srt": str(streams["V"]),
        })

    summary_csv = outdir / "all_recordings_summary.csv"
    batch_txt = outdir / "batch_summary.txt"

    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)

        recs = [r["recording_id"] for r in summary_rows]

        def save_bar(values, title, ylabel, fname):
            plt.figure(figsize=(10, 4.8))
            plt.bar(range(len(values)), values)
            plt.xticks(range(len(values)), recs, rotation=45, ha="right")
            plt.title(title)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(outdir / fname, dpi=180)
            plt.close()

        save_bar([r["start_offset_ms"] for r in summary_rows], "Start offset between thermal and visible streams", "ms", "start_offsets.png")
        save_bar([r["mean_time_diff_ms"] for r in summary_rows], "Mean nearest-frame time difference", "ms", "mean_time_diff.png")
        save_bar([r["mean_gps_distance_m"] for r in summary_rows], "Mean GPS distance of matched pairs", "m", "mean_gps_distance.png")

        plt.figure(figsize=(8, 5))
        x = [r["visible_focal_len"] for r in summary_rows]
        y = [r["thermal_focal_len"] for r in summary_rows]
        plt.scatter(x, y)
        for r, xv, yv in zip(summary_rows, x, y):
            plt.annotate(r["recording_id"].split("_")[-1], (xv, yv), fontsize=8)
        plt.xlabel("Visible focal length")
        plt.ylabel("Thermal focal length")
        plt.title("Focal lengths across recordings")
        plt.tight_layout()
        plt.savefig(outdir / "focal_lengths_scatter.png", dpi=180)
        plt.close()

        with batch_txt.open("w", encoding="utf-8") as f:
            f.write("DJI batch metadata summary\n")
            f.write("==========================\n\n")
            f.write(f"Recordings with both thermal and visible metadata: {len(summary_rows)}\n")
            f.write(f"Mean start offset (thermal-visible): {mean([r['start_offset_ms'] for r in summary_rows]):.3f} ms\n")
            f.write(f"Mean nearest-frame time difference: {mean([r['mean_time_diff_ms'] for r in summary_rows]):.3f} ms\n")
            f.write(f"Mean GPS distance: {mean([r['mean_gps_distance_m'] for r in summary_rows]):.6f} m\n")
            f.write(f"Mean yaw difference: {mean([r['mean_yaw_diff_deg'] for r in summary_rows]):.6f} deg\n")
            f.write(f"Mean pitch difference: {mean([r['mean_pitch_diff_deg'] for r in summary_rows]):.6f} deg\n")
            f.write(f"Mean roll difference: {mean([r['mean_roll_diff_deg'] for r in summary_rows]):.6f} deg\n")
    else:
        batch_txt.write_text(
            "No T/V recording pairs were found.\n"
            "Check that your files are named like DJI_YYYYMMDDhhmmss_####_T.SRT and ..._V.SRT.\n",
            encoding="utf-8"
        )

    print(f"Done.\nOutput folder: {outdir}\nSummaries found: {len(summary_rows)}")

if __name__ == "__main__":
    main()
