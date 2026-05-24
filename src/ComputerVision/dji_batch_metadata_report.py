#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


BLOCK_RE = re.compile(
    r"(?P<idx>\d+)\s+"
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})\s+"
    r"<font[^>]*>FrameCnt:\s*(?P<framecnt>\d+),\s*DiffTime:\s*(?P<difftime>\d+)ms\s+"
    r"(?P<abs_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"\[focal_len:\s*(?P<focal>[-\d.]+)\]\s*\[dzoom_ratio:\s*(?P<zoom>[-\d.]+)\],\s*"
    r"\[latitude:\s*(?P<lat>[-\d.]+)\]\s*\[longitude:\s*(?P<lon>[-\d.]+)\]\s*"
    r"\[rel_alt:\s*(?P<rel_alt>[-\d.]+)\s*abs_alt:\s*(?P<abs_alt>[-\d.]+)\]\s*"
    r"\[gb_yaw:\s*(?P<yaw>[-\d.]+)\s*gb_pitch:\s*(?P<pitch>[-\d.]+)\s*gb_roll:\s*(?P<roll>[-\d.]+)\]\s*</font>",
    re.MULTILINE,
)

NAME_RE = re.compile(r"^(DJI_\d{14}_\d{4})_([STV])\.SRT$", re.IGNORECASE)


@dataclass
class FrameMeta:
    frame: int
    diff_ms: int
    timestamp: datetime
    focal_len: float
    dzoom_ratio: float
    latitude: float
    longitude: float
    rel_alt: float
    abs_alt: float
    gb_yaw: float
    gb_pitch: float
    gb_roll: float


@dataclass
class PairSummary:
    recording_id: str
    thermal_srt: str
    visible_srt: str
    thermal_frames: int
    visible_frames: int
    thermal_start: str
    visible_start: str
    start_offset_ms: float
    thermal_duration_s: float
    visible_duration_s: float
    thermal_focal_len: float
    visible_focal_len: float
    mean_time_diff_ms: float
    median_time_diff_ms: float
    min_time_diff_ms: float
    max_time_diff_ms: float
    mean_gps_distance_m: float
    median_gps_distance_m: float
    yaw_mean_deg: float
    pitch_mean_deg: float
    roll_mean_deg: float


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def parse_srt(path: Path) -> List[FrameMeta]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    frames: List[FrameMeta] = []
    for m in BLOCK_RE.finditer(text):
        frames.append(
            FrameMeta(
                frame=int(m.group("framecnt")),
                diff_ms=int(m.group("difftime")),
                timestamp=datetime.strptime(m.group("abs_time"), "%Y-%m-%d %H:%M:%S.%f"),
                focal_len=float(m.group("focal")),
                dzoom_ratio=float(m.group("zoom")),
                latitude=float(m.group("lat")),
                longitude=float(m.group("lon")),
                rel_alt=float(m.group("rel_alt")),
                abs_alt=float(m.group("abs_alt")),
                gb_yaw=float(m.group("yaw")),
                gb_pitch=float(m.group("pitch")),
                gb_roll=float(m.group("roll")),
            )
        )
    return frames


def nearest_match_summaries(thermal: List[FrameMeta], visible: List[FrameMeta]) -> Tuple[List[dict], PairSummary]:
    visible_ts = [v.timestamp for v in visible]
    matches: List[dict] = []
    j = 0
    for t in thermal:
        while j + 1 < len(visible) and abs((visible[j + 1].timestamp - t.timestamp).total_seconds()) <= abs((visible[j].timestamp - t.timestamp).total_seconds()):
            j += 1
        v = visible[j]
        dt_ms = abs((v.timestamp - t.timestamp).total_seconds()) * 1000.0
        gps_m = haversine_m(t.latitude, t.longitude, v.latitude, v.longitude)
        matches.append({
            "T_frame": t.frame,
            "T_time": t.timestamp.isoformat(sep=" ", timespec="milliseconds"),
            "V_frame": v.frame,
            "V_time": v.timestamp.isoformat(sep=" ", timespec="milliseconds"),
            "time_diff_ms": dt_ms,
            "gps_distance_m": gps_m,
            "yaw_diff_deg": abs(t.gb_yaw - v.gb_yaw),
            "pitch_diff_deg": abs(t.gb_pitch - v.gb_pitch),
            "roll_diff_deg": abs(t.gb_roll - v.gb_roll),
            "T_focal_len": t.focal_len,
            "V_focal_len": v.focal_len,
            "T_lat": t.latitude,
            "T_lon": t.longitude,
            "V_lat": v.latitude,
            "V_lon": v.longitude,
        })

    time_diffs = [m["time_diff_ms"] for m in matches]
    gps = [m["gps_distance_m"] for m in matches]
    yaw = [m["yaw_diff_deg"] for m in matches]
    pitch = [m["pitch_diff_deg"] for m in matches]
    roll = [m["roll_diff_deg"] for m in matches]

    pair = PairSummary(
        recording_id="",
        thermal_srt="",
        visible_srt="",
        thermal_frames=len(thermal),
        visible_frames=len(visible),
        thermal_start=thermal[0].timestamp.isoformat(sep=" ", timespec="milliseconds"),
        visible_start=visible[0].timestamp.isoformat(sep=" ", timespec="milliseconds"),
        start_offset_ms=(thermal[0].timestamp - visible[0].timestamp).total_seconds() * 1000.0,
        thermal_duration_s=(thermal[-1].timestamp - thermal[0].timestamp).total_seconds(),
        visible_duration_s=(visible[-1].timestamp - visible[0].timestamp).total_seconds(),
        thermal_focal_len=thermal[0].focal_len,
        visible_focal_len=visible[0].focal_len,
        mean_time_diff_ms=mean(time_diffs),
        median_time_diff_ms=median(time_diffs),
        min_time_diff_ms=min(time_diffs),
        max_time_diff_ms=max(time_diffs),
        mean_gps_distance_m=mean(gps),
        median_gps_distance_m=median(gps),
        yaw_mean_deg=mean(yaw),
        pitch_mean_deg=mean(pitch),
        roll_mean_deg=mean(roll),
    )
    return matches, pair


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_txt(path: Path, pairs: List[PairSummary]) -> None:
    lines = ["DJI batch metadata summary", "==========================", ""]
    lines.append(f"Recordings analysed: {len(pairs)}")
    if pairs:
        lines.extend([
            f"Mean start offset (T-V): {mean(p.start_offset_ms for p in pairs):.3f} ms",
            f"Mean nearest-frame time diff: {mean(p.mean_time_diff_ms for p in pairs):.3f} ms",
            f"Mean GPS distance: {mean(p.mean_gps_distance_m for p in pairs):.6f} m",
            f"Mean yaw diff: {mean(p.yaw_mean_deg for p in pairs):.6f} deg",
            "",
            "Per recording:",
        ])
        for p in pairs:
            lines.append(
                f"- {p.recording_id}: start offset={p.start_offset_ms:.1f} ms, mean dt={p.mean_time_diff_ms:.2f} ms, mean GPS={p.mean_gps_distance_m:.5f} m"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_bar(values: List[float], labels: List[str], title: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_scatter(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def find_groups(folder: Path) -> Dict[str, Dict[str, Path]]:
    groups: Dict[str, Dict[str, Path]] = {}
    for p in folder.glob("*.SRT"):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        rec_id, mod = m.groups()
        groups.setdefault(rec_id, {})[mod.upper()] = p
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse all DJI SRT files in a folder and create metadata summaries/plots.")
    ap.add_argument("--input_dir", required=True, help="Folder containing DJI .SRT files")
    ap.add_argument("--outdir", default="dji_batch_results", help="Output folder")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    matches_dir = outdir / "pair_match_tables"
    matches_dir.mkdir(exist_ok=True)

    groups = find_groups(input_dir)
    pair_summaries: List[PairSummary] = []

    for rec_id, mods in sorted(groups.items()):
        if "T" not in mods or "V" not in mods:
            continue
        thermal = parse_srt(mods["T"])
        visible = parse_srt(mods["V"])
        if not thermal or not visible:
            continue
        matches, pair = nearest_match_summaries(thermal, visible)
        pair.recording_id = rec_id
        pair.thermal_srt = mods["T"].name
        pair.visible_srt = mods["V"].name
        pair_summaries.append(pair)
        write_csv(matches_dir / f"{rec_id}_matched_frames.csv", matches)

    pair_rows = [asdict(p) for p in pair_summaries]
    write_csv(outdir / "all_recordings_summary.csv", pair_rows)
    save_summary_txt(outdir / "batch_summary.txt", pair_summaries)

    if pair_summaries:
        labels = [p.recording_id.split("_")[-1] for p in pair_summaries]
        plot_bar([p.start_offset_ms for p in pair_summaries], labels,
                 "Thermal-visible start offset per recording", "Offset (ms)", outdir / "start_offsets.png")
        plot_bar([p.mean_time_diff_ms for p in pair_summaries], labels,
                 "Mean nearest-frame time difference", "Time difference (ms)", outdir / "mean_time_diff.png")
        plot_bar([p.mean_gps_distance_m for p in pair_summaries], labels,
                 "Mean GPS distance after timestamp pairing", "Distance (m)", outdir / "mean_gps_distance.png")
        plot_scatter([p.thermal_focal_len for p in pair_summaries], [p.visible_focal_len for p in pair_summaries],
                     "Thermal vs visible focal length", "Thermal focal length", "Visible focal length",
                     outdir / "focal_lengths_scatter.png")

    print("Done.")
    print(f"Output folder: {outdir.resolve()}")
    print(f"Summaries found: {len(pair_summaries)}")


if __name__ == "__main__":
    main()
