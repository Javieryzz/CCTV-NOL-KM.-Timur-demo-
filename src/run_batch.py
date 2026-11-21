#!/usr/bin/env python3
"""
run_batch.py - Eksekusi batch pengujian keempat video CCTV Nol Km - Timur
dan pembandingan performa Sebelum vs Sesudah perbaikan.
"""

import os
import sys
import glob
import json
import time

sys.path.insert(0, os.path.abspath("."))
from src.main import run_pipeline

VIDEOS = [
    "samples/nol_km_malam_1.mp4",
    "samples/nol_km_malam_2.mp4",
    "samples/nol_km_siang_2.mp4",
    "samples/nol_km_siang_1.mp4",
]

CONFIG_PATH = "configs/nol_km_timur.json"

BASELINE_STATS = {
    "nol_km_siang_1": {
        "frames": 6612,
        "avg_fps": 16.48,
        "alerts": 152,
    },
    "nol_km_siang_2": {
        "frames": 3908,
        "avg_fps": 16.19,
        "alerts": 74,
    },
    "nol_km_malam_1": {
        "frames": 3563,
        "avg_fps": 14.85,
        "alerts": 36,
    },
    "nol_km_malam_2": {
        "frames": 4354,
        "avg_fps": 14.52,
        "alerts": 47,
    },
}

ROUND1_STATS = {
    "nol_km_siang_1": {
        "avg_fps": 10.60,
        "alerts": 100,
    },
    "nol_km_siang_2": {
        "avg_fps": 7.47,
        "alerts": 37,
    },
    "nol_km_malam_1": {
        "avg_fps": 10.50,
        "alerts": 32,
    },
    "nol_km_malam_2": {
        "avg_fps": 8.93,
        "alerts": 40,
    },
}


def clean_snapshots_for_stem(stem):
    """Menghapus snapshot lama untuk stem tertentu agar penghitungan snapshot baru akurat."""
    pattern = os.path.join("outputs", "snapshots", f"{stem}_*.jpg")
    old_files = glob.glob(pattern)
    for f in old_files:
        try:
            os.remove(f)
        except OSError:
            pass


def main():
    print("=" * 70)
    print("BATCH RUNNER EVALUASI CCTV NOL KM (POST-IMPROVEMENT)")
    print("=" * 70)
    print(f"Total video yang akan diproses: {len(VIDEOS)}")
    for v in VIDEOS:
        print(f" - {v}")
    print(f"Config: {CONFIG_PATH}")
    print("=" * 70, flush=True)

    results = {}
    batch_start = time.time()

    for idx, video_path in enumerate(VIDEOS, 1):
        stem = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[{idx}/{len(VIDEOS)}] Memproses {video_path} ...", flush=True)

        # Bersihkan snapshot lama untuk klip ini
        clean_snapshots_for_stem(stem)

        # Jalankan pipeline
        res = run_pipeline(
            source_path=video_path,
            config_path=CONFIG_PATH,
            output_path=os.path.join("outputs", "videos", f"{stem}_result.mp4"),
            max_frames=None,
            device="cpu",
            show=False
        )
        results[stem] = res

    batch_duration = time.time() - batch_start

    # Simpan laporan JSON
    report_data = {
        "batch_duration_seconds": batch_duration,
        "results": results,
        "baseline": BASELINE_STATS
    }
    with open("outputs/logs/evaluation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Cetak ringkasan tabel
    print("\n" + "=" * 105)
    print("RINGKASAN PERBANDINGAN TIGA TAHAP: BASELINE vs PUTARAN 1 vs FINAL")
    print("=" * 105)
    header = f"{'Nama Klip':<16} | {'FPS Base':<9} | {'FPS Final':<9} | {'Alert Base':<10} | {'Alert R1':<9} | {'Alert Final':<11} | {'Delta Final':<12}"
    print(header)
    print("-" * 105)

    total_alerts_before = 0
    total_alerts_r1 = 0
    total_alerts_final = 0

    for stem in ["nol_km_siang_1", "nol_km_siang_2", "nol_km_malam_1", "nol_km_malam_2"]:
        if stem in results:
            res = results[stem]
            base = BASELINE_STATS.get(stem, {"avg_fps": 0.0, "alerts": 0})
            r1 = ROUND1_STATS.get(stem, {"avg_fps": 0.0, "alerts": 0})

            fps_b = base["avg_fps"]
            fps_f = res["avg_fps"]
            alt_b = base["alerts"]
            alt_r1 = r1["alerts"]
            alt_f = res["total_incidents"]
            diff = alt_f - alt_b
            pct = ((alt_f - alt_b) / alt_b * 100) if alt_b > 0 else 0

            total_alerts_before += alt_b
            total_alerts_r1 += alt_r1
            total_alerts_final += alt_f

            print(f"{stem:<16} | {fps_b:<9.2f} | {fps_f:<9.2f} | {alt_b:<10d} | {alt_r1:<9d} | {alt_f:<11d} | {diff:+d} ({pct:+.1f}%)")

    print("-" * 105)
    total_diff = total_alerts_final - total_alerts_before
    total_pct = ((total_alerts_final - total_alerts_before) / total_alerts_before * 100) if total_alerts_before > 0 else 0
    print(f"{'TOTAL':<16} | {'-':<9} | {'-':<9} | {total_alerts_before:<10d} | {total_alerts_r1:<9d} | {total_alerts_final:<11d} | {total_diff:+d} ({total_pct:+.1f}%)")
    print("=" * 105)
    print(f"Total waktu komputasi batch: {batch_duration/60:.2f} menit")
    print(f"File log perbandingan tersimpan di: outputs/logs/evaluation_comparison.json\n", flush=True)


if __name__ == "__main__":
    main()
