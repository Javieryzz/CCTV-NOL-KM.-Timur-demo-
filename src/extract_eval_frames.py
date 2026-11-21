#!/usr/bin/env python3
"""
extract_eval_frames.py - Langkah 5: Persiapan Evaluasi Manual CCTV Nol Km

Fungsi:
1. Membaca setiap file video hasil deteksi di outputs/videos/ (atau samples/)
2. Mengambil 30 frame tersebar merata (evenly spaced) per video
3. Menyimpan gambar frame ke outputs/eval_frames/<nama_klip>/frame_<idx:05d>.jpg
4. Membuat/mengisi outputs/logs/manual_eval.csv dengan kolom standar evaluasi PRD:
   - klip
   - nama_frame
   - jumlah_pelanggar_asli
   - terdeteksi_benar
   - terlewat
   - salah_alarm_non_pengendara
   - salah_alarm_pengendara
   - catatan

Penggunaan:
    python src/extract_eval_frames.py
    python src/extract_eval_frames.py --videos-dir outputs/videos --num-frames 30
"""

import argparse
import csv
import glob
import os
import re
import sys
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Ekstraksi frame sampel untuk evaluasi manual Langkah 5.")
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="outputs/videos",
        help="Direktori video sumber (default: outputs/videos)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_frames",
        help="Direktori penyimpanan frame ekstraksi (default: outputs/eval_frames)."
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=30,
        help="Jumlah frame per klip yang diambil (default: 30 frame)."
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="outputs/logs/manual_eval.csv",
        help="Path file CSV evaluasi manual (default: outputs/logs/manual_eval.csv)."
    )
    return parser.parse_args()


def get_clip_name(video_filename):
    """Membersihkan nama file dari extension dan akhiran _result untuk folder klip."""
    base = os.path.splitext(os.path.basename(video_filename))[0]
    return re.sub(r"_result$", "", base)


def extract_frames_from_video(video_path, output_clip_dir, num_frames=30):
    """
    Mengambil `num_frames` tersebar merata dari video_path
    dan menyimpannya ke output_clip_dir.
    Mengembalikan daftar info frame yang berhasil diekstraksi.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        print(f"[ERROR] Total frame video tidak valid ({total_frames}) pada: {video_path}")
        cap.release()
        return []

    os.makedirs(output_clip_dir, exist_ok=True)

    # Ambil index frame tersebar merata dari frame 0 hingga total_frames - 1
    if total_frames <= num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

    extracted_records = []
    print(f"\n[INFO] Mengekstrak {len(frame_indices)} frame dari: {os.path.basename(video_path)}")
    print(f"       Total Frame: {total_frames} | Resolusi: {int(cap.get(3))}x{int(cap.get(4))} @ {fps:.2f} FPS")

    for rank, frame_idx in enumerate(frame_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  [WARNING] Gagal membaca frame ke-{frame_idx}, lewati.")
            continue

        frame_filename = f"frame_{frame_idx:05d}.jpg"
        frame_path = os.path.join(output_clip_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        time_sec = frame_idx / fps
        extracted_records.append({
            "frame_idx": frame_idx,
            "frame_filename": frame_filename,
            "time_sec": time_sec,
            "frame_path": frame_path
        })

    cap.release()
    print(f"[SUCCESS] Berhasil mengekstrak {len(extracted_records)} frame ke: {output_clip_dir}")
    return extracted_records


def main():
    args = parse_args()

    # Cari seluruh video MP4 di direktori videos-dir
    video_files = sorted(glob.glob(os.path.join(args.videos_dir, "*.mp4")))
    if not video_files:
        print(f"[ERROR] Tidak ada file .mp4 yang ditemukan di: {args.videos_dir}")
        sys.exit(1)

    print("=" * 65)
    print("EKSTRAKSI FRAME EVALUASI MANUAL (LANGKAH 5)")
    print("=" * 65)
    print(f"Direktori Video Sumber : {args.videos_dir}")
    print(f"Total Video Ditemukan  : {len(video_files)}")
    print(f"Target Frame per Klip  : {args.num_frames}")
    print(f"Direktori Output Frame : {args.output_dir}")
    print(f"File CSV Evaluasi      : {args.csv_output}")
    print("=" * 65)

    all_eval_rows = []

    for video_file in video_files:
        clip_name = get_clip_name(video_file)
        clip_output_dir = os.path.join(args.output_dir, clip_name)

        extracted = extract_frames_from_video(video_file, clip_output_dir, args.num_frames)

        for item in extracted:
            all_eval_rows.append({
                "klip": clip_name,
                "nama_frame": item["frame_filename"],
                "jumlah_pelanggar_asli": "",
                "terdeteksi_benar": "",
                "terlewat": "",
                "salah_alarm_non_pengendara": "",
                "salah_alarm_pengendara": "",
                "catatan": f"Frame {item['frame_idx']} (t={item['time_sec']:.2f}s)"
            })

    # Tulis outputs/logs/manual_eval.csv
    csv_dir = os.path.dirname(args.csv_output)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    fieldnames = [
        "klip",
        "nama_frame",
        "jumlah_pelanggar_asli",
        "terdeteksi_benar",
        "terlewat",
        "salah_alarm_non_pengendara",
        "salah_alarm_pengendara",
        "catatan"
    ]

    with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_eval_rows)

    print("\n" + "=" * 65)
    print("HASIL PEMBUATAN MANUAL EVALUATION")
    print("=" * 65)
    print(f"File CSV Berhasil Dibuat : {args.csv_output}")
    print(f"Total Baris Frame Dicatat: {len(all_eval_rows)}")
    print("Rincian per Klip:")
    clip_counts = {}
    for row in all_eval_rows:
        clip_counts[row["klip"]] = clip_counts.get(row["klip"], 0) + 1
    for clip, count in sorted(clip_counts.items()):
        print(f"  - {clip:20s}: {count} frame tersimpan di {os.path.join(args.output_dir, clip)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
