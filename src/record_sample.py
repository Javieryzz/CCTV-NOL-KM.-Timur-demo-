#!/usr/bin/env python3
"""
record_sample.py
Merekam klip video dari live stream CCTV menggunakan ffmpeg,
kemudian memverifikasi hasil rekaman (resolusi, FPS, durasi) dengan OpenCV.

Contoh penggunaan:
    python src/record_sample.py --url "https://stream-url.m3u8" --duration 150 --output samples/nol_km_malam_1.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Rekam klip sampel CCTV menggunakan ffmpeg dan cek dengan OpenCV.")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL stream CCTV (HLS/m3u8 atau RTSP/HTTP stream)."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=150,
        help="Durasi rekaman dalam detik (default: 150 detik)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="samples/nol_km_malam_1.mp4",
        help="Path file output (default: samples/nol_km_malam_1.mp4)."
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=str,
        default="ffmpeg",
        help="Path ke binary ffmpeg (default: 'ffmpeg' dari PATH)."
    )
    return parser.parse_args()


def verify_with_opencv(file_path):
    """Membuka file video yang telah disimpan dan mencetak info resolusi, FPS, dan durasi."""
    try:
        import cv2
    except ImportError:
        print("[WARNING] Modul cv2 belum terpasang. Lewati verifikasi OpenCV.")
        return

    if not os.path.isfile(file_path):
        print(f"[ERROR] File rekaman tidak ditemukan: {file_path}")
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video dengan OpenCV: {file_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps and fps > 0 else 0.0

    cap.release()

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("HASIL VERIFIKASI VIDEO (OpenCV)")
    print("=" * 50)
    print(f"File       : {file_path}")
    print(f"Ukuran     : {file_size_mb:.2f} MB")
    print(f"Resolusi   : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Total Frame: {total_frames}")
    print(f"Durasi     : {duration:.2f} detik")
    print("=" * 50 + "\n")


def get_ffmpeg_executable(custom_path="ffmpeg"):
    """Mencari executable ffmpeg dari custom_path, PATH sistem, atau imageio_ffmpeg."""
    # 1. Cek custom_path eksplisit atau jika ada di PATH
    if custom_path and shutil.which(custom_path):
        return shutil.which(custom_path)
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    # 2. Cek apakah ada di PATH dengan nama 'ffmpeg'
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg

    # 3. Fallback ke binary dari imageio_ffmpeg jika terinstall
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.isfile(exe):
            return exe
    except ImportError:
        pass

    return None


def record_stream(url, duration, output_path, ffmpeg_bin):
    """Menjalankan proses ffmpeg untuk merekam stream."""
    # Pastikan direktori output ada
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Validasi executable ffmpeg
    resolved_ffmpeg = get_ffmpeg_executable(ffmpeg_bin)
    if not resolved_ffmpeg:
        print(f"[ERROR] ffmpeg tidak ditemukan di PATH, di path '{ffmpeg_bin}', atau imageio-ffmpeg.")
        print("Silakan install ffmpeg atau sediakan path melalui argumen --ffmpeg-path.")
        sys.exit(1)

    print(f"[INFO] Memulai perekaman:")
    print(f"  Stream URL : {url}")
    print(f"  Durasi     : {duration} detik")
    print(f"  Output     : {output_path}")
    print(f"  FFmpeg Bin : {resolved_ffmpeg}")

    # -y (overwrite), -i url, -t duration, -c copy (stream copy tanpa re-encode)
    cmd = [
        resolved_ffmpeg,
        "-y",
        "-i", url,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[SUCCESS] Perekaman selesai: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Gagal merekam video: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    record_stream(args.url, args.duration, args.output, args.ffmpeg_path)
    verify_with_opencv(args.output)


if __name__ == "__main__":
    main()
