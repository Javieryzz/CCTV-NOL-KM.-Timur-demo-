#!/usr/bin/env python3
"""
draw_roi.py
Membaca frame dari file video di samples/ untuk menentukan polygon red_zone secara interaktif.
Menyimpan koordinat polygon dalam bentuk rasio (0.0 - 1.0) ke file config.

Fitur:
- Klik kiri: Tambah titik polygon.
- Klik kanan / tombol 'u': Undo titik terakhir.
- Tombol 'r': Reset seluruh titik.
- Tombol 's' atau 'c': Simpan polygon ke config (dikonversi ke rasio 0-1) dan simpan gambar verifikasi.
- Tombol 'q' / ESC: Keluar tanpa menyimpan.
- Flag --verify: Mode non-interaktif untuk membaca polygon dari config dan menyimpan visualisasi overlay ke outputs/.

Contoh penggunaan:
    # Mode interaktif klik polygon:
    python src/draw_roi.py --video samples/nol_km_siang_1.mp4 --config configs/nol_km_timur.json

    # Mode verifikasi overlay:
    python src/draw_roi.py --video samples/nol_km_siang_1.mp4 --config configs/nol_km_timur.json --verify
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Gambar dan verifikasi ROI Polygon Red Zone.")
    parser.add_argument(
        "--video",
        type=str,
        default="samples/nol_km_siang_1.mp4",
        help="Path video sampel (default: samples/nol_km_siang_1.mp4)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nol_km_timur.json",
        help="Path file config JSON (default: configs/nol_km_timur.json)."
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=100,
        help="Indeks frame yang digunakan untuk menggambar ROI (default: 100)."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Hanya simpan overlay verifikasi dari config yang sudah ada ke outputs/ tanpa membuka window klik."
    )
    parser.add_argument(
        "--output-verify",
        type=str,
        default="outputs/verify_roi.jpg",
        help="Path gambar hasil verifikasi overlay (default: outputs/verify_roi.jpg)."
    )
    return parser.parse_args()


def load_frame(video_path, frame_idx=100):
    """Membaca satu frame tertentu dari video."""
    if not os.path.isfile(video_path):
        print(f"[ERROR] File video tidak ditemukan: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_idx = min(max(0, frame_idx), max(0, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[ERROR] Gagal membaca frame ke-{target_idx} dari {video_path}")
        sys.exit(1)

    return frame, target_idx


def draw_overlay(frame, points_px, fill_color=(0, 0, 255), alpha=0.35):
    """Menggambar polygon semi-transparan dengan garis tepi dan titik sudut."""
    overlay = frame.copy()
    pts = np.array(points_px, dtype=np.int32).reshape((-1, 1, 2))

    # Fill polygon
    cv2.fillPoly(overlay, [pts], fill_color)
    output = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Outline polygon
    cv2.polylines(output, [pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # Titik sudut
    for i, pt in enumerate(points_px):
        cv2.circle(output, pt, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            output,
            str(i + 1),
            (pt[0] + 6, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

    return output


def save_config_roi(config_path, points_ratio):
    """Menyimpan titik rasio polygon ke config JSON."""
    if not os.path.isfile(config_path):
        data = {}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    data["red_zone"] = points_ratio

    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[SUCCESS] Polygon ROI berhasil disimpan ke: {config_path}")
    print(f"Jumlah titik: {len(points_ratio)}")
    print("Koordinat rasio:")
    for i, pt in enumerate(points_ratio):
        print(f"  Titik {i + 1}: [{pt[0]:.4f}, {pt[1]:.4f}]")


def verify_mode(frame, config_path, output_verify_path):
    """Mode verifikasi: membaca config dan menggambar overlay ke file output."""
    if not os.path.isfile(config_path):
        print(f"[ERROR] Config tidak ditemukan: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    red_zone = data.get("red_zone", [])
    if not red_zone or len(red_zone) < 3:
        print(f"[ERROR] 'red_zone' dalam {config_path} tidak valid atau kurang dari 3 titik.")
        sys.exit(1)

    h, w = frame.shape[:2]
    points_px = [[int(pt[0] * w), int(pt[1] * h)] for pt in red_zone]

    result = draw_overlay(frame, points_px)

    # Tambahkan header teks info
    info_text = f"Red Zone ROI Overlay ({len(points_px)} points) - W:{w} H:{h}"
    cv2.putText(
        result,
        info_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    out_dir = os.path.dirname(output_verify_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_verify_path, result)
    print(f"[SUCCESS] Gambar verifikasi ROI disimpan ke: {output_verify_path}")


def interactive_mode(frame, frame_idx, video_path, config_path, output_verify_path):
    """Mode interaktif GUI dengan klik mouse."""
    h, w = frame.shape[:2]
    points_px = []
    window_name = "Draw Red Zone ROI (Left-click: add, Right-click: undo, 's': save, 'r': reset, 'q': quit)"

    def redraw():
        temp = frame.copy()
        if len(points_px) >= 3:
            temp = draw_overlay(temp, points_px)
        elif len(points_px) > 0:
            for i, pt in enumerate(points_px):
                cv2.circle(temp, pt, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.putText(temp, str(i + 1), (pt[0] + 6, pt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            if len(points_px) >= 2:
                for i in range(len(points_px) - 1):
                    cv2.line(temp, points_px[i], points_px[i + 1], (0, 0, 255), 2, lineType=cv2.LINE_AA)

        instructions = [
            f"Video: {os.path.basename(video_path)} | Frame: {frame_idx} | Dim: {w}x{h}",
            f"Titik saat ini: {len(points_px)}",
            "Klik Kiri: Tambah | Klik Kanan/U: Undo | R: Reset | S/C: Simpan | Q: Keluar"
        ]

        # Draw semi-transparent header bar
        header_bar = temp[:80, :].copy()
        cv2.rectangle(temp, (0, 0), (w, 80), (20, 20, 20), -1)
        temp[:80, :] = cv2.addWeighted(temp[:80, :], 0.75, header_bar, 0.25, 0)

        for idx, text in enumerate(instructions):
            cv2.putText(temp, text, (15, 22 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return temp

    def mouse_callback(event, x, y, flags, param):
        nonlocal points_px
        if event == cv2.EVENT_LBUTTONDOWN:
            points_px.append((x, y))
            cv2.imshow(window_name, redraw())
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points_px:
                points_px.pop()
                cv2.imshow(window_name, redraw())

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, redraw())

    print("[INFO] Buka jendela gambar ROI...")
    print("Petunjuk:")
    print(" - Klik kiri untuk menentukan titik sudut polygon red zone.")
    print(" - Pastikan memberi jarak/margin kecil dari garis bollard ke arah jalan.")
    print(" - Tekan 's' atau 'c' untuk menyimpan titik ke config JSON.")
    print(" - Tekan 'r' untuk reset titik.")
    print(" - Tekan 'u' atau klik kanan untuk undo titik terakhir.")
    print(" - Tekan 'q' atau ESC untuk membatalkan.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in [ord("q"), 27]:  # ESC or q
            print("[INFO] Dibatalkan tanpa menyimpan.")
            break
        elif key in [ord("u"), ord("U")]:
            if points_px:
                points_px.pop()
                cv2.imshow(window_name, redraw())
        elif key in [ord("r"), ord("R")]:
            points_px.clear()
            cv2.imshow(window_name, redraw())
            print("[INFO] Titik di-reset.")
        elif key in [ord("s"), ord("S"), ord("c"), ord("C")]:
            if len(points_px) < 3:
                print("[WARNING] Polygon memerlukan minimal 3 titik!")
                continue

            points_ratio = [[round(x / w, 4), round(y / h, 4)] for (x, y) in points_px]
            save_config_roi(config_path, points_ratio)

            # Simpan juga gambar verifikasi
            result_overlay = draw_overlay(frame, points_px)
            out_dir = os.path.dirname(output_verify_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(output_verify_path, result_overlay)
            print(f"[SUCCESS] Gambar verifikasi disimpan ke: {output_verify_path}")
            break

    cv2.destroyAllWindows()


def main():
    args = parse_args()
    frame, frame_idx = load_frame(args.video, args.frame_idx)

    if args.verify:
        verify_mode(frame, args.config, args.output_verify)
    else:
        interactive_mode(frame, frame_idx, args.video, args.config, args.output_verify)


if __name__ == "__main__":
    main()
