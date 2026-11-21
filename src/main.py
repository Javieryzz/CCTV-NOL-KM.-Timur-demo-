#!/usr/bin/env python3
"""
main.py - Red Zone Detection pada CCTV Kota Yogyakarta (Versi Teroptimasi)

Penyempurnaan berdasarkan evaluasi Langkah 5:
1. Audit tracking: Memastikan model.track(..., persist=True) dan menggunakan
   configs/tracker_custom.yaml dengan track_buffer=60 untuk menangani oklusi tiang/kendaraan.
2. Logika pengecualian pengendara disempurnakan:
   - Padding asimetris kotak motor (35% ke atas, 15% samping dan bawah) untuk mencakup tubuh/helm pengendara.
   - Ambang overlap diturunkan ke 0.20.
   - Sticky rider status (rider_sticky_seconds: 2.0s): mempertahankan status pengendara bila deteksi motor hilang sesaat.
3. Model & konfidensi fleksibel dari config:
   - Mendukung key 'model' (yolov8s.pt / yolov8n.pt / yolo11s.pt).
   - Ambang konfidensi diturunkan ke 0.25 untuk kestabilan deteksi jarak menengah/jauh.
4. Dwell Time & Cooldown tetap aktif:
   - Dwell time 0.8s berbasis waktu video sebelum box berubah merah.
   - Cooldown 10s per track_id untuk mencegah spam alert/snapshot/log CSV.

Penggunaan:
    python src/main.py --source samples/nol_km_siang_1.mp4 --config configs/nol_km_timur.json
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Deteksi Red Zone CCTV Nol Km - Versi Teroptimasi")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path video input atau URL live stream HLS (contoh: samples/nol_km_siang_1_16x9.mp4 atau https://.../playlist.m3u8)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nol_km_timur.json",
        help="Path file config JSON (default: configs/nol_km_timur.json)."
    )
    parser.add_argument(
        "--fallback",
        type=str,
        default="samples/nol_km_siang_1_16x9.mp4",
        help="Path klip lokal cadangan jika live stream gagal dibuka atau terputus (default: samples/nol_km_siang_1_16x9.mp4)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path file output video (default: outputs/videos/<nama_file>_result.mp4)."
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Simpan video output (pada mode file aktif secara default; pada mode live hanya aktif jika flag ini atau --output diberikan)."
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Jangan simpan file video output ke disk (cocok untuk mode live preview real-time)."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maksimal frame yang diproses (opsional, untuk pengujian cepat)."
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Maksimal durasi pemrosesan dalam detik waktu nyata/wall-clock (opsional, berguna untuk mode live stream)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device untuk komputasi ('cpu' atau 'cuda', default: 'cpu')."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model YOLO dari config (contoh: yolov8n.pt agar lebih ringan untuk live preview)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override ukuran input imgsz dari config (contoh: 640 agar lebih cepat untuk live preview)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Tampilkan window video real-time saat pemrosesan."
    )
    return parser.parse_args()


def is_stream_source(source_str):
    """Mendeteksi apakah source berupa URL live stream (HLS/RTSP/HTTP/RTMP)."""
    return str(source_str).strip().lower().startswith(("http://", "https://", "rtsp://", "rtmp://"))


def derive_source_stem(source_str, config):
    """Menentukan nama dasar file output/log yang rapi untuk file lokal maupun URL stream."""
    if is_stream_source(source_str):
        cam = config.get("camera", "cctv")
        return f"{cam}_live"
    return os.path.splitext(os.path.basename(source_str))[0]


def _open_stream_capture(source_path):
    """Membuka VideoCapture untuk live stream HLS/HTTP dengan opsi reconnect FFmpeg & buffer minimal."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2|rw_timeout;4000000"
    )
    cap = cv2.VideoCapture(source_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source_path)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def open_source_with_fallback(source_path, fallback_path):
    """
    Membuka sumber video atau live stream. Jika sumber adalah URL stream dan gagal
    dibuka atau gagal membaca frame perdana, otomatis beralih ke klip fallback lokal.
    Mengembalikan tuple: (cap, active_source, is_live, used_fallback, first_frame).
    """
    is_live = is_stream_source(source_path)
    if is_live:
        print(f"[INFO] Membuka live stream: {source_path}", flush=True)
        cap = _open_stream_capture(source_path)

        first_frame = None
        if cap.isOpened():
            ret, frm = cap.read()
            if ret and frm is not None:
                first_frame = frm

        if cap.isOpened() and first_frame is not None:
            return cap, source_path, True, False, first_frame

        # Gagal membuka atau membaca frame perdana stream -> beralih ke fallback
        if cap.isOpened():
            cap.release()
        print(f"[FALLBACK] Live stream gagal dibuka atau tidak merespons: {source_path}", flush=True)
        if fallback_path and os.path.isfile(fallback_path):
            print(f"[FALLBACK] Beralih otomatis ke klip lokal: {fallback_path}", flush=True)
            fb_cap = cv2.VideoCapture(fallback_path)
            if not fb_cap.isOpened():
                raise RuntimeError(f"Gagal membuka klip fallback: {fallback_path}")
            return fb_cap, fallback_path, False, True, None
        raise RuntimeError(f"Gagal membuka stream ({source_path}) dan file fallback tidak ditemukan ({fallback_path}).")

    # Mode file lokal
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video sumber: {source_path}")
    return cap, source_path, False, False, None


class ThreadedStreamReader:
    """
    Membaca frame dari live stream di background thread secara kontinu,
    otomatis mencoba reconnect jika koneksi HLS mengalami EOF/jeda segmen,
    dan hanya menyimpan frame terbaru agar pemrosesan tidak menumpuk delay.
    """

    def __init__(self, cap, source_url, max_consecutive_failures=2, max_reconnects=5):
        self.cap = cap
        self.source_url = source_url
        self.max_consecutive_failures = max_consecutive_failures
        self.max_reconnects = max_reconnects
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_seq = 0
        self.disconnected = False
        self.reconnecting = False
        self._stopped = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _try_reconnect(self):
        """Mencoba membuka ulang koneksi HLS saat mengalami EOF atau putus sesaat."""
        with self.lock:
            self.reconnecting = True
        try:
            for attempt in range(1, self.max_reconnects + 1):
                if self._stopped:
                    return False
                print(f"[INFO] Koneksi stream terputus, mencoba reconnect ({attempt}/{self.max_reconnects})...", flush=True)
                try:
                    self.cap.release()
                except Exception:
                    pass
                time.sleep(0.5)
                new_cap = _open_stream_capture(self.source_url)
                if new_cap.isOpened():
                    ret, frm = new_cap.read()
                    if ret and frm is not None:
                        self.cap = new_cap
                        with self.lock:
                            self.latest_frame = frm
                            self.frame_seq += 1
                        print("[INFO] Reconnect live stream berhasil!", flush=True)
                        return True
                if new_cap.isOpened():
                    new_cap.release()
            return False
        finally:
            with self.lock:
                self.reconnecting = False

    def _reader_loop(self):
        failures = 0
        while not self._stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                failures += 1
                if failures >= self.max_consecutive_failures:
                    if self._try_reconnect():
                        failures = 0
                        continue
                    with self.lock:
                        self.disconnected = True
                    break
                time.sleep(0.05)
                continue
            failures = 0
            with self.lock:
                self.latest_frame = frame
                self.frame_seq += 1

    def read_latest(self, last_seq, timeout=25.0):
        """Mengambil frame terbaru yang lebih baru dari last_seq, atau False jika stream putus permanen."""
        deadline = time.monotonic() + timeout
        while not self._stopped:
            with self.lock:
                if self.frame_seq > last_seq and self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None
                    return True, frame, self.frame_seq
                if self.disconnected:
                    return False, None, last_seq
            if time.monotonic() >= deadline:
                return False, None, last_seq
            time.sleep(0.01)
        return False, None, last_seq

    def stop(self):
        self._stopped = True
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self.cap.release()
        except Exception:
            pass


def load_config(config_path):
    """Membaca konfigurasi JSON."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_tracker_config(tracker_path, track_buffer=60):
    """Memastikan file konfigurasi tracker YAML tersedia dan berisi track_buffer."""
    if not os.path.isfile(tracker_path):
        tracker_yaml = f"""# Custom Tracker Config untuk CCTV Nol Km
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {track_buffer}
match_thresh: 0.8
fuse_score: True
"""
        os.makedirs(os.path.dirname(tracker_path) or ".", exist_ok=True)
        with open(tracker_path, "w", encoding="utf-8") as f:
            f.write(tracker_yaml)
    return tracker_path


def format_timestamp(seconds):
    """Format waktu dalam detik ke format HH:MM:SS.ff."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:05.2f}"


def expand_box_asymmetric(box, pad_up=0.35, pad_side=0.15, pad_down=0.15, frame_w=1920, frame_h=1080):
    """
    Memperlebar bounding box kendaraan secara asimetris:
    - 35% ke atas (mencakup kepala, helm, dan torso pengendara pada sudut kamera tinggi)
    - 15% ke kiri, kanan, dan bawah (kaki yang menapak ke aspal)
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    nx1 = max(0.0, x1 - w * pad_side)
    ny1 = max(0.0, y1 - h * pad_up)
    nx2 = min(float(frame_w), x2 + w * pad_side)
    ny2 = min(float(frame_h), y2 + h * pad_down)

    return [nx1, ny1, nx2, ny2]


def compute_overlap_ratio(person_box, vehicle_box):
    """Menghitung rasio overlap: Intersection Area / Luas Box Person."""
    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = vehicle_box

    inter_x1 = max(px1, vx1)
    inter_y1 = max(py1, vy1)
    inter_x2 = min(px2, vx2)
    inter_y2 = min(py2, vy2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    person_area = max(1.0, (px2 - px1) * (py2 - py1))
    return inter_area / person_area


def is_point_inside_or_near_box(point, box, margin=20.0):
    """Memeriksa apakah sebuah titik berada di dalam atau dekat bounding box."""
    px, py = point
    x1, y1, x2, y2 = box
    return (x1 - margin) <= px <= (x2 + margin) and (y1 - margin) <= py <= (y2 + margin)


def draw_hud(frame, frame_idx, total_frames, video_time, fps_instant, fps_avg,
             safe_count, active_violators, total_incidents, riders_count, model_name,
             banner_alert=False, is_live=False, used_fallback=False):
    """Menggambar HUD informasi dan status di layar."""
    h, w = frame.shape[:2]

    # Panel status semi-transparan (kiri atas)
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (510, 165), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (15, 15), (510, 165), (80, 80, 80), 1)

    time_str = format_timestamp(video_time)
    mode_tag = "LIVE" if is_live else ("FALLBACK" if used_fallback else "FILE")
    cv2.putText(frame, f"CCTV Nol Km - Timur [{mode_tag}] | Model: {model_name}", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    if is_live or not total_frames:
        progress_str = f"Frame: {frame_idx} (LIVE) | Waktu: {time_str}"
    else:
        progress_str = f"Frame: {frame_idx}/{total_frames} | Waktu Video: {time_str}"
    cv2.putText(frame, progress_str, (25, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    fps_str = f"FPS Pemrosesan: {fps_instant:.1f} (Rata-rata: {fps_avg:.1f})"
    cv2.putText(frame, fps_str, (25, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    stat_str = f"Pelanggar Aktif: {active_violators}  |  Total Kejadian (Alert): {total_incidents}"
    stat_color = (0, 0, 255) if active_violators > 0 else (100, 255, 100)
    cv2.putText(frame, stat_str, (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, stat_color, 2, cv2.LINE_AA)
    rider_str = f"Orang Aman: {safe_count}  (Pengendara: {riders_count})"
    cv2.putText(frame, rider_str, (25, 134),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
    info_str = "Dwell Time, Cooldown, Sticky Rider & Asym-Padding Aktif"
    cv2.putText(frame, info_str, (25, 156),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # Banner alert mencolok di tengah atas frame jika ada pelanggar aktif
    if banner_alert or active_violators > 0:
        alert_w = 540
        alert_x1 = (w - alert_w) // 2
        alert_x2 = alert_x1 + alert_w
        cv2.rectangle(frame, (alert_x1, 15), (alert_x2, 65), (0, 0, 220), -1)
        cv2.rectangle(frame, (alert_x1, 15), (alert_x2, 65), (0, 0, 255), 2)
        alert_text = f"PERINGATAN: {active_violators} PELANGGAR RED ZONE AKTIF!"
        cv2.putText(frame, alert_text, (alert_x1 + 18, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def save_alert_snapshot(frame, snapshot_path, video_time, frame_idx, track_id, dwell_time):
    """Menyimpan 1 snapshot gambar kejadian dengan timestamp video dan alert tercetak."""
    snap = frame.copy()
    h, w = snap.shape[:2]

    banner_h = 55
    cv2.rectangle(snap, (0, h - banner_h), (w, h), (10, 10, 10), -1)
    cv2.line(snap, (0, h - banner_h), (w, h - banner_h), (0, 0, 255), 2)

    time_str = format_timestamp(video_time)
    caption_left = f"ALERT RED ZONE | WAKTU VIDEO: {time_str} (Frame: {frame_idx}) | TRACK ID: {track_id}"
    caption_right = f"DURASI ZONA: {dwell_time:.2f} detik"

    cv2.putText(snap, caption_left, (20, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(snap, caption_right, (w - 320, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    cv2.imwrite(snapshot_path, snap)


def run_pipeline(source_path, config_path, output_path=None, max_frames=None, device="cpu", show=False,
                 fallback_path="samples/nol_km_siang_1_16x9.mp4", save_video=None, no_save_video=False,
                 max_seconds=None, model_override=None, imgsz_override=None):
    """Menjalankan pipeline Red Zone teroptimasi (mendukung file lokal & live stream HLS dengan fallback)."""
    config = load_config(config_path)

    model_name = model_override or config.get("model", "yolov8s.pt")
    conf_thresh = config.get("conf", 0.25)
    iou_thresh = config.get("iou", 0.65)
    imgsz = imgsz_override or config.get("imgsz", 960)
    track_buffer = config.get("track_buffer", 60)
    tracker_config_file = ensure_tracker_config(config.get("tracker_config", "configs/tracker_custom.yaml"), track_buffer)

    # Parameter logika pengendara asimetris & sticky (motor & sepeda)
    pad_up = config.get("rider_box_pad_up", 0.35)
    pad_side = config.get("rider_box_pad_side", 0.15)
    pad_down = config.get("rider_box_pad_down", 0.15)
    rider_overlap_threshold = config.get("rider_overlap_threshold", 0.20)
    rider_sticky_seconds = config.get("rider_sticky_seconds", 2.0)

    # Parameter logika penumpang di dalam kendaraan besar (mobil, bus, truk)
    # Tanpa padding dan overlap >= 0.80 agar pejalan kaki di dekat mobil tidak salah dianggap rider
    vehicle_overlap_threshold = config.get("vehicle_overlap_threshold", 0.80)

    dwell_seconds = config.get("dwell_seconds", 0.8)
    cooldown_seconds = config.get("cooldown_seconds", 10.0)
    save_snapshots = config.get("save_snapshots", True)
    target_class_names = set(config.get("classes", ["person", "bicycle", "motorcycle", "car", "bus", "truck"]))

    requested_live = is_stream_source(source_path)

    # Buka sumber (stream atau file), otomatis beralih ke fallback jika stream gagal dibuka
    cap, active_source, is_live, used_fallback, pending_first_frame = open_source_with_fallback(
        source_path, fallback_path
    )

    source_stem = derive_source_stem(source_path, config)
    if used_fallback:
        source_stem = f"{source_stem}_fallback"

    # Kontrol penyimpanan video: --no-save-video menonaktifkan penulisan file video sepenuhnya
    if no_save_video:
        should_save_video = False
    elif output_path is not None:
        should_save_video = True
    elif save_video is not None:
        should_save_video = bool(save_video)
    else:
        should_save_video = not requested_live

    if should_save_video and output_path is None:
        output_path = os.path.join("outputs", "videos", f"{source_stem}_result.mp4")

    if should_save_video and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    snapshots_dir = os.path.join("outputs", "snapshots")
    logs_dir = os.path.join("outputs", "logs")
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Inisialisasi CSV log file (FR-08)
    csv_log_path = os.path.join(logs_dir, f"{source_stem}_events.csv")
    csv_file = open(csv_log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp_video",
        "frame_number",
        "track_id",
        "foot_x",
        "foot_y",
        "zone_duration",
        "source_file",
        "snapshot_path"
    ])
    csv_file.flush()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_in = raw_fps if (raw_fps and 1.0 <= raw_fps <= 120.0) else 25.0
    total_frames = 0 if is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = min(total_frames, max_frames) if (max_frames and total_frames > 0) else total_frames

    print("\n" + "=" * 65)
    print("MEMULAI EVALUASI PIPELINE (VERSI PENYEMPURNAAN EVALUASI)")
    print("=" * 65)
    print(f"Sumber Diminta   : {source_path}")
    print(f"Sumber Aktif     : {active_source} ({'LIVE STREAM' if is_live else ('FALLBACK FILE' if used_fallback else 'FILE LOKAL')})")
    print(f"Output Video     : {output_path if should_save_video else 'Tidak disimpan (mode live tanpa --save-video)'}")
    print(f"Log CSV          : {csv_log_path}")
    print(f"Model YOLO       : {model_name} (conf={conf_thresh}, iou={iou_thresh}, imgsz={imgsz})")
    print(f"Tracker Config   : {tracker_config_file} (track_buffer={track_buffer})")
    print(f"Padding Motor    : Up {pad_up*100:.0f}%, Side {pad_side*100:.0f}%, Down {pad_down*100:.0f}% (Overlap >= {rider_overlap_threshold})")
    print(f"Mobil/Bus/Truk   : Tanpa padding (Overlap >= {vehicle_overlap_threshold} di dalam kendaraan)")
    print(f"Sticky Rider     : {rider_sticky_seconds}s | Dwell: {dwell_seconds}s | Cooldown: {cooldown_seconds}s")
    if max_seconds:
        print(f"Batas Durasi     : {max_seconds:.1f} detik (wall-clock)")
    print("=" * 65 + "\n", flush=True)

    # Inisialisasi polygon dari rasio ke pixel
    red_zone_ratios = config.get("red_zone", [])
    polygon_px = np.array(
        [[int(pt[0] * width), int(pt[1] * height)] for pt in red_zone_ratios],
        dtype=np.int32
    )

    # Inisialisasi YOLO model
    model = YOLO(model_name)
    target_class_ids = [k for k, v in model.names.items() if v in target_class_names]

    if device == "cpu":
        torch.set_num_threads(8)

    # VideoWriter (hanya dibuat jika should_save_video=True)
    out_writer = None
    if should_save_video and output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

    # Struktur pelacak
    zone_tracker = {}        # track_id -> {entry_time, last_seen_time, last_alert_time}
    rider_last_seen = {}     # track_id -> last_video_time_seen_as_rider (Sticky status)
    total_incidents = 0
    total_snapshots_saved = 0
    total_csv_rows_written = 0

    # Aktifkan threaded frame grabber (dengan auto-reconnect HLS) untuk input live stream
    stream_reader = ThreadedStreamReader(cap, active_source, max_consecutive_failures=2, max_reconnects=5) if is_live else None
    last_stream_seq = 0

    if show:
        cv2.namedWindow("Red Zone CCTV Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Red Zone CCTV Preview", 1280, 720)

    frame_idx = 0
    local_frame_idx = 0
    fallback_time_offset = 0.0
    start_total_time = time.monotonic()
    processing_times = []

    try:
        while True:
            if max_frames and frame_idx >= max_frames:
                break
            if max_seconds and (time.monotonic() - start_total_time) >= max_seconds:
                print(f"[INFO] Batas durasi --max-seconds ({max_seconds}s) tercapai. Menghentikan pemrosesan.", flush=True)
                break

            if pending_first_frame is not None:
                ret, frame = True, pending_first_frame
                pending_first_frame = None
            elif stream_reader is not None:
                ret, frame, last_stream_seq = stream_reader.read_latest(last_stream_seq, timeout=25.0)
            else:
                ret, frame = cap.read()

            if not ret or frame is None:
                if is_live:
                    # Stream terputus di tengah jalan -> hentikan reader & beralih otomatis ke fallback
                    print("[FALLBACK] Live stream terputus atau tidak mengirim frame baru.", flush=True)
                    if stream_reader is not None:
                        stream_reader.stop()
                        stream_reader = None
                    cap.release()
                    if fallback_path and os.path.isfile(fallback_path):
                        print(f"[FALLBACK] Beralih otomatis ke klip lokal: {fallback_path}", flush=True)
                        cap = cv2.VideoCapture(fallback_path)
                        if not cap.isOpened():
                            print(f"[ERROR] Gagal membuka file fallback: {fallback_path}", flush=True)
                            break
                        is_live = False
                        used_fallback = True
                        active_source = fallback_path
                        raw_fb_fps = cap.get(cv2.CAP_PROP_FPS)
                        fps_in = raw_fb_fps if (raw_fb_fps and 1.0 <= raw_fb_fps <= 120.0) else 25.0
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frames_to_process = min(total_frames, max_frames) if (max_frames and total_frames > 0) else total_frames
                        fallback_time_offset = time.monotonic() - start_total_time
                        local_frame_idx = 0
                        zone_tracker.clear()
                        rider_last_seen.clear()
                        continue
                    else:
                        print("[ERROR] File fallback tidak tersedia. Menghentikan pemrosesan.", flush=True)
                        break
                else:
                    break

            frame_idx += 1
            local_frame_idx += 1
            # Mode live: gunakan time.monotonic() agar dwell_seconds akurat meski FPS stream fluktuatif
            if is_live:
                video_time = time.monotonic() - start_total_time
            else:
                video_time = fallback_time_offset + (local_frame_idx / fps_in)
            t0 = time.time()

            # Tracking frame dengan YOLO (persist=True, tracker=tracker_config_file, iou=iou_thresh)
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_config_file,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                classes=target_class_ids,
                device=device,
                verbose=False
            )

            # Ekstraksi deteksi
            persons = []
            two_wheelers = []
            large_vehicles = []

            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    coords = boxes.xyxy[i].cpu().numpy().tolist()
                    cls_id = int(boxes.cls[i].item())
                    cls_name = model.names[cls_id]
                    conf = float(boxes.conf[i].item())
                    track_id = int(boxes.id[i].item()) if (boxes.id is not None and boxes.id[i] is not None) else None

                    x1, y1, x2, y2 = coords
                    foot_point = ((x1 + x2) / 2.0, y2)

                    det_obj = {
                        "box": [x1, y1, x2, y2],
                        "cls_name": cls_name,
                        "conf": conf,
                        "track_id": track_id,
                        "foot_point": foot_point
                    }

                    if cls_name == "person":
                        persons.append(det_obj)
                    elif cls_name in ["motorcycle", "bicycle"]:
                        two_wheelers.append(det_obj)
                    elif cls_name in ["car", "bus", "truck"]:
                        large_vehicles.append(det_obj)

            # Overlay red zone polygon (semi-transparan)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_px], (0, 0, 220))
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
            cv2.polylines(frame, [polygon_px], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            safe_count = 0
            active_violators = 0
            frame_riders_count = 0

            alerts_triggered_in_frame = []

            # Evaluasi setiap orang (Person)
            for p in persons:
                p_box = p["box"]
                foot = p["foot_point"]
                track_id = p["track_id"]

                is_rider = False
                matched_vehicle = None

                # 1. Cek terhadap motor & sepeda menggunakan padding asimetris
                for tw in two_wheelers:
                    expanded_vbox = expand_box_asymmetric(
                        tw["box"],
                        pad_up=pad_up,
                        pad_side=pad_side,
                        pad_down=pad_down,
                        frame_w=width,
                        frame_h=height
                    )
                    overlap = compute_overlap_ratio(p_box, expanded_vbox)

                    if overlap >= rider_overlap_threshold:
                        if is_point_inside_or_near_box(foot, expanded_vbox, margin=20.0):
                            is_rider = True
                            matched_vehicle = tw["cls_name"]
                            break

                # 2. Cek terhadap kendaraan besar (mobil, bus, truk):
                # TANPA padding, person hanya dianggap di dalam kendaraan jika overlap >= vehicle_overlap_threshold (0.80)
                # agar pejalan kaki di dekat atau depan mobil tidak salah dianggap rider
                if not is_rider:
                    for lv in large_vehicles:
                        car_box = lv["box"]  # Box mobil asli tanpa padding
                        overlap = compute_overlap_ratio(p_box, car_box)
                        if overlap >= vehicle_overlap_threshold:
                            is_rider = True
                            matched_vehicle = lv["cls_name"]
                            break

                # 3. Logika Sticky Rider Status (FR-04 extension)
                if is_rider:
                    if track_id is not None:
                        rider_last_seen[track_id] = video_time
                elif track_id is not None and track_id in rider_last_seen:
                    # Jika pernah jadi pengendara dan belum lewat dari rider_sticky_seconds
                    time_since_rider = video_time - rider_last_seen[track_id]
                    if 0.0 <= time_since_rider <= rider_sticky_seconds:
                        is_rider = True
                        matched_vehicle = "rider (sticky)"

                # 4. Uji Red Zone via Foot Point
                in_red_zone = cv2.pointPolygonTest(
                    polygon_px,
                    (float(foot[0]), float(foot[1])),
                    False
                ) >= 0

                is_violator = False
                dwell_time = 0.0

                if in_red_zone and not is_rider:
                    if track_id is not None:
                        if track_id not in zone_tracker:
                            zone_tracker[track_id] = {
                                "entry_time": video_time,
                                "last_seen_time": video_time,
                                "last_alert_time": -999.0
                            }
                        else:
                            # Toleransi hilang sesaat karena oklusi tiang (hingga 2.0 detik)
                            if (video_time - zone_tracker[track_id]["last_seen_time"]) > 2.0:
                                zone_tracker[track_id]["entry_time"] = video_time
                            zone_tracker[track_id]["last_seen_time"] = video_time

                        dwell_time = video_time - zone_tracker[track_id]["entry_time"]

                        # 5. Evaluasi Dwell Time (FR-06)
                        if dwell_time >= dwell_seconds:
                            is_violator = True
                            active_violators += 1

                            # 6. Evaluasi Cooldown & Alert Triggering
                            last_alert = zone_tracker[track_id]["last_alert_time"]
                            if (video_time - last_alert) >= cooldown_seconds:
                                zone_tracker[track_id]["last_alert_time"] = video_time
                                total_incidents += 1
                                alerts_triggered_in_frame.append({
                                    "track_id": track_id,
                                    "foot_point": foot,
                                    "dwell_time": dwell_time
                                })
                    else:
                        dwell_time = 0.0
                else:
                    if is_rider:
                        frame_riders_count += 1

                # Visualisasi bounding box & label sesuai status
                px1, py1, px2, py2 = map(int, p_box)

                if is_violator:
                    # STATUS: PELANGGAR TERKONFIRMASI (Merah)
                    box_color = (0, 0, 255)
                    foot_color = (0, 0, 255)
                    thickness = 3
                    label_text = f"ID:{track_id} [PELANGGAR {dwell_time:.1f}s]"
                elif in_red_zone and not is_rider:
                    # STATUS: DALAM RED ZONE TETAPI BELUM MEMENUHI DWELL TIME (Kuning/Aman)
                    safe_count += 1
                    box_color = (0, 215, 255)  # Kuning
                    foot_color = (0, 215, 255)
                    thickness = 2
                    label_text = f"ID:{track_id} [ZONA {dwell_time:.1f}s]"
                elif is_rider:
                    # STATUS: PENGENDARA KENDARAAN (Hijau)
                    safe_count += 1
                    box_color = (0, 230, 0)
                    foot_color = (0, 255, 0)
                    thickness = 2
                    label_text = f"ID:{track_id} [{matched_vehicle}]"
                else:
                    # STATUS: PEJALAN KAKI DI LUAR ZONA (Hijau)
                    safe_count += 1
                    box_color = (0, 230, 0)
                    foot_color = (0, 255, 0)
                    thickness = 2
                    label_text = f"ID:{track_id} [AMAN]"

                # Gambar kotak & titik kaki
                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, thickness)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 5, foot_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                # Label tag
                (tw_label, th_label), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (px1, max(0, py1 - 18)), (px1 + tw_label + 4, max(18, py1)), box_color, -1)
                cv2.putText(frame, label_text, (px1 + 2, max(14, py1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Hitung waktu komputasi & FPS
            dt = time.time() - t0
            processing_times.append(dt)
            fps_instant = 1.0 / dt if dt > 0 else 0.0
            fps_avg = len(processing_times) / sum(processing_times)

            # Tangani snapshot dan CSV untuk setiap alert yang terpicu pada frame ini
            for alert in alerts_triggered_in_frame:
                a_id = alert["track_id"]
                a_foot = alert["foot_point"]
                a_dwell = alert["dwell_time"]

                snapshot_filename = f"{source_stem}_frame_{frame_idx:05d}_id_{a_id}.jpg"
                snapshot_rel_path = os.path.join("outputs", "snapshots", snapshot_filename)

                if save_snapshots:
                    save_alert_snapshot(frame, snapshot_rel_path, video_time, frame_idx, a_id, a_dwell)
                    total_snapshots_saved += 1

                time_str = format_timestamp(video_time)
                csv_writer.writerow([
                    time_str,
                    frame_idx,
                    a_id,
                    round(a_foot[0], 1),
                    round(a_foot[1], 1),
                    round(a_dwell, 2),
                    active_source if is_live else os.path.basename(active_source),
                    snapshot_rel_path
                ])
                csv_file.flush()
                total_csv_rows_written += 1

                print(f"[ALERT BARU #{total_incidents}] Waktu: {time_str} | Frame: {frame_idx:5d} | Track ID: {a_id:3d} | "
                      f"Dwell: {a_dwell:.2f}s | Snapshot: {snapshot_filename}", flush=True)

            # Gambar HUD di layar
            draw_hud(frame, frame_idx, frames_to_process, video_time, fps_instant, fps_avg,
                     safe_count, active_violators, total_incidents, frame_riders_count, model_name,
                     is_live=is_live, used_fallback=used_fallback)

            # Tulis ke video file jika diaktifkan
            if out_writer is not None:
                out_writer.write(frame)

            if show:
                cv2.imshow("Red Zone CCTV Preview", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty("Red Zone CCTV Preview", cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Pemrosesan dihentikan pengguna.", flush=True)
                    break

            if is_live:
                if frame_idx % 50 == 0:
                    elapsed = time.monotonic() - start_total_time
                    time_str = format_timestamp(video_time)
                    print(f"[LIVE Frame {frame_idx:5d}] Time: {time_str} | FPS: {fps_avg:5.2f} | "
                          f"Pelanggar Aktif: {active_violators} | Total Alert: {total_incidents} | Elapsed: {elapsed:5.1f}s", flush=True)
            elif frame_idx % 250 == 0 or (frames_to_process > 0 and frame_idx == frames_to_process):
                elapsed = time.monotonic() - start_total_time
                eta = (frames_to_process - frame_idx) / fps_avg if (fps_avg > 0 and frames_to_process > 0) else 0
                time_str = format_timestamp(video_time)
                print(f"[{frame_idx:5d}/{frames_to_process}] VideoTime: {time_str} | FPS: {fps_avg:5.2f} | "
                      f"Pelanggar Aktif: {active_violators} | Total Alert: {total_incidents} | "
                      f"Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s", flush=True)

    finally:
        if stream_reader is not None:
            stream_reader.stop()
        cap.release()
        if out_writer is not None:
            out_writer.release()
        csv_file.close()
        if show:
            cv2.destroyAllWindows()

    total_time = time.monotonic() - start_total_time
    avg_fps = len(processing_times) / sum(processing_times) if processing_times else 0.0

    print("\n" + "=" * 65)
    print("HASIL PEMROSESAN VIDEO (TEROPTIMASI)")
    print("=" * 65)
    print(f"Sumber Diminta           : {source_path}")
    print(f"Sumber Aktif Akhir       : {active_source} ({'LIVE' if is_live else ('FALLBACK' if used_fallback else 'FILE')})")
    print(f"File Output Video        : {output_path if should_save_video else 'Tidak disimpan'}")
    print(f"File Log CSV             : {csv_log_path}")
    print(f"Model Digunakan          : {model_name}")
    print(f"Total Frame Diproses     : {frame_idx}")
    print(f"Total Waktu Komputasi    : {total_time:.2f} detik ({total_time / 60:.2f} menit)")
    print(f"FPS Rata-rata            : {avg_fps:.2f}")
    print(f"Total Alert (Kejadian)   : {total_incidents}")
    print(f"Total Snapshot Disimpan  : {total_snapshots_saved}")
    print(f"Total Baris Log CSV      : {total_csv_rows_written}")
    print(f"Integritas 1-1-1 Alert   : {'VALID (1 Alert = 1 Snapshot = 1 CSV)' if total_incidents == total_snapshots_saved == total_csv_rows_written else 'MISMATCH'}")
    print("=" * 65 + "\n", flush=True)

    return {
        "source": source_path,
        "active_source": active_source,
        "used_fallback": used_fallback,
        "output_video": output_path if should_save_video else None,
        "csv_log": csv_log_path,
        "model": model_name,
        "frames": frame_idx,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "total_incidents": total_incidents,
        "snapshots_saved": total_snapshots_saved,
        "csv_rows_written": total_csv_rows_written
    }


def main():
    args = parse_args()
    run_pipeline(
        source_path=args.source,
        config_path=args.config,
        output_path=args.output,
        max_frames=args.max_frames,
        device=args.device,
        show=args.show,
        fallback_path=args.fallback,
        save_video=True if args.save_video else None,
        no_save_video=args.no_save_video,
        max_seconds=args.max_seconds,
        model_override=args.model,
        imgsz_override=args.imgsz
    )


if __name__ == "__main__":
    main()
