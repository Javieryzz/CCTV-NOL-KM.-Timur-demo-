# Panduan Demo: Red Zone Detection CCTV Nol Km - Timur

Panduan langkah demi langkah untuk menjalankan demo dari nol: mulai dari merekam cuplikan CCTV, menggambar dan memverifikasi polygon *red zone*, menguji klip sampel lokal, hingga menjalankan deteksi secara *live* langsung dari stream CCTV.

> Jalankan seluruh perintah dari **root direktori proyek**.

---

## 1. Persiapan Environment

```bash
pip install -r requirements.txt
```

*(Opsional)* Jika `ffmpeg` belum terpasang di PATH sistem dan Anda ingin merekam sampel via `src/record_sample.py`, pasang `imageio-ffmpeg`:

```bash
pip install imageio-ffmpeg
```

Bobot model YOLO (`yolov8s.pt` / `yolov8n.pt`) akan diunduh otomatis oleh `ultralytics` saat skrip pertama kali dijalankan.

---

## 2. Ambil Cuplikan Video CCTV (`src/record_sample.py`)

Rekam klip sampel (misalnya 150 detik) dari stream HLS CCTV Nol Km - Timur ke folder `samples/`. Skrip akan otomatis memverifikasi resolusi, FPS, dan jumlah frame menggunakan OpenCV setelah selesai.

```bash
python src/record_sample.py \
  --url "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/playlist.m3u8" \
  --duration 150 \
  --output samples/nol_km_siang_1_16x9.mp4
```

Anda dapat mengulangi perintah di atas untuk merekam kondisi lain (misalnya `samples/nol_km_malam_1_16x9.mp4`).

---

## 3. Gambar & Verifikasi Red Zone (`src/draw_roi.py`)

### A. Menggambar Polygon Red Zone Secara Interaktif

Buka satu frame dari video sampel untuk mengklik titik-titik batas badan jalan (*red zone*). Koordinat disimpan otomatis sebagai **rasio 0–1** ke `configs/nol_km_timur.json`.

```bash
python src/draw_roi.py --video samples/nol_km_siang_1_16x9.mp4 --config configs/nol_km_timur.json --frame-idx 100
```

**Kontrol di jendela gambar:**
- **Klik kiri:** Tambah titik sudut polygon (beri jarak margin kecil dari garis bollard trotoar).
- **Klik kanan / `u`:** Hapus titik terakhir (*undo*).
- **`c`:** Reset seluruh titik.
- **`s` / `Enter`:** Simpan koordinat ke `configs/nol_km_timur.json`.
- **`q` / `Esc`:** Keluar tanpa menyimpan.

### B. Verifikasi Overlay Polygon

Pastikan polygon menempel tepat di area badan jalan pada kondisi siang maupun malam tanpa membuka jendela interaktif:

```bash
python src/draw_roi.py --video samples/nol_km_siang_1_16x9.mp4 --config configs/nol_km_timur.json --verify --output-verify outputs/verify_siang.jpg
python src/draw_roi.py --video samples/nol_km_malam_1_16x9.mp4 --config configs/nol_km_timur.json --verify --output-verify outputs/verify_malam.jpg
```

Buka `outputs/verify_siang.jpg` dan `outputs/verify_malam.jpg` untuk mengecek posisi polygon.

---

## 4. Uji pada Sampel Video Lokal (`src/main.py`)

Jalankan pipeline deteksi, ByteTrack tracking, pengecualian pengendara, *dwell time* (0.8 detik), dan *cooldown* (10 detik) pada satu video sampel:

```bash
python src/main.py --source samples/nol_km_siang_1_16x9.mp4 --config configs/nol_km_timur.json
```

Tambahkan `--show` jika ingin melihat preview real-time saat pemrosesan berlangsung, atau `--max-frames 200` untuk uji cepat.

### Hasil yang Dihasilkan
Setiap pemrosesan file lokal menghasilkan:
1. **Video overlay:** `outputs/videos/<nama_klip>_result.mp4`
2. **Snapshot kejadian (1 alert = 1 snapshot):** `outputs/snapshots/<nama_klip>_frame_<no>_id_<track_id>.jpg`
3. **Log kejadian CSV (FR-08):** `outputs/logs/<nama_klip>_events.csv`

### Uji Batch & Ekstraksi Frame Evaluasi Manual (Opsional)
- Ekstrak 30 frame merata per klip hasil untuk evaluasi manual (`outputs/eval_frames/` & `outputs/logs/manual_eval.csv`):
  ```bash
  python src/extract_eval_frames.py
  ```
- Jalankan batch evaluasi multi-klip:
  ```bash
  python src/run_batch.py
  ```

---

## 5. Menjalankan Langsung di Live Stream CCTV

Sistem mendukung input URL HLS langsung dengan **threaded frame grabber** (anti-delay), **auto-reconnect** saat jeda segmen HLS, perhitungan `dwell_seconds` berbasis `time.monotonic()`, jendela preview 1280×720 (`--show`), tanpa menulis video ke disk (`--no-save-video`), serta **fallback otomatis** ke klip lokal jika stream benar-benar putus.

### Mode Live Ringan & Stabil (Direkomendasikan di Laptop/CPU)
Menggunakan `--model yolov8n.pt --imgsz 640` agar FPS tinggi dan responsif:

```bash
python src/main.py --source "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/playlist.m3u8" --config configs/nol_km_timur.json --fallback samples/nol_km_siang_1_16x9.mp4 --model yolov8n.pt --imgsz 640 --show --no-save-video
```

### Mode Live Akurasi Standar (Sesuai Config `yolov8s.pt`, `imgsz=960`)

```bash
python src/main.py --source "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/playlist.m3u8" --config configs/nol_km_timur.json --fallback samples/nol_km_siang_1_16x9.mp4 --show --no-save-video
```

**Kontrol & Catatan Mode Live:**
- Tekan **`q`** pada jendela `Red Zone CCTV Preview` (atau tutup jendela) untuk berhenti.
- Gunakan `--max-seconds 30` jika ingin membatasi durasi eksekusi otomatis.
- Jika stream gagal dibuka atau terputus permanen setelah 5x percobaan reconnect, sistem otomatis beralih ke file `--fallback` tanpa crash.
- Snapshot alert dan log CSV (`outputs/logs/nol_km_timur_live_events.csv`) tetap tercatat meskipun `--no-save-video` aktif.
