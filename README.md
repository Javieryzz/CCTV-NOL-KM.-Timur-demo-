# Red Zone Detection — CCTV Nol Km Timur (Demo)

Prototype *computer vision* untuk mendeteksi pejalan kaki yang berada di area badan jalan raya (*red zone*) pada CCTV publik **Nol Km - Timur, Yogyakarta** menggunakan model YOLO pretrained dan ByteTrack tracking, tanpa pelatihan ulang (*zero fine-tuning*).

> Untuk panduan lengkap langkah demi langkah (dari rekam cuplikan hingga live preview CCTV), lihat **[GUIDE.md](GUIDE.md)**.  
> Dokumen kebutuhan dan spesifikasi lengkap tersedia di **[CCTV_PRD.md](CCTV_PRD.md)**.

---

## Fitur Utama

- **Deteksi & Tracking Stabil:** Menggunakan YOLO pretrained (`yolov8s.pt` / `yolov8n.pt`) dengan custom ByteTrack (`configs/tracker_custom.yaml`, `track_buffer: 60`) untuk menahan *track ID* saat oklusi tiang/kendaraan.
- **Pengecualian Pengendara (Motor vs Mobil Terpisah):**
  - `motorcycle` & `bicycle`: ekspansi kotak asimetris (+35% ke atas, +15% samping & bawah), overlap $\ge 0.20$, dan status *sticky rider* 2.0 detik agar pengendara motor tidak memicu alarm palsu.
  - `car`, `bus`, & `truck`: tanpa padding dengan syarat overlap $\ge 0.80$ sehingga pejalan kaki di dekat mobil tetap terdeteksi.
- **Red Zone Berbasis Foot Point:** Titik tengah bawah kotak orang (`x_center`, `y2`) diuji terhadap polygon rasio `0–1` di `configs/nol_km_timur.json`.
- **Dwell Time & Cooldown:** Alert hanya aktif setelah berada di *red zone* minimal `0.8` detik (`dwell_seconds`) dengan jeda `10` detik per `track_id` (`cooldown_seconds`).
- **Integritas 1-1-1 Alert:** Setiap kejadian menghasilkan 1 banner peringatan, 1 file snapshot (dengan timestamp video), dan 1 baris log CSV di `outputs/logs/`.
- **Mode Live Stream HLS + Fallback Otomatis:** Mendukung URL `.m3u8` langsung dengan *threaded frame grabber*, auto-reconnect, perhitungan *dwell time* berbasis `time.monotonic()`, preview `1280×720`, serta fallback otomatis ke klip lokal jika stream putus.

---

## Struktur Proyek

```text
├── configs/
│   ├── nol_km_timur.json      # Konfigurasi polygon red_zone, model, threshold, dwell & cooldown
│   └── tracker_custom.yaml    # Konfigurasi ByteTrack (track_buffer=60, track_low_thresh=0.1)
├── samples/                   # Klip video uji lokal (.mp4 tidak di-commit ke repo)
├── outputs/
│   ├── videos/                # Video hasil deteksi dengan overlay
│   ├── snapshots/             # Snapshot kejadian pelanggaran (1 alert = 1 file)
│   ├── eval_frames/           # Frame ekstraksi evaluasi manual
│   └── logs/                  # Log kejadian CSV & ringkasan evaluasi
├── src/
│   ├── main.py                # Pipeline utama (file lokal & live stream HLS + fallback)
│   ├── draw_roi.py            # Alat interaktif menggambar & verifikasi polygon red_zone
│   ├── record_sample.py       # Perekam cuplikan stream CCTV via ffmpeg
│   ├── extract_eval_frames.py # Ekstraktor 30 frame merata untuk evaluasi manual
│   └── run_batch.py           # Runner evaluasi batch multi-klip
├── CCTV_PRD.md                # Product Requirements Document
├── GUIDE.md                   # Panduan demo lengkap langkah demi langkah
└── requirements.txt           # Dependensi Python
```

---

## Instalasi Cepat

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Uji pada video sampel lokal:**
   ```bash
   python src/main.py --source samples/nol_km_siang_1_16x9.mp4 --config configs/nol_km_timur.json
   ```

2. **Jalankan langsung di Live CCTV Nol Km Timur (preview ringan):**
   ```bash
   python src/main.py --source "https://cctvjss.jogjakota.go.id/malioboro/NolKm_Timur.stream/playlist.m3u8" --config configs/nol_km_timur.json --fallback samples/nol_km_siang_1_16x9.mp4 --model yolov8n.pt --imgsz 640 --show --no-save-video
   ```

Lihat **[GUIDE.md](GUIDE.md)** untuk alur demo selengkapnya (rekam sampel, gambar & verifikasi polygon ROI, uji batch, hingga live preview).

---

## Sumber Data & Etika Demo
Footage CCTV berasal dari stream publik Pemerintah Kota Yogyakarta (`cctvjss.jogjakota.go.id`). File video dan snapshot hanya digunakan untuk keperluan demo internal dan tidak disertakan dalam repository ini.
