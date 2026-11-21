# PRD: Red Zone Detection pada CCTV Kota Yogyakarta (Demo)

**Versi:** 1.1 | **Kamera:** Nol Km - Timur (cctv.jogjakota.go.id) | **Jenis:** Demo / prototype

---

## 1. Tujuan dan Ruang Lingkup

### Tujuan
Membuat demo computer vision yang mendeteksi **orang yang berada di badan jalan raya (red zone)** pada footage CCTV publik. Saat itu terjadi, bounding box orang tersebut berubah **merah** dan muncul **notifikasi alert**.

Proyek ini adalah kelanjutan dari prototype sebelumnya yang baru diuji pada satu video. Kali ini sistem diuji pada footage CCTV nyata dengan kondisi yang lebih sulit (siang/malam, ramai/sepi, banyak kendaraan).

### In-scope
- Deteksi person dan kendaraan dengan model pretrained (tanpa training).
- Satu red zone (badan jalan) untuk satu kamera.
- Pengecualian pengendara motor, sepeda, dan mobil agar tidak memicu alert.
- Tracking, dwell time, cooldown, visualisasi, snapshot kejadian, dan log CSV.
- Input berupa file video hasil rekaman.

### Out-of-scope (sengaja tidak dikerjakan)
- Training atau fine-tuning model.
- Pengecualian zebra cross (zebra cross dibiarkan masuk red zone; bisa jadi pengembangan lanjutan, lihat Langkah 7).
- Multi-kamera, dashboard web (Streamlit/Gradio), alert audio, dan notifikasi eksternal (Telegram, email, WhatsApp, dsb.).
- Blur wajah dan eskalasi level bahaya berdasarkan jarak ke kendaraan.
- Penilaian akurasi formal (mAP, IoU, dataset berlabel bounding box).

---

## 2. Kebutuhan Fungsional

| ID | Kebutuhan |
|---|---|
| FR-01 | Mendeteksi kelas `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck` menggunakan YOLO nano pretrained (COCO). |
| FR-02 | Red zone didefinisikan sebagai polygon di file config dengan **koordinat rasio** (0-1), sehingga tidak bergantung pada resolusi video. Polygon digambar dengan **margin kecil dari garis bollard** (lihat bagian 3). |
| FR-03 | Posisi orang ditentukan dari **foot point** (titik tengah bawah bounding box). Orang dianggap di red zone jika foot point berada di dalam polygon. |
| FR-04 | Orang yang **menunggangi motor/sepeda** atau berada di dalam kendaraan **tidak** dianggap pelanggar (lihat bagian 4). |
| FR-05 | Setiap orang diberi **tracking ID** yang konsisten antar frame, dengan toleransi hilang sesaat (`track_buffer`) untuk menangani oklusi tiang dan kendaraan. |
| FR-06 | **Dwell time:** alert baru aktif setelah foot point orang yang sama berada di red zone selama minimal `dwell_seconds` (dalam satuan detik waktu video, bukan jumlah frame, supaya stabil walau FPS pemrosesan berubah). **Cooldown:** alert untuk ID yang sama tidak diulang dalam `cooldown_seconds`. |
| FR-07 | Visualisasi: box **hijau** (aman), box **merah** (pelanggar), overlay polygon zona, banner alert, counter pelanggar aktif dan total, serta **FPS pemrosesan** di layar. |
| FR-08 | Setiap kejadian alert dicatat ke **CSV** (timestamp video, nomor frame, track ID, posisi foot point, durasi di zona, nama file sumber, path snapshot). |
| FR-09 | Saat alert terpicu, sistem menyimpan **satu snapshot per kejadian** (frame dengan box merah dan timestamp video tercetak) ke folder `outputs/snapshots/`. |
| FR-10 | Video hasil olahan (dengan overlay) dapat disimpan sebagai file output. |
| FR-11 | *(Opsional, dikerjakan terakhir)* Input dari URL live stream HLS, dengan fallback ke klip jika stream gagal. |

---

## 3. Penentuan Zona dan Posisi

- **Foot point, bukan seluruh box.** Dengan sudut kamera yang tinggi, kepala dan bahu orang di trotoar bisa tampak masuk ke area jalan. Karena itu hanya titik tengah bawah box (`x_center`, `y2`) yang diuji terhadap polygon.
- **Margin di tepi bollard.** Kamera ini punya deretan bollard bulat di pinggir jalan. Saat menggambar polygon, jaga jarak kecil dari garis bollard ke arah jalan supaya orang yang berdiri tepat di sisi trotoar tidak masuk zona karena selisih beberapa piksel. Verifikasi secara visual pada klip siang dan malam.
- **Zebra cross** pada versi ini ikut masuk red zone (lihat bagian 1). Hal ini perlu dijelaskan saat demo supaya tidak dianggap kesalahan deteksi.

---

## 4. Logika Pengecualian Pengendara (FR-04)

Tanpa pengecualian ini, hampir semua pengendara di jalan akan memicu alert.

**Metode awal (sederhana):**
- Untuk setiap `person`, hitung overlap dengan bounding box `motorcycle` dan `bicycle` terdekat.
- Box kendaraan diperlebar sedikit (**padding** sekitar 10%, nilai awal, disetel saat pengujian) sebelum dicek, agar pengendara yang menurunkan kaki saat berhenti di lampu merah atau ruang henti tetap terhitung.
- Gunakan rasio *intersection / luas box person*. Jika di atas ambang (nilai awal **0,4**) **dan** foot point person berada di dalam atau dekat box kendaraan yang sudah diperlebar, person dianggap **pengendara** dan diberi status aman.
- Person yang tertutup atau berada di dalam `car/bus/truck` biasanya tidak terdeteksi; jika terdeteksi dan overlap dengan box mobil tinggi, perlakukan sama.

**Kasus tepi yang perlu diketahui:**
- **Pembonceng:** orang kedua di motor mungkin hanya sebagian terlihat atau box-nya menyatu dengan pengemudi.
- **Orang berdiri di dekat motor parkir** atau menuntun motor: bisa salah dianggap pengendara (false negative).
- **Motor tidak terdeteksi** (kecil, jauh, malam hari): pengendaranya terhitung sebagai pejalan kaki (false positive).
- **Kendaraan saling menutupi** di lalu lintas padat.
- **Oklusi tiang** (tiang lampu dan rambu): box hilang sesaat dan ID bisa berganti, sehingga alert berulang. Ditangani dengan `track_buffer`, dwell time, dan cooldown.

Kasus-kasus ini tidak ditargetkan selesai sempurna, tetapi dicatat sebagai batas kemampuan sistem.

---

## 5. Pipeline dan Struktur Proyek

### Alur pipeline
```
Video/Stream → Deteksi (YOLO nano) → Tracking → Asosiasi person-kendaraan
→ Cek foot point di polygon → Dwell time + cooldown
→ Visualisasi + Alert + Snapshot + Log CSV
```

### Struktur folder (usulan)
```
red-zone-cctv/
├── configs/
│   └── nol_km_timur.json
├── samples/
│   ├── nol_km_timur_siang_01.mp4
│   ├── nol_km_timur_malam_01.mp4
│   └── ...
├── outputs/
│   ├── videos/
│   ├── snapshots/
│   └── logs/
├── src/
└── PRD.md
```

### Contoh config
```json
{
  "camera": "nol_km_timur",
  "red_zone": [[0.18, 0.42], [0.92, 0.38], [0.97, 0.62], [0.30, 0.68]],
  "classes": ["person", "bicycle", "motorcycle", "car", "bus", "truck"],
  "conf": 0.35,
  "imgsz": 960,
  "track_buffer": 30,
  "rider_overlap_threshold": 0.4,
  "rider_box_padding": 0.10,
  "dwell_seconds": 0.8,
  "cooldown_seconds": 10,
  "save_snapshots": true,
  "show_fps": true
}
```
Nilai di atas hanyalah titik awal dan disesuaikan setelah baseline dijalankan. Koordinat polygon adalah contoh dan harus digambar ulang untuk kamera ini.

---

## 6. Cara Pengecekan Hasil

Karena model tidak dilatih, tujuan pengecekan hanya menilai **apakah demo layak**, bukan mengklaim akurasi.

**Sampel:** 4 klip berdurasi 2-3 menit: siang ramai, siang sepi, malam ramai, malam sepi. Klip direkam dari stream menggunakan ffmpeg dan dipakai untuk pengembangan serta pengecekan.

**Pengecekan manual:**
- Pilih sekitar **30 frame per kondisi** (atau 30 frame total jika waktu terbatas): campuran frame kosong, ramai, dan orang tepat di batas zona.
- Untuk tiap frame, catat berapa orang yang **sebenarnya** berada di red zone (bukan pengendara), lalu bandingkan dengan status merah dari sistem.
- Hasilkan angka sederhana per kondisi: **terdeteksi benar**, **terlewat**, dan **salah alarm**. Catat terpisah salah alarm yang berasal dari pengendara, karena itu indikator utama logika pengecualian.
- Cek juga jumlah entri log per kejadian: satu kejadian seharusnya menghasilkan satu entri (dan satu snapshot), bukan beberapa.

**Kriteria layak demo (hipotesis awal, ditinjau ulang setelah baseline):**
- Pada kondisi siang, mayoritas orang yang jelas berada di jalan terdeteksi dan alert tidak muncul untuk pengendara pada sebagian besar frame.
- Video berjalan cukup mulus dan tidak ada spam alert untuk satu kejadian yang sama.
- Kondisi yang gagal (malam ramai, orang kecil di kejauhan) didokumentasikan secara jujur sebagai batas kemampuan, bukan disembunyikan.

---

## 7. Risiko dan Batas Kemampuan

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Orang di kejauhan tampak kecil | Recall turun | Naikkan `imgsz`, turunkan `conf`, atau abaikan area terjauh dari zona |
| Kerumunan padat dan oklusi | ID switch, deteksi terlewat | `track_buffer`, dwell time, cooldown; catat sebagai batas kemampuan |
| Malam hari dan silau lampu | Deteksi tidak stabil | Evaluasi terpisah per kondisi; pertimbangkan model lebih besar dari nano |
| Motor tidak terdeteksi | Pengendara salah dianggap pejalan kaki | Setel ambang overlap dan padding, periksa visual pada klip uji |
| Polygon terlalu dekat dengan bollard | Orang di trotoar salah ditandai | Beri margin kecil, verifikasi overlay secara visual |
| FPS pemrosesan rendah di laptop | Demo tersendat | Turunkan `imgsz`, lewati sebagian frame, pantau FPS di layar |
| Stream tidak ada API resmi dan bisa putus | Demo live gagal | Mode file sebagai jalur utama, live hanya opsional |
| Privasi dan etika | Penyalahgunaan footage | Footage dan snapshot hanya untuk demo internal, tidak disebarluaskan, sebutkan sumber (Pemkot Yogyakarta) |

---

## 8. Langkah Implementasi

Setiap langkah dikerjakan dan diverifikasi sebelum lanjut.

| # | Langkah | Selesai jika |
|---|---|---|
| 0 | Rekam 4 klip sampel dari stream (siang dan malam, ramai dan sepi) | File `.mp4` terbaca OpenCV; FPS dan resolusi tercatat |
| 1 | Jalankan pipeline prototype lama pada satu klip (baseline) | Video output tersimpan; FPS pemrosesan dan masalah utama tercatat |
| 2 | Buat config baru dan gambar polygon red zone untuk kamera ini (dengan margin dari bollard) | Overlay zona sesuai badan jalan pada satu frame siang dan satu frame malam |
| 3 | Tambahkan deteksi kendaraan dan logika pengecualian pengendara (dengan padding box motor) | Pengendara motor yang melintas tidak berubah merah pada klip siang |
| 4 | Sesuaikan tracking (`track_buffer`), dwell time, cooldown, banner, FPS di layar, snapshot, dan log CSV | Satu kejadian menghasilkan satu alert, satu snapshot, dan satu baris CSV |
| 5 | Pengecekan manual 30 frame per kondisi | Tabel hasil per kondisi dan catatan batas kemampuan |
| 6 | *(Opsional)* Mode live `--source URL` dengan fallback ke klip | Stream terbaca; jika gagal otomatis beralih ke klip |
| 7 | *(Opsional)* Pengecualian zebra cross sebagai safe zone | Orang di zebra cross tidak memicu alert |

---

## 9. Pertanyaan Terbuka

- Apakah folder proyek dilanjutkan dari `red-zone-demo` atau dibuat folder baru?
- Berapa FPS pemrosesan yang cukup untuk demo (perlu diketahui setelah Langkah 1)?
- Perlukah zebra cross dikecualikan setelah demo dasar berjalan?
