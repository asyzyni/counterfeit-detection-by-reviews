# Laporan Eksekusi RM 1

Tanggal eksekusi: 3 Oktober 2026

## 1. Struktur data dan path

Folder sumber yang diminta dapat diakses di `/Users/asyzyni/Desktop/TA /Kode dan Eksperimen/Data TA`. Saat diperiksa sebelum run, folder tersebut berisi 593 CSV mentah secara langsung di root dan tidak memiliki subfolder, hasil `cleaned_reviews`, manifest, maupun artefak model/checkpoint. Pemeriksaan mencari `config.json`, `model.safetensors`, `pytorch_model.bin`, dan nama checkpoint, tetapi tidak menemukan model tersimpan.

Kode proyek berada di `/Users/asyzyni/Desktop/TA /Kode dan Eksperimen/Kode TA`:

- Input raw: `/Users/asyzyni/Desktop/TA /Kode dan Eksperimen/Data TA` (593 CSV).
- Parquet hasil cleaning: `/Users/asyzyni/Desktop/TA /Kode dan Eksperimen/Kode TA/outputs/cleaned/` (folder per produk, part parquet per file sumber).
- Manifest cleaning: `/Users/asyzyni/Desktop/TA /Kode dan Eksperimen/Kode TA/outputs/logs/cleaning_manifest.csv`.
- Modul inference: `src/inference.py`.
- Smoke test: `tests/test_inference_smoke.py`.

Karena `Data TA` hanya berisi CSV datar dan tidak punya pola folder cleaned, hasil cleaning disimpan pada jalur `outputs/cleaned` yang sudah ditetapkan oleh `ExperimentConfig` proyek.

## 2. Perubahan kode

- `src/preprocessing.py:1080-1119`: `clean_all()` kini menambahkan satu hasil per file, kemudian memanggil `save_manifest(results)` sekali setelah loop. Sebelumnya manifest juga ditulis ulang pada setiap iterasi.
- `src/preprocessing.py:935-1019`: saat kode diperiksa, agregasi `total_input`, `total_output`, `part_number`, metadata JSON, marker selesai, dan return sudah berada setelah loop `for chunk in reader`. Penghitungan dilakukan lintas seluruh chunk. Bagian ini sudah sesuai, sehingga tidak perlu diubah.
- `src/preprocessing.py:1372-1381` dan `1510-1511`: `selected_files` sudah dibentuk dari `indices`, lalu dipakai dalam loop; fungsi mengembalikan `(report, sample)`, sesuai anotasi return. Bagian ini sudah sesuai saat diperiksa.
- `src/config.py:80`: ditambahkan `sentiment_max_length = 128`. Run awal terhenti karena `validate()` memakai atribut ini tetapi field belum ada. Penambahan ini membuat konfigurasi valid untuk preprocessing dan inference/training yang sudah membaca panjang maksimum; implementasi trainer tidak diubah.
- `src/inference.py`: menambahkan `predict_sentiment_probs(df, model_dir, text_col="text_light", batch_size=32)`. Model dan tokenizer dimuat dari direktori lokal; tidak ada kolom label yang dibutuhkan. Fungsi membuat softmax per batch, mempertahankan baris dan urutan DataFrame, mempertahankan kolom metadata, serta menaruh NaN pada probabilitas baris dengan teks kosong/NaN sambil mencatat jumlahnya ke log.
- Pemetaan label membaca `config.id2label`, menormalisasi label Indonesia/Inggris, dan menolak konfigurasi yang bukan tepat tiga kelas sentimen. Logit dipetakan ke urutan `negative`, `neutral`, `positive` berdasarkan nama kelas, bukan urutan indeks yang diasumsikan. Assertion di `resolve_sentiment_indices()` memeriksa himpunan kelas dan indeks.

## 3. Hasil `clean_all()` penuh

Eksekusi menggunakan interpreter `/Users/asyzyni/.global-venv/bin/python`, dengan `ExperimentConfig(raw_data_dir=Path('../Data TA').resolve())`.

| Ukuran | Hasil |
|---|---:|
| CSV input | 593 |
| Entri manifest | 593 |
| Status sukses / skip / gagal | 593 / 0 / 0 |
| Produk berhasil dibersihkan | 593 |
| Baris input | 303.687 |
| Baris output cleaned | 15.783 |
| Baris terbuang | 287.904 |
| Part parquet | 593 |

Pemeriksaan semua parquet menemukan 0 teks kosong/NaN dan 0 `review_id` hilang. QA pada sampel 10 part (239 baris) menemukan 0 review hilang, 0 `text_light` hilang, 0 duplikasi ID, dan 0 duplikasi review dalam produk. Semua 10 part sampel memiliki timestamp valid.

## 4. Smoke test inference

Test yang dibuat berisi 10 kalimat sentimen Indonesia. Test pemetaan kelas dijalankan dan lulus; model smoke test tidak menghasilkan prediksi karena tidak ada model fine-tuning tersimpan untuk dimuat.

Kalimat yang disiapkan:

1. “Barangnya bagus banget, sesuai foto, recommended!”
2. “Kecewa, barang rusak, tidak sesuai deskripsi.”
3. “Barang sampai, belum dicoba.”
4. “Kualitasnya mantap, pengiriman cepat dan aman.”
5. “Produk cacat, layar tidak menyala sejak dibuka.”
6. “Paket diterima oleh saya pagi ini.”
7. “Sangat puas, berfungsi dengan baik dan harganya murah.”
8. “Buruk sekali, cepat panas dan baterainya boros.”
9. “Warna hitam, kapasitas memori 128 GB.”
10. “Tidak rekomendasi, sering mati sendiri setelah dipakai.”

Probabilitas aktual untuk semua kalimat: **tidak tersedia**. Model belum ditemukan; memberi nilai dari model dummy atau model dasar yang bukan hasil fine-tuning akan melanggar instruksi RM 1. Hasil test: `1 passed, 1 skipped`; test inference diskalakan hanya setelah `INDOBERT_SENTIMENT_MODEL_DIR` menunjuk ke model tersimpan.

## 5. Hasil full inference dan blocker

**Full inference belum dijalankan.** Tidak ada direktori model/checkpoint fine-tuned di dalam `Data TA`, sehingga tidak ada `model_dir` sah untuk smoke test atau prediksi korpus. Karena smoke test model belum dapat lolos, prediksi untuk 15.783 baris cleaned tidak dibuat.

Akibatnya, belum tersedia:

- Parquet probabilitas dengan skema `review_id, product_id, timestamp, text_light, p_negative, p_neutral, p_positive`.
- Jumlah teks yang ter-skip oleh inference, statistik probabilitas seluruh korpus, dan lima contoh baris prediksi.
- Sepuluh vektor probabilitas aktual dari smoke test.

Untuk melanjutkan tanpa mengubah kode, letakkan model hasil fine-tuning beserta tokenizer/config hasil `save()` di direktori lokal, lalu jalankan smoke test dengan variabel `INDOBERT_SENTIMENT_MODEL_DIR` menunjuk ke direktori tersebut. Setelah smoke test berhasil ditinjau, jalankan inference pada part cleaned dan gabungkan output per produk ke struktur parquet proyek.

Perintah test:

```bash
INDOBERT_SENTIMENT_MODEL_DIR="/path/ke/model-finetuned" \
  /Users/asyzyni/.global-venv/bin/python -m pytest tests/test_inference_smoke.py -q -s
```
