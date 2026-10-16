# Pipeline Deteksi Skor Kecurigaan Produk Counterfeit (Tugas Akhir)

Repositori ini berisi kode sumber, notebook eksperimen, dan dokumentasi pipeline riset Tugas Akhir untuk analisis skor kecurigaan produk counterfeit dari ulasan e-commerce (Shopee) menggunakan **IndoBERT**, **Hidden Markov Model (HMM)**, dan statistik sentimen.

---

## 📁 Struktur Repositori

```text
.
├── README.md                 # Dokumentasi utama repositori
├── .gitignore                # Aturan ignoransi file cache, debug, data, & output
├── requirements.txt          # Paket dependensi Python
├── src/                      # Source code Python modular
│   ├── __init__.py
│   ├── config.py             # Konfigurasi path & hiperparameter eksperimen
│   ├── preprocessing.py      # Cleaning, dedup, & normalisasi teks ulasan
│   └── sentiment.py          # Training & inferensi IndoBERT SMSA
├── notebooks/                # Jupyter Notebooks eksperimen & pemrosesan
│   ├── 01_clean_process_data.ipynb
│   └── 02_pipeline_llm_hmm_xgb.ipynb
├── docs/                     # Dokumentasi riset & alur pipeline
│   └── pipeline.md           # Diagram alur & penjelas desain riset
├── data/                     # Direktori dataset
│   ├── raw/                  # Data mentah CSV ulasan produk
│   └── processed/            # Data hasil pembersihan/ekstraksi fitur (Parquet)
└── outputs/                  # Cache keluaran model, log, & checkpoint (di-gitignore)
```

---

## ⚙️ Persyaratan Sistem & Instalasi

### 1. Prasyarat
- Python 3.9+
- CUDA (Opsional, disarankan untuk akselerasi inferensi/training IndoBERT dengan PyTorch)

### 2. Instalasi Dependensi
Jalankan perintah berikut untuk menginstal dependensi yang dibutuhkan:
```bash
pip install -r requirements.txt
```

---

## 🚀 Alur Penggunaan

### 1. Konfigurasi Eksperimen
Semua konfigurasi (path dataset, hiperparameter IndoBERT, komponen HMM, serta direktori output) dikelola terpusat di [`src/config.py`](src/config.py).

Panjang inferensi IndoBERT mengikuti `ExperimentConfig.sentiment_max_length` (128), sama dengan fine-tuning. Kapasitas arsitektur 512 merupakan batas posisi model. CLI `src.run_sentiment_inference` dan notebook `notebooks/inferensi_rm1.ipynb` membaca konfigurasi tersebut. Penyimpangan panjang membutuhkan `--allow-length-mismatch`; peringatan dan keputusan dicatat dalam metadata. Nilai di atas `max_position_embeddings` tetap ditolak.

Pada v4, 123 dari 73.184 ulasan yang diinferensi terpotong di 128 token (0,17%). Prediksi v5 mewarisi hasil v4 dengan pembaruan timestamp melalui `review_id`, tanpa inferensi ulang. Provenance v5 lama tidak memiliki field panjang inferensi; buktinya berasal dari metadata v4 dan kecocokan kolom prediksi. v2 berjalan dengan 512, digantikan oleh keputusan 128; angka historis v2 tetap dipertahankan.

### 2. Jalankan Pipeline

Cleaning resmi dijalankan dengan `python -m src.cleaning.run` dan menghasilkan satu file `"outputs/cleaned/reviews_clean.parquet"` beserta laporan di `"outputs/cleaned/reports/"`. Baca hasil melalui `from src.cleaning import load_clean_reviews`, lalu `reviews = load_clean_reviews()`; secara bawaan hanya baris yang layak inferensi dikembalikan, terurut per produk dan `seq_in_product`. Gunakan `only_included=False` untuk seluruh baris. Pipeline lama `DataPreprocessor` dinonaktifkan untuk folder berformat baru agar hasil tidak tercampur. Rincian ada di [laporan cleaning](docs/laporan_cleaning.md).

Gunakan modul Python dari `src/` atau jalankan notebook di folder `notebooks/`:
- `notebooks/clean-process-data.ipynb`: Pembersihan awal dataset ulasan produk.
- `notebooks/pipeline_llm_hmm_xgb.ipynb`: Pipeline inferensi IndoBERT, estimasi HMM, ekstraksi fitur, dan pemodelan skor kecurigaan.

### 3. Dokumentasi Arsitektur
Detail arsitektur pipeline, desain eksperimen (Model A / B / C), serta justifikasi metrik evaluasi dapat dilihat pada dokumen [`docs/pipeline.md`](docs/pipeline.md).

---

## 🛡️ Kebijakan Data & Kerahasiaan (GitIgnore)
File besar seperti dataset CSV/Parquet, cache Python (`__pycache__`), direktori `debug_*`, serta checkpoint model (`outputs/`, `models/`) tidak di-commit ke Git.
