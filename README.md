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

### 2. Jalankan Pipeline
Gunakan modul Python dari `src/` atau jalankan notebook di folder `notebooks/`:
- `notebooks/01_clean_process_data.ipynb`: Pembersihan awal dataset ulasan produk.
- `notebooks/02_pipeline_llm_hmm_xgb.ipynb`: Pipeline inferensi IndoBERT, estimasi HMM, ekstraksi fitur, dan pemodelan skor kecurigaan.

### 3. Dokumentasi Arsitektur
Detail arsitektur pipeline, desain eksperimen (Model A / B / C), serta justifikasi metrik evaluasi dapat dilihat pada dokumen [`docs/pipeline.md`](docs/pipeline.md).

---

## 🛡️ Kebijakan Data & Kerahasiaan (GitIgnore)
File besar seperti dataset CSV/Parquet, cache Python (`__pycache__`), direktori `debug_*`, serta checkpoint model (`outputs/`, `models/`) tidak di-commit ke Git.
