# IndoBERT + HMM untuk Deteksi Produk Counterfeit

Proyek ini mengimplementasikan sistem deteksi produk *counterfeit* (tiruan) berbasis sekuens sentimen ulasan konsumen pada marketplace (Shopee, Lazada). Model menggunakan kombinasi **IndoBERT** untuk mengklasifikasi sentimen tingkat ulasan dan **Hidden Markov Model (HMM) Supervised** dengan **Viterbi Decoding** untuk memprediksi label toko/produk secara sekuensial.

## Arsitektur Sistem

```
Ulasan mentah  ->  Preprocessing  ->  IndoBERT (klasifikasi sentimen per ulasan)
                                            |
                                            v
                              Sekuens sentimen per toko/produk (urut waktu)
                                            |
                                            v
                          HMM supervised (hidden state = label asli/counterfeit)
                                            |
                                            v
                        Viterbi decoding -> prediksi label toko/produk
                                            |
                                            v
                    Evaluasi (accuracy, F1, confusion matrix, ablation study)
```

---

## Struktur Folder Project

```
counterfeit-detection/
├── data/
│   ├── raw/                  # Hasil scraping mentah per platform (.csv)
│   ├── interim/              # Data setelah cleaning & sentimen labeling
│   └── processed/            # Data train/val/test final (split & sequence)
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # Cleaning, normalisasi teks, & parsing tanggal
│   │   └── split.py          # Split data secara kronologis per toko (70:15:15)
│   ├── sentiment/
│   │   ├── dataset.py        # Dataset class PyTorch untuk tokenisasi IndoBERT
│   │   ├── train_indobert.py # Script fine-tuning IndoBERT (opsional)
│   │   └── infer_indobert.py # Sentiment labeling untuk seluruh ulasan
│   ├── sequence/
│   │   └── build_sequences.py# Agregasi ulasan -> sekuens sentiment temporal
│   ├── hmm/
│   │   ├── supervised_hmm.py # Estimasi parameter HMM (pi, A, B) via counting
│   │   └── decode.py         # Implementasi manual Algoritma Viterbi
│   ├── evaluation/
│   │   ├── metrics.py        # Metrik evaluasi & plotting confusion matrix
│   │   └── ablation.py       # Perbandingan IndoBERT-only vs IndoBERT+HMM & uji statistik
│   └── utils/
│       └── config.py         # Hyperparameter, path, dan label mapping
├── models/
│   ├── indobert_finetuned/   # Checkpoint model IndoBERT (jika di-fine-tune)
│   └── hmm_params.json       # Berkas parameter hasil pelatihan HMM (pi, A, B)
├── results/
│   ├── figures/              # Plot Confusion Matrix
│   └── reports/              # Laporan hasil Ablation Study
├── requirements.txt          # Daftar dependensi library Python
└── README.md                 # Petunjuk penggunaan proyek ini
```

---

## Dependensi Library

Pastikan Python 3.9+ sudah terpasang. Pasang dependensi menggunakan:
```bash
pip install -r requirements.txt
```

Dependensi utama:
- `transformers`
- `torch`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`
- `tqdm`
- `matplotlib`
- `seaborn`

---

## Langkah Menjalankan Pipeline (End-to-End)

Jalankan seluruh modul secara berurutan:

### 1. Preprocessing Data
Dibersihkan dari duplikasi, emoji, dan tanggal diseragamkan:
```bash
python3 src/data/preprocessing.py
```

### 2. Dataset Splitting
Ulasan dibagi menjadi 70% Train, 15% Val, dan 15% Test secara kronologis (waktu ulasan) per toko untuk mencegah data leakage:
```bash
python3 src/data/split.py
```

### 3. Sentiment Inference (IndoBERT)
Jalankan inferensi sentimen ulasan. Menggunakan model HuggingFace `mdhugol/indonesia-bert-sentiment-classification` secara default jika model lokal belum dilatih:
```bash
python3 src/sentiment/infer_indobert.py
```

### 4. Ekstraksi Sekuens
Gabungkan hasil sentimen ulasan menjadi sekuens temporal per toko dengan pembagian windowing ulasan (default: 10 ulasan per sekuens):
```bash
python3 src/sequence/build_sequences.py
```

### 5. Training HMM (Supervised)
Estimasi parameter HMM ($\pi$, $A$, $B$) lewat metode counting langsung dari data berlabel dengan Laplace smoothing:
```bash
python3 src/hmm/supervised_hmm.py
```

### 6. Evaluasi & Ablation Study
Memprediksi data uji menggunakan Algoritma Viterbi dan membandingkan performa model IndoBERT+HMM dengan baseline IndoBERT-only (voting sederhana), serta melakukan uji statistik McNemar:
```bash
python3 src/evaluation/ablation.py
```

Hasil laporan akhir akan disimpan ke berkas `results/reports/ablation_study_report.md` dan plot confusion matrix disimpan ke folder `results/figures/`.
