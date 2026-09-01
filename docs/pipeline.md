# Pipeline: Suspicion Score Produk Counterfeit dari Ulasan

Diagram ini merepresentasikan pipeline final hasil revisi: TF-IDF + leksikon
menghasilkan **skor kecurigaan kontinu** (bukan label biner genuine/counterfeit),
sementara IndoBERT + HMM menyediakan fitur dinamika sentimen temporal yang diuji
apakah menambah daya prediksi di atas statistik sentimen statis.

```
                        RAW REVIEW CSV
                              │
                              ▼
                     1. PREPROCESSING
              (cleaning, dedup, normalisasi teks)
                              │
                              ▼
                       cleaned_reviews
        (review_id, product_id, timestamp, text)
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
   2A. INDOBERT INFERENCE               2B. TF-IDF + LEKSIKON
   p_negative | p_neutral | p_positive  representasi TF-IDF ulasan
            │                                   │
            ▼                                   ▼
     sort per produk                    similarity ke leksikon
     berdasarkan waktu                  genuine & counterfeit
            │                                   │
            │                                   ▼
            │                          skor_kecurigaan =
            │                        sim_counterfeit − sim_genuine
            │                        (skor KONTINU, bukan label)
            │                                   │
            │                                   ▼
            │                        agregasi ke level produk
            │                     → suspicion_score per produk (y)
            │                                   │
            │                                   ▼
            │                       3. PRODUCT-LEVEL SPLIT
            │                          TRAIN / TEST
            │               (stratifikasi kuantil suspicion_score)
            │                                   │
            │                 ┌─────────────────┴─────────────────┐
            │                 ▼                                   ▼
            │             TRAIN IDs                           TEST IDs
            │                 │                                   │
            │                 ▼                                   │
            │          4. FIT GLOBAL HMM                          │
            │        (unsupervised; hanya sekuens                 │
            │         sentimen TRAIN; tidak melihat                │
            │              suspicion_score)                       │
            │                 │                                   │
            │                 └─────────────────┬─────────────────┘
            │                                   │
            ▼                                   ▼
   AGGREGATE STATISTICS               5. HMM TRANSFORM / PREDICT
   mean + std sentimen                (parameter dibekukan,
   per produk                          diterapkan ke TRAIN & TEST)
            │                                   │
            │                                   ▼
            │                       loglik + state proportions
            │                                   │
            └─────────────────┬─────────────────┘
                               ▼
                    6. FEATURE EXTRACTION
                     9D vektor per produk
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
       X_statis (2D)    X_hmm (≈7D)     X_gabungan (9D)
               │               │               │
               ▼               ▼               ▼
        7A. MODEL A      7B. MODEL B      7C. MODEL C
       (statis saja)    (HMM saja)      (statis + HMM)
               │               │               │
               └───────────────┼───────────────┘
                               ▼
               8. PREDIKSI suspicion_score
                     pada TEST IDs
                               │
                               ▼
                  9. EVALUASI RANKING
         Spearman correlation, precision@k,
        perbandingan performa Model A / B / C
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    9A. ABLATION REPORT                9B. VALIDASI EKSTERNAL
    apakah fitur HMM menambah          sampel top-k skor tertinggi
    nilai dibanding statis saja?       dicek manual / sinyal independen
               │                               │
               └───────────────┬───────────────┘
                               ▼
                10. HASIL & INTERPRETASI
             ranking produk berdasarkan
                  skor kecurigaan
```

## Catatan desain (kenapa tiap keputusan diambil)

| # | Keputusan | Alasan |
|---|---|---|
| 2B | Skor similarity dipakai **mentah/kontinu**, bukan `argmax` similarity-ke-genuine vs similarity-ke-counterfeit | Menghindari klaim "ini pasti counterfeit" yang tidak bisa dibuktikan dari teks saja; skor kecurigaan adalah klaim yang jauh lebih defensif (bandingkan dengan tradisi *fraud detection*: Bolton & Hand 2002) |
| 3 | Split TRAIN/TEST dilakukan di **level produk**, sebelum HMM di-fit | Mencegah *leakage* — HMM men-*pool* parameter lintas produk, jadi tidak boleh "mengintip" produk yang nanti dievaluasi (Kaufman et al. 2012) |
| 4 | HMM di-fit **unsupervised**, tidak diberi `suspicion_score` sama sekali | Memisahkan representation learning (HMM) dari discriminative learning (Model A/B/C); state HMM tidak perlu dipetakan ke makna genuine/counterfeit — cukup jadi fitur mentah (Jaakkola & Haussler 1999) |
| AGGREGATE STATISTICS | Dihitung per-produk, boleh sebelum atau sesudah split | **Bukan** leakage selama hanya memakai ulasan produk itu sendiri — leakage hanya terjadi kalau statistik dihitung lintas produk melewati batas split |
| 7A/7B/7C | Tiga model ablasi, bukan satu model gabungan langsung | Ini pertanyaan riset utamanya: apakah dinamika temporal sentimen (HMM) menambah informasi di atas rata-rata sentimen statis semata? Tanpa ablasi, kontribusi HMM tidak terbukti |
| 9B | Validasi eksternal terpisah dari train/test utama | `suspicion_score` tetap berasal dari teks ulasan yang sama dengan fitur — validasi terhadap sampel manual/sinyal independen (delisting, komplain terverifikasi) diperlukan untuk *face validity*, bukan sekadar akurasi internal yang berisiko sirkular |

## Yang masih perlu diputuskan sebelum implementasi

- **Level agregasi skor kecurigaan ke produk**: mean seluruh ulasan, atau max (ulasan paling mencurigakan mendominasi)? Keduanya punya justifikasi berbeda dan sebaiknya dibandingkan.
- **Representasi observasi HMM**: probabilitas softmax IndoBERT hidup di simpleks (jumlah = 1), sehingga Gaussian HMM berkovarians penuh berisiko singular. Pertimbangkan logit pra-softmax atau transformasi log-ratio (Aitchison) sebagai observasi.
- **Kalibrasi softmax IndoBERT**: jaringan modern cenderung *overconfident* (Guo et al. 2017); pertimbangkan temperature scaling sebelum skor dipakai sebagai observasi HMM.
- **Ukuran dan sumber sampel validasi eksternal (9B)**: makin penting sekarang karena ini satu-satunya pengangan di luar teks ulasan itu sendiri.