# RM 2: HMM pola temporal sentimen per produk

Kode: `src/temporal/`. Tes: `tests/temporal/`.

## Menjalankan

```bash
python -m src.temporal.run_hmm --input "outputs/sentiment/sentimen_ulasan_v2.csv" \
    --split-ids "outputs/labels/split_ids.csv"
```

- `--input`: keluaran RM 1 (satu baris per ulasan). Kolom yang dibaca: `product_id`, `timestamp`,
  `p_positif`, `p_netral`, `p_negatif`, opsional `seq_in_product`, `row_order`,
  `include_for_inference`, `is_synthetic`, dan `logit_*` (hanya untuk `--representation logit`).
  Nama kolom diubah lewat `HMMConfig`.
- `--split-ids`: CSV/parquet berkolom `product_id`, `split` (`train` atau `test`). Daftar ini dibuat
  di luar modul HMM (stratifikasi memakai label). Tanpa daftar ini, `--temporary-split` memakai split
  acak yang ditandai TEMPORARY di `hmm_meta.json`. Hasil dari split itu tidak untuk dilaporkan.
- File keluaran yang sudah ada tidak ditimpa kecuali `--overwrite`.
- Tes: `python -m pytest tests/temporal -q` (atau `python -m unittest tests.temporal.test_gaussian_hmm tests.temporal.test_pipeline`).

## Keluaran

| File | Isi |
|---|---|
| `outputs/features/hmm_features.parquet` | satu baris per produk: `log_lik`, `prop_state1`, `pos_transisi` (+ `idx_transisi`, `occ_state1`, `n_reviews`, `split`, `is_synthetic`) |
| `outputs/models/hmm_model.pkl` | parameter beku (joblib; array numpy biasa, bukan objek hmmlearn) |
| `outputs/models/hmm_meta.json` | versi pustaka, konfigurasi, seed, hash id Train, urutan state, hasil tiap restart, stabilitas seed, ringkasan fitur Train vs Test, korelasi antar fitur |

## Keputusan desain

- Split dilakukan sebelum fit. Fit hanya pada produk Train bukan sintetis, lalu parameter dibeku.
  `assert_no_leakage` memeriksa: id fit bagian dari Train, tidak beririsan dengan Test, sintetis tidak ikut fit dan tidak masuk Test.
- Modul hanya membaca kolom daftar putih; kolom label tidak pernah disentuh (ada tesnya).
- Multi-sekuens: tiap produk satu sekuens, tanpa padding atau penyambungan (`lengths`).
- Forward, backward, dan posterior dihitung dengan numpy di ruang log (`gaussian_hmm.py`), jadi fitur identik
  apa pun pustaka pelatihnya. Pelatihan: `hmmlearn.GaussianHMM(covariance_type="diag")` bila terpasang,
  kalau tidak Baum-Welch numpy (`--backend` mengatur).
- Restart: restart ke-0 memakai inisialisasi proposal (π, A, μ₀, μ₁, σ² = 0.04); sisanya acak dari satu seed.
  Dipilih log-likelihood latih terbaik; semua restart dicatat.
- Urutan state: diurutkan menurut rata-rata p_negatif berbobot posterior di data Train (naik). "State 1" =
  state paling negatif. Nama state netral (`state_negatif_tinggi`); state negatif belum tentu counterfeit.
- Urutan sekuens: `product_id`, `timestamp`, lalu `seq_in_product`, lalu `row_order` (stabil). Harus sama dengan aturan di cleaning/RM 1.
- `pos_transisi` = max |Δγ_t(1)| (sesuai Tabel 3.6); posisinya ada di `idx_transisi` (indeks 0-based titik sesudah lonjakan).

## Keputusan terbuka (nilai bawaan dapat diubah di `HMMConfig`)

Jumlah state (2), representasi observasi (`prob`; opsi `logit`, `alr`, `clr`), `min_reviews` (3; 5-10 lebih aman),
arti `pos_transisi`, temperature scaling, metode Labeling, dan aturan urutan timestamp kembar.

## Catatan untuk naskah

Contoh hitungan HMM di proposal (Tabel 3.4, 3.5, γ₁) tidak bisa direproduksi dari parameter yang tertulis,
dimensi Σ tidak seragam (3 vs 4), dan Gambar 3.1 menaruh split setelah fit HMM. Rincian di konteks proyek, bagian F.
