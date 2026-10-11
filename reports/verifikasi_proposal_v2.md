# Verifikasi perubahan proposal — 11 Oktober 2026

Tugas 1 dan 2 diimplementasikan dan diuji dengan model/tokenizer palsu. Tugas 3 memiliki
fungsi pembersihan, pembaca kamus, CLI v2, dan tes. **Penulisan parquet v2 final tertunda**:
kamus slang tidak ditemukan; menunggu path CSV dua kolom `slang,baku` dari pengguna.
Tidak ada normalisasi slang produksi yang dikarang atau diunduh.

## Lingkup dan file

Diubah:

- "src/sentiment_inference.py": default 512, validasi kapasitas model, informasi tokenizer/pelatihan,
  tiga kalimat tetap, probabilitas/argmax, ringkasan dan keterbatasan.
- "src/run_sentiment_inference.py": `--max-length`, default input/teks/output v2,
  ringkasan saat pemeriksaan gagal, exit code 1, opsi lanjut dengan kegagalan tetap tercatat.
- "src/cleaning/config.py": `PROTECTED_TERMS`.
- "tests/test_sentiment_inference.py": batas token, pemetaan semantik, CLI dan pengaman.
- "docs/inferensi_sentimen.md": petunjuk, asumsi, batasan, perilaku duplikat, serta perintah tes.

Dibuat:

- "src/cleaning/proposal.py": fungsi murni dan laporan preprocessing.
- "src/cleaning/run_proposal.py": CLI penulisan v2, wajib kamus, tanpa menimpa.
- "tests/cleaning/test_proposal.py": aturan teks, kamus sintetis, integritas kolom/baris, NaN, round-trip.
- "reports/preprocessing_proposal_v2_parsial.md": audit data nyata tanpa kamus, 20 contoh seed 42,
  daftar istilah terlindungi dan SHA-256 keluaran lama.
- "reports/verifikasi_proposal_v2.md": laporan ini.

"src/config.py", text_light, text_lex, aturan include_for_inference, pelabelan genuine/counterfeit,
HMM, XGBoost, dan hyperparameter pelatihan tidak diubah. Tidak melakukan commit/push,
menghapus file proyek, atau menimpa keluaran lama. Tidak memuat training_args.bin atau memakai rating sebagai label.

## Bukti perintah tes

Perintah seluruh tes (mencakup "tests/cleaning"):

```text
python -m pytest "tests" -q -rs
112 passed, 2 skipped in 6.16s
```

Kedua skipped adalah tes model nyata pada "tests/test_inference_smoke.py" dan
"tests/test_sentiment_inference.py"; INDOBERT_SENTIMENT_MODEL_DIR belum diisi.
torch/transformers terdeteksi terpasang, tetapi bobot/forward model nyata tidak diuji.

Perintah kelas unittest sentimen (subset, tidak dijumlahkan dengan hitungan pytest):

```text
python -m unittest discover -s "tests" -p "test_sentiment_inference.py"
Ran 34 tests in 0.118s

OK (skipped=1)
```

Tes terarah sebelumnya: `59 passed, 1 skipped in 0.36s`.
Pemeriksaan sintaks empat modul baru/diubah dan `git diff --check` keluar dengan kode 0.
Help CLI menampilkan default `--max-length` 512 dan `--text-col` text_proposal.

Harness audit lama di luar penemuan pytest:

```text
PYTHONPATH="." python "tests/label_readiness/synthetic_label_readiness.py"
synthetic_rows=320
synthetic_products=2
boundary_probe=['kandidat', 'seed_counterfeit']
IMPORT_ERROR[src.features.lexicon]=ModuleNotFoundError: No module named 'src.features.tfidf'
IMPORT_ERROR[src.features.tfidf]=ModuleNotFoundError: No module named 'src.features.tfidf'
test_1_SO_and_CI=BLOCKED: Jalur B module unavailable
test_2_review_score_ranking=BLOCKED: Jalur B module unavailable
test_3_mean_and_max_product_ranking=BLOCKED: Jalur B module unavailable
test_4_swapped_seed_ranking=BLOCKED: Jalur B module unavailable
test_5_review_boundary=BLOCKED: Jalur B module unavailable
test_6_determinism=BLOCKED: Jalur B module unavailable
```

Enam pemeriksaan BLOCKED tersebut tidak diklaim lulus dan modul Jalur B tidak disentuh.

## Data nyata dan status pembuatan v2

Path yang ada adalah "/Users/asyzyni/Desktop/TA/Kode dan Eksperimen/Kode TA".
Folder induk "/Users/asyzyni/Desktop/TA " tidak ada menurut `ls`.

Audit parsial membaca "outputs/cleaned/reviews_clean.parquet" dan membangun tabel di memori:

- 64.699 baris; 36.378 berubah; 58 kosong (45 masih include=True).
- Per aturan: URL 6; emoji/karakter khusus 27.710; tanda baca berulang 10.478;
  angka panjang 11; typo huruf berulang 4.143; slang 0 **karena belum dijalankan**; spasi 28.998.
- Duplikat teknis 347; yang masih include=True 0.
- Duplikat teks 4.668; yang masih include=True 4.647.
- Include total 64.332. Flag tidak diubah; CLI sentimen tetap memproses semua teks nonkosong seperti sebelumnya.
- Jumlah, urutan review_id/seq_in_product, tipe, dan seluruh kolom lama identik pada tabel di memori.
  Belum ada perbandingan file parquet v2 nyata karena file final belum ditulis.
- Empat keluaran lama (parquet dan tiga file uji sentimen) identik menurut SHA-256 selama audit.

```text
python -m src.cleaning.run_proposal
BERHENTI: kamus slang belum tersedia. Sediakan CSV dua kolom slang,baku lewat --slang-csv.
exit code: 2
```

## Asumsi dan tindak lanjut

Asumsi: review pada tabel lama adalah sumber teks asli yang dimaksud; karakter terlarang diganti
spasi agar kata tidak tersambung; hanya tanda baca identik berurutan diringkas; hanya angka berdiri
sendiri >= 6 digit dihapus (nomor terpisah tidak digabung). Angka pendek/alfanumerik tetap.
Kamus dicocokkan tanpa peka kapital, sekali jalan pada batas kata/frasa, dengan kapital seluruh/awal token
dipertahankan; tidak ada lowercase paksa. Sampel laporan berasal dari seluruh tabel dengan seed 42.
Nilai max_length pelatihan 128 berasal dari konteks pengguna, bukan inspeksi training_args.bin.
Proteksi kw adalah penyimpangan disengaja dari contoh proposal 3.3.

Pengguna perlu memberikan lokasi CSV `slang,baku`. Setelah tersedia, jalankan:

```sh
python -m src.cleaning.run_proposal --slang-csv "<path CSV slang,baku>"
python -m src.run_sentiment_inference --device mps --max-length 512 --limit 200 --output "outputs/sentiment/uji_200_v2.csv"
```

Pada Kaggle gunakan `--device cuda`; pilih batch yang muat di GPU. Konfigurasikan revision hash
40 karakter dan HF_TOKEN melalui lingkungan/Secrets sesuai dokumentasi, atau gunakan `--model-dir`.
Jika tiga kalimat pada model nyata gagal, laporkan probabilitasnya dan berhenti; jangan mengubah
kalimat atau harapan argmax. Pilih nama keluaran baru untuk pengulangan bila ringkasan sudah tercipta.
