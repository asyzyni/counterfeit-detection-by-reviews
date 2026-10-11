# Inferensi sentimen RM 1

Modul: "src/sentiment_inference.py" (fungsi) dan "src/run_sentiment_inference.py" (CLI).
Model: `asyzyni/indobert-nlu-smsa-counterfeit-ta` (IndoBERT IndoNLU, fine-tune SmSA, 3 kelas).
Pelatihan memakai max_length 128 (sumber: konteks pengguna); default inferensi 512 sesuai proposal.
Kolom keluaran: `p_negatif`, `p_netral`, `p_positif` (urutan kolom diturunkan dari `config.id2label`).

## Yang harus disiapkan
1. Hash commit **lengkap (40 karakter)** model di Hugging Face. Hash singkat (mis. `5366aae`) ditolak, karena tidak dipin.
2. Token Hugging Face lewat variabel lingkungan `HF_TOKEN` (Kaggle: Secrets). Jangan menulisnya di kode, notebook, atau log.
3. `pip install -r "requirements.txt"` (butuh torch, transformers, huggingface_hub, safetensors, pyarrow).
4. Tabel proposal v2. Jalur baru membutuhkan kamus slang CSV UTF-8 dua kolom `slang,baku` dari pengguna.
   Kamus tidak dibuat atau diunduh otomatis. Tanpa kamus, pembuatan v2 berhenti tanpa menulis parquet.

## Membuat preprocessing proposal

    python -m src.cleaning.run_proposal --slang-csv "references/slang.csv"

Perintah membaca "outputs/cleaned/reviews_clean.parquet" dan menulis
"outputs/cleaned/reviews_clean_v2.parquet" serta "reports/preprocessing_proposal_v2.md".
Path kamus di contoh harus diganti sesuai file yang disediakan pengguna. File keluaran yang sudah ada ditolak.
`--input`, `--output`, dan `--report` dapat menunjuk nama baru untuk percobaan berikutnya.

`text_proposal` dibuat dari `review` asli: hapus URL; emoji/karakter di luar huruf Unicode, digit desimal,
spasi, dan `. , ! ? ' -` diganti spasi; tanda baca identik berulang diringkas; angka berdiri sendiri
>= 6 digit dihapus; huruf identik >= 3 kali menjadi satu; slang dari kamus dinormalkan; spasi dirapikan.
Angka pendek dan alfanumerik seperti `128gb`, `5g`, `a07`, `iphone 17` tetap: asumsi konservatif
untuk angka kontekstual. Nomor yang dipisahkan spasi/tanda baca tidak digabung untuk dihapus.
Pencocokan slang tidak peka kapital, pada batas kata/frasa dan sekali jalan. Kapital seluruh token
atau awal token dipertahankan; tidak ada lowercase paksa, stemming, atau penghapusan stopword.

`PROTECTED_TERMS` ada di "src/cleaning/config.py": `kw`, `ori`, `original`, `palsu`, `replika`,
`imitasi`, `hdc`, `bukan ori`, `sesuai deskripsi`. Istilah/frasa ini tidak dinormalisasi oleh slang/typo.
Mempertahankan `kw` adalah **penyimpangan disengaja** dari contoh proposal Bab 3.3.

`text_light`, `text_lex` (jika ada), dan semua kolom lama tetap. `text_proposal_kosong` menandai hasil
kosong. Tidak ada baris dihapus/disortir ulang. Kesamaan seluruh kolom lama, jumlah dan urutan
`review_id, seq_in_product` diperiksa sebelum dan sesudah round-trip parquet di memori.
Laporan memuat hitungan per aturan, jumlah penggantian slang, kosong, duplikat, dan 20 contoh
acak dari seluruh tabel dengan seed 42. Satu baris dapat berubah pada beberapa aturan.

Cleaning mengecualikan duplikat teknis (produk + text_light + timestamp berjam identik, sesudah
kemunculan pertama) dari `include_for_inference`. Duplikat teks hanya ditandai; flag tidak diubah.
Rekomendasi: tinjau duplikat teks sebelum menetapkan bahwa teks sama adalah spam/error.

## Menjalankan (Mac M2, MPS)

    # Set HF_TOKEN melalui lingkungan lokal/Secrets; jangan menulis nilainya di notebook atau log.
    export INDOBERT_SENTIMENT_REVISION=<hash 40 karakter>
    python -m src.run_sentiment_inference --device mps --limit 200 --output "outputs/sentiment/uji_200_v2.csv"
    python -m src.run_sentiment_inference --device mps --batch-size 16 --max-length 512

Input bawaan "outputs/cleaned/reviews_clean_v2.parquet" (kolom teks `text_proposal`). Kolom tidak tersedia
membuat run berhenti dengan pesan jelas. Output bawaan "outputs/sentiment/sentimen_ulasan_v2.csv".
Semua baris dipertahankan;
teks kosong mendapat probabilitas NaN dan kolom `include_for_inference` dari tahap cleaning tetap ikut,
jadi tahap berikutnya bisa memfilter. **CLI tetap menginferensi semua teks nonkosong**, termasuk
baris dengan `include_for_inference=False`, seperti versi lama; tidak ada perubahan filter otomatis.
Keluaran: CSV + "*_ringkasan.json" + "*_ringkasan.md". File yang sudah ada tidak ditimpa.

Opsi: `--save-logits` (logit mentah, untuk temperature scaling/ALR/CLR), `--save-diagnostics` (`n_token`, `terpotong`),
`--model-dir` (folder model lokal, tanpa jaringan), `--device cuda` (Kaggle), `--no-argmax`,
`--max-length` (default 512, harus positif dan <= `model.config.max_position_embeddings`).
Device tidak pernah jatuh ke CPU secara diam-diam: `auto` memilih CUDA lalu MPS, dan berhenti dengan pesan bila keduanya tidak ada.
Batch diurutkan stabil menurut panjang token, memakai padding dinamis, lalu dikembalikan ke urutan semula.
Pada Kaggle gunakan batch yang muat di GPU; turunkan `--batch-size` bila kehabisan memori pada 512 token.

## Pemeriksaan otomatis sebelum run penuh

`id2label = {0:negatif, 1:netral, 2:positif}`; batas panjang <= kapasitas posisi model (512);
tanpa NaN/Inf; jumlah probabilitas = 1 (<=1e-6); hasil batch = hasil satu-satu (<=1e-4).
`tokenizer.model_max_length` boleh tetap 128: hanya dicatat di meta sebagai `tokenizer_model_max_length`,
bersama `max_length_pelatihan: 128`, `max_length` aktual, dan `max_position_embeddings`.
`--no-tokenizer-strict` tetap diterima untuk kompatibilitas, tanpa efek. Tokenisasi untuk forward selalu
`truncation=True, max_length=<konfigurasi>` eksplisit. Pengukuran panjang diagnostik saja memakai
`truncation=False`. Token khusus [CLS]/[SEP] dihitung; `n_terpotong` dan `persen_terpotong`
menggunakan panjang aktual, dengan penyebut jumlah teks nonkosong yang diinferensi.

Bab 3.4.2 diuji dengan tiga kalimat tetap melalui bundle yang sama:

| Harapan argmax | Kalimat |
| --- | --- |
| positif | Barangnya bagus banget, sesuai deskripsi, pengiriman cepat |
| netral | Barang sudah sampai, belum saya coba |
| negatif | Barang rusak, tidak sesuai foto, kecewa berat |

Setiap kalimat menghasilkan satu entri pemeriksaan: ketiga probabilitas (4 desimal), argmax, harapan,
dan status lulus. Hasil dicetak serta disimpan dalam JSON/Markdown, termasuk saat pemeriksaan gagal.
JSON juga menyimpan probabilitas numerik penuh. Gagal argmax berarti temuan, bukan alasan mengganti kalimat.
Run berhenti dengan exit code 1 tanpa CSV inferensi penuh; ringkasan memiliki `ringkasan: null` dan status berhenti.
`--continue-on-failed-checks` melanjutkan dengan kegagalan tetap tercatat, tetapi tidak dapat melewati
batas kapasitas posisi model. Setelah run berhenti, gunakan nama output baru karena ringkasan lama dilindungi.

## Tes

    python -m pytest "tests" -q -rs
    python -m unittest discover -s "tests" -p "test_sentiment_inference.py"
    INDOBERT_SENTIMENT_MODEL_DIR="<folder model>" INDOBERT_SENTIMENT_DEVICE=mps python -m pytest "tests/test_sentiment_inference.py" -s

Unittest hanya mencakup kelas unittest sentimen. Pytest mencakup seluruh tes, termasuk fungsi pytest
di "tests/cleaning". Tanpa folder model, tes model nyata dilewati; model palsu tidak memvalidasi kualitas model.

## Keterbatasan (tulis juga di naskah)

Model difine-tune dengan max_length 128; inferensi memakai 512 sesuai proposal; perilaku pada ulasan > 128 token tidak divalidasi.

SmSA adalah ulasan umum, bukan marketplace. Kualitas pada ulasan ponsel belum terukur dan butuh sampel berlabel manual
(rencana 150 baris, bukan dari rating bintang). Distribusi argmax dan rata-rata `p_netral` di ringkasan adalah
gambaran keluaran model, bukan akurasi. Test set SmSA hanya boleh dipakai sekali setelah model final dipastikan.
