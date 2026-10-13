# Inferensi sentimen RM 1

Modul: "src/sentiment_inference.py" (fungsi) dan "src/run_sentiment_inference.py" (CLI).
Model: `asyzyni/indobert-nlu-smsa-counterfeit-ta` (IndoBERT IndoNLU, fine-tune SmSA, 3 kelas).
Pelatihan memakai max_length 128 (sumber: konteks pengguna); default inferensi 512 sesuai proposal.
Kolom keluaran: `p_negatif`, `p_netral`, `p_positif` (urutan kolom diturunkan dari `config.id2label`).

## Yang harus disiapkan
1. Hash commit **lengkap (40 karakter)** model di Hugging Face. Hash singkat (mis. `5366aae`) ditolak, karena tidak dipin.
2. Token Hugging Face lewat variabel lingkungan `HF_TOKEN` (Kaggle: Secrets). Jangan menulisnya di kode, notebook, atau log.
3. `pip install -r "requirements.txt"` (butuh torch, transformers, huggingface_hub, safetensors, pyarrow).
4. Tabel proposal v2. Kamus CSV UTF-8 dua kolom `slang,baku` dikonversi secara lokal dari ekspor
   SEACrowd/kamus_alay yang disediakan pengguna. Tidak ada unduhan otomatis; tanpa kamus,
   pembuatan v2 berhenti tanpa menulis parquet.

## Membuat preprocessing proposal

Sumber: Salsabila dkk. (2018), *Colloquial Indonesian Lexicon*, melalui ekspor loader SEACrowd/kamus_alay
config `*_source` dengan `load_dataset` (versi ekspor menurut pengguna: datasets 3.6.0, seacrowd 0.2.2,
Python 3.13.1). URL asal loader, hash dan batas verifikasi dicatat dalam "references/slang_PROVENANCE.md".
Lisensi dicatat **Unknown** menurut informasi halaman SEACrowd yang disampaikan pengguna; tidak diasumsikan
sebagai izin redistribusi. Pengguna memilih agar "references/slang.csv" tetap lokal dan diabaikan Git.
Aturan `*.csv` sudah ada; usulan `references/sumber/` untuk ".gitignore" hanya ditampilkan di provenance.

    python -m src.cleaning.build_slang_dictionary \
      --input "references/sumber/colloquial-indonesian-lexicon.csv" \
      --output "references/slang.csv" \
      --excluded "references/slang_dikecualikan.csv" \
      --profile "reports/profil_kamus_alay.md" \
      --expected-sha256 "48e7450875b75935cf15b200524e97927a8c27f8169e658cf4a73993c7818d76"

Profil terverifikasi: 15.006 record CSV, 4.330 slang unik, 4.409 pasangan unik, dan 71 kunci multi-baku.
CSV mempunyai field komentar berkutip dengan newline, sehingga 15.397 LF bukan 15.396 record.
Arti `in_dictionary` tidak diasumsikan; `in_dictionary`/`categories`/`context` tidak menyaring kamus.
Isi komentar Instagram tidak disalin ke keluaran konverter. Kedua CSV turunan hanya memuat
`slang,baku` atau `slang,baku,alasan`.

Asumsi filter: casefold dan rapikan spasi kunci; baku hanya dirapikan spasinya; buang kosong,
duplikat pasangan, slang sama dengan baku, substring `PROTECTED_TERMS` pada kedua sisi,
semua pemetaan kunci multi-baku, serta kunci 1–2 karakter. Konflik dideteksi sebelum filter lain;
frekuensi tidak dipakai memilih baku. Pencocokan substring di konverter juga membuang `ori` dalam
`favorit`/`tutorial`. Daftar seluruh 131 kunci dua karakter tersedia di "reports/profil_kamus_alay.md"
untuk keputusan lanjutan; belum ada whitelist.

Hasilnya 4.124 pasangan. Kunci yang hilang 206/4.330 = 4,757506%: **gerbang 30% tidak terpicu**.
Bila >30% kunci nonkosong unik hilang, konverter menulis bahan tinjauan lalu exit 2; berhenti dan tunggu
keputusan pengguna sebelum perintah berikutnya. Duplikasi record konteks tidak dihitung sebagai kehilangan kunci.

    python -m src.cleaning.run_proposal \
      --input "outputs/cleaned/reviews_clean (2).parquet" \
      --slang-csv "references/slang.csv" \
      --output "outputs/cleaned/reviews_clean_v2.parquet" \
      --report "reports/preprocessing_proposal_v2.md"

Input eksplisit wajib untuk berkas lokal ini; "outputs/cleaned/reviews_clean.parquet" tidak tersedia.
SHA-256 input yang diverifikasi sebelum run:
`c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0`.
File keluaran yang sudah ada ditolak. `--input`, `--output`, `--report`, `--excluded`, dan `--profile`
dapat menunjuk nama baru sesuai CLI untuk percobaan berikutnya; input tidak diganti nama.

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

Audit artefak yang sudah ditulis dan ekspor tinjauan manual:

    python -m src.cleaning.audit_proposal \
      --input "outputs/cleaned/reviews_clean (2).parquet" \
      --output "outputs/cleaned/reviews_clean_v2.parquet" \
      --slang-csv "references/slang.csv" \
      --report "reports/verifikasi_proposal_v2_alay.md" \
      --top-csv "reports/top_penggantian_slang.csv" \
      --expected-input-sha256 "c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0" \
      --expected-rows 64699 --expected-columns 27

Pada CLI audit, `--output` menunjuk parquet v2 **yang dibaca**. Laporan dan CSV yang sudah ada ditolak;
gunakan nama laporan/CSV baru untuk audit ulang. Transformasi diregenerasi dengan pencatatan penggantian
aktual dan dibandingkan persis terhadap v2. CSV berisi 100 pasangan teratas menurut frekuensi, dengan
3 ulasan berbeda yang dipilih `random_state=42` per pasangan; konteks berasal dari ulasan pengguna.
Jika kurang dari tiga ulasan tersedia, slot sisanya kosong dan jumlah contoh dilaporkan.

Hasil lokal terverifikasi pada 2026-10-13: **64.699 baris, 27 kolom lama identik, 29 kolom v2**;
urutan `(review_id, seq_in_product)` dan nilai `text_light` identik. **`text_lex` tidak terdapat pada input
ini maupun v2**, sehingga tidak ada nilai kolom tersebut yang dapat dibandingkan; kolom tidak ditambahkan.
Ada 45.082 ulasan berubah, 58 `text_proposal` kosong (45 masih include), serta 61.003 penggantian slang
dari 1.150 pasangan yang terpakai. Slang mengubah 26.594 ulasan; satu ulasan dapat memuat banyak penggantian.
Seluruh sembilan istilah terlindungi memiliki nol kehilangan kemunculan per baris; hitungan sebelum/sesudah
lengkap ada di laporan audit. SHA-256 input tetap sama dengan hash wajib di atas.

`text_proposal` hanya masukan untuk IndoBERT. Audit ini tidak memvalidasi kualitas linguistik atau kualitas
model; keputusan `--max-length` 128/512 tetap terbuka dan default tidak diubah.

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

Run penuh integrasi kamus pada 2026-10-13: `1 failed, 176 passed, 2 skipped in 6.48s` (exit 1).
Keluaran asli lengkap: "reports/hasil_tes_integrasi_alay.txt". Semua tes konverter/audit/proposal lulus;
kegagalan berada pada `TestHmmlearnParity.test_forward_matches_hmmlearn_score_and_posterior`
di "tests/temporal/test_pipeline.py": "src/temporal/fitting.py" mengharapkan kovarians `(2, 3)`,
sedangkan yang diterima `(2, 3, 3)`. HMM di luar lingkup dan tidak diubah. Dua tes model nyata dilewati
karena `INDOBERT_SENTIMENT_MODEL_DIR` belum ditetapkan.

## Keterbatasan (tulis juga di naskah)

Model difine-tune dengan max_length 128; inferensi memakai 512 sesuai proposal; perilaku pada ulasan > 128 token tidak divalidasi.

SmSA adalah ulasan umum, bukan marketplace. Kualitas pada ulasan ponsel belum terukur dan butuh sampel berlabel manual
(rencana 150 baris, bukan dari rating bintang). Distribusi argmax dan rata-rata `p_netral` di ringkasan adalah
gambaran keluaran model, bukan akurasi. Test set SmSA hanya boleh dipakai sekali setelah model final dipastikan.
