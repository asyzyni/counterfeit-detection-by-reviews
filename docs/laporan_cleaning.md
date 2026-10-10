# Laporan Cleaning — Data Beneran → tabel review RM 1

Run: 2026-10-09 · Kode: `src/cleaning/` · Perintah: `python -m src.cleaning.run`
Output: `outputs/cleaned/reviews_clean.parquet` + `outputs/cleaned/reports/`

Tahap ini hanya cleaning/preprocessing. Tidak ada label, skor kecurigaan, atau sentimen.
Folder `Data Beneran` hanya dibaca. Tidak ada file lama yang diubah.

---

## 1. Angka utama

| Besaran | Nilai |
|---|---|
| Baris di file pemetaan | 405 (404 tanpa baris yang merujuk dirinya sendiri) |
| File diproses (status `ok`) | **403** |
| File dikecualikan | 2: `CHIC PHONE.csv` (status "dikecualikan: hanya nama toko") dan `pemetaan_file_toko_produk.csv` (baris pemetaan merujuk dirinya sendiri) |
| File tanpa kolom review_text | 0 |
| Baris sumber | **100.625** |
| Baris tanpa sel teks terisi (tidak masuk tabel) | 35.498 |
| Baris yang selnya hanya non-teks (tidak masuk tabel) | 428 |
| Baris di tabel (baris dengan teks) | **64.699** |
| `is_short` (< 2 karakter) | 20 |
| `is_technical_duplicate` | 347 |
| `is_text_duplicate` (hanya ditandai) | 4.668 |
| `seller_reply_suspect` (hanya ditandai) | 28 |
| **`include_for_inference`** | **64.332** |
| Timestamp NaT (semuanya tanggal relatif) | 939 baris di tabel |
| Produk (`product_id`) | 401 |
| Produk dengan ≥3 / ≥5 / ≥10 ulasan include | 401 / 398 / 381 |
| Grup file berisi identik (hash sama) | **21 grup, 42 file** |

Setiap baris sumber tercatat. Untuk tiap file berlaku: baris sumber = baris kosong + baris keluar (lihat `file_manifest.csv`, kolom `cek_seimbang`).

Rincian per keluarga skema:

| Keluarga | File | Baris sumber | Kosong: semua sel teks kosong | Kosong: hanya non-teks | Baris di tabel | Include | Baris timestamp berjam |
|---|---|---|---|---|---|---|---|
| A web-scraper Shopee | 198 | 41.782 | 21.298 | 421 | 20.063 | 19.715 | 15.213 |
| B Lazada | 107 | 30.362 | 5.354 | 6 | 25.002 | 24.996 | 0 |
| C Shopee kolom acak | 98 | 28.481 | 8.846 | 1 | 19.634 | 19.621 | 19.634 |

Sebaran ulasan `include_for_inference` per produk (401 produk): min 3, Q1 25, median 61, Q3 167, maks 2.452, rata-rata 160,4.

| Ulasan include | 3–4 | 5–9 | 10–49 | 50–99 | 100–499 | ≥500 |
|---|---|---|---|---|---|---|
| Jumlah produk | 3 | 17 | 152 | 80 | 130 | 19 |

Tahun timestamp: 2019: 11 · 2020: 247 · 2021: 1.952 · 2022: 2.580 · 2023: 2.911 · 2024: 9.589 · 2025: 16.475 · 2026: 29.995 (rentang 2019-06-19 s.d. 2026-09-07).

Catatan: angka run lama di `RM1_execution_report.md` (303.687 → 15.783) berasal dari folder data lain (`data/raw`), jadi tidak sebanding langsung. Pada data ini, baris yang tidak masuk tabel hampir semuanya (35.498 dari 35.926) adalah baris yang seluruh sel teksnya kosong. Kemungkinan besar ini ulasan bintang tanpa teks.

### Keluaran perintah (run resmi)

```
[build] file diproses=403, baris output=64,699
File diproses           : 403
File dikecualikan       : 2
Baris sumber            : 100,625
Baris kosong (luar tabel): 35,926
Baris di tabel          : 64,699
  is_empty              : 0
  is_short              : 20
  is_technical_duplicate: 347
  is_text_duplicate     : 4,668
  include_for_inference : 64,332
  timestamp NaT         : 939
Produk (product_id)     : 401
  produk >=  3 include  : 401
  produk >=  5 include  : 398
  produk >= 10 include  : 381
Grup file identik       : 21

PEMERIKSAAN OTOMATIS
[LULUS] jumlah file status ok = harapan -> ok=403, harapan=403
[LULUS] file diproses + dilewati + dikecualikan-duplikat = file ok -> diproses=403, tanpa review_text=0, duplikat-dikecualikan=0, ok=403
[LULUS] file dikecualikan dilaporkan -> CHIC PHONE.csv (...); pemetaan_file_toko_produk.csv (...)
[LULUS] file tanpa review_text <= 5% -> 0 file (0.0%): []
[LULUS] tiap file: baris sumber = baris kosong + baris keluar -> 403 file diperiksa; tidak seimbang: []
[LULUS] baris keluar di manifest = baris di tabel -> selisih: []
[LULUS] review_id unik & tanpa NaN -> 64,699 baris, 64,699 id unik
[LULUS] tidak ada review/text_light kosong pada include_for_inference -> 64,332 baris include; kosong=0
[LULUS] product_id seluruhnya berasal dari pemetaan -> tidak dikenal: []
[LULUS] fragmen review hanya dari kolom berperan review_text -> peran asal fragmen: ['review_text']
[LULUS] seq_in_product = 1..n per produk -> 401 produk
```

**Determinisme.** Run kedua dijalankan di proses terpisah dengan `PYTHONHASHSEED` berbeda (1 vs 999) ke folder scratchpad. Perintah `diff` pada `reports/output_hashes.csv` kedua run tidak menunjukkan perbedaan: ke-14 file output, termasuk parquet (`sha256 c15e1ab7…`), identik.

**Tes.** `python -m pytest tests/cleaning -q` → `30 passed in 0.53s`. Tes memakai CSV mini sintetis untuk tiap keluarga skema. Kasus yang dicakup:
- kolom `data` tanpa angka, `review-attribute`, dan kolom acak;
- balasan penjual dengan penanda, nama pengguna tersamar;
- baris kosong, duplikat teknis vs duplikat teks;
- tanggal relatif dan format tak terparsing (`07/03/2026`);
- kosakata tag aspek, grup file identik, ekspor anonim;
- penolakan folder output yang sudah ada, dan determinisme.

---

## 2. Alur dan keputusan

1. **Daftar file.** Daftar file diambil hanya dari pemetaan (utf-8-sig), bukan dari glob folder. Baris yang merujuk file pemetaan itu sendiri dilewati; ini disetujui Asyifa pada sesi ini.
2. **Integritas file.** Hash SHA-256 isi setiap CSV ditulis ke `duplicate_files.csv`. Kolom `file_dup_group` (FD01…FD21) ada di tabel. Tidak ada file yang dikeluarkan (`exclude_duplicate_file_groups=False`).
3. **Peran kolom ditentukan dari profil nilai** (`column_map.csv`, 2.035 baris file×kolom). Untuk setiap nilai sel dihitung jenisnya: tanggal, tanggal relatif, URL, angka, label berakhiran `:`, nama tersamar `x***y`, akun `abc123`, gaya bahasa balasan penjual, atau teks. Peran ditetapkan dari proporsinya. Nama kolom hanya dipakai sebagai petunjuk awal (kolom `name_hint`). Hasil:

   | Peran | Kolom | Contoh |
   |---|---|---|
   | review_text | 837 (806 keyakinan tinggi, 31 sedang) | `data`, `data2..9`, `review*`, `meQyXP`, `f35Wh2`, `YNedDV`, `item-content-…` |
   | timestamp | 403 (tepat satu per file) | `timestamp`, `time`, `XYk98l`, `phone` (50 file), `data3` (13), `data4` (4) |
   | metadata | 313 | angka (durasi video, `+N` foto, web_scraper_order); `skuInfo-value` (variasi); nama pengguna (4 kolom, 1.800 sel); label UI konstan |
   | url_media | 231 | `web_scraper_start_url`, `HcSdrS src` |
   | aspect_tag | 43 | `review-attribute`, `K5v3lN`, `skuInfo-label`, beberapa `data4/5` berisi label saja |
   | seller_reply | 10 | `QSiE2A` (bersama penanda "Respon Penjual:"), `item-content-… (5)` Lazada (gaya "Pelanggan yang Terhormat…", "toko kami") |
   | lainnya | 198 | kolom kosong (`pagination`) |
   | perlu_cek | 0 | — |

   Temuan yang tidak ada di deskripsi awal:
   - kolom `phone`, `data3`, `data4` di sebagian file berisi tanggal;
   - `data3` di satu file berisi nama pembeli;
   - `data` (Smart Hub Store_vivo y17) dan `review2` (2 file ZEBA MARK) berisi nama akun pembeli;
   - `rating` (king77.toko_vivo y71) berisi angka seperti 163 atau 87, bukan skala 1–5, sehingga diberi peran metadata;
   - balasan penjual muncul juga di kolom Lazada `item-content-… (5)`.
4. **Penyaringan per fragmen** di kolom review_text, dengan alasan tercatat di `cleaning_audit.csv` dan `fragment_counts.csv`:
   - 435 label berakhiran `:` (mis. "Kualitas:", "⚡Kinerja:") dipindah ke `aspect_tags`;
   - 50 nilai angka saja;
   - 16 simbol saja.
5. **`review`**: fragmen apa adanya (di-strip), digabung dengan ` | ` sesuai urutan kolom. `fragment_cols` mencatat kolom asalnya.
6. **`text_light`**:
   - *Pemisah:* ` | ` diganti `. ` jika fragmen belum diakhiri tanda baca, selain itu spasi. Alasannya, `|` bukan token bermakna bagi IndoBERT.
   - *Langkah, berurutan:* HTML/entity → URL → karakter kontrol & zero-width (baris baru → spasi; ZWJ di dalam emoji dipertahankan) → NFKC → huruf berulang >2 → 2 → rapikan spasi.
   - *Yang tidak dilakukan:* lowercase, stemming, stopword, slang, ejaan. Emoji dan angka tetap utuh.
   - *Jumlah baris berubah per langkah:* URL 6, kontrol/zero-width 4.237, NFKC 735, huruf berulang 4.143, spasi 2.879.
   - *Tidak memakai `normalize_text_series` lama* karena fungsi itu memotong angka (lihat §4).
7. **Timestamp**:
   - *Format yang ditemukan* (`timestamp_formats.csv`) hanya dua: `YYYY-MM-DD HH:MM` (Shopee, kadang diikuti `| Variasi: …`) dan `YYYY-MM-DD` (Lazada dan sebagian Shopee). Ditambah tanggal relatif `N minggu/hari/jam lalu`.
   - *Tanggal relatif tidak ditebak*: 939 baris di tabel → NaT, `timestamp_status="relatif"`. Pertanyaan ambang >10% dijawab "lewati dulu", saya artikan: lanjut dengan NaT dan laporkan.
   - *Sebaran NaT:* 131 dari 403 file punya NaT, 54 file ≥10%, 6 file ≥50%, dan 1 file 100% (`Abassystore.id_VIVO Y20 RAM 8+256GB.csv`, 5 baris). Lihat `file_manifest.csv` kolom `pct_timestamp_nat`.
8. **Duplikat**:
   - *Teknis:* product + `text_light` + timestamp berjam sama persis. Kemunculan pertama (urut source_file, row_order) dipertahankan, sisanya `include_for_inference=False`.
   - *Teks:* teks sama dalam produk yang sama, timestamp berbeda. Hanya ditandai.
   - *`dup_group_size`:* jumlah baris dengan produk + teks yang sama.
9. **`seq_in_product`**: urut timestamp (NaT di akhir), lalu source_file, lalu row_order. Timestamp tanpa jam diperlakukan sebagai 00:00 saat diurutkan.
10. **`review_id`** = `product_id__<8 heksa sha256 nama file>__row_order` (row_order mulai dari 0 = baris data pertama setelah header). Tag nama file diperlukan karena dua product_key dipakai oleh dua file (lihat §3.2). Tanpa tag itu, id akan bertabrakan.
11. **Ekspor Tabel 3.1**: `export_tabel_contoh(df, n=3, anonimkan=False)` di `src/cleaning/export.py`. Hasilnya `tabel_contoh.csv` dan `tabel_contoh_anonim.csv` (Toko A/Produk 1). Anonimisasi hanya diterapkan pada ekspor ini.

---

## 3. Butuh keputusan Asyifa

1. **21 grup file identik dengan product_key berbeda (42 file, 2.478 baris tabel, 2.334 include).** Isi CSV-nya sama byte-per-byte, padahal produknya berbeda. Contoh: `DIGITEL CELLULAR_Vivo Y20` = `…_Vivo Y27`; `PINDUODUO` 6 pasang; `TEMU.TOKO` 3 pasang; `HP Mart_Infinix Hot 60 Pro+ 8:256gb .csv` = `HP Mart_iPhone 15 Pro Max.csv`. Kemungkinan besar file salah simpan atau salah scrape, sehingga ulasan yang sama terhitung untuk dua produk. Opsi `--exclude-duplicate-file-groups` hanya mempertahankan file pertama menurut abjad, belum tentu produk yang benar. **Saran:** cek manual atau scrape ulang sebelum RM 1 dipakai untuk analisis.
2. **Dua product_key dipakai dua file berbeda isi.** `Gemilang Cell_Infinix Hot 60 Pro(.csv / spasi.csv)` dan `HP Mart_Infinix Hot 60 Pro+ 8:256gb(.csv / spasi.csv)`. Saat ini keduanya digabung sebagai satu produk sesuai pemetaan. Benarkah produknya sama?
3. **Tag aspek.** `aspect_tag_candidates.csv` berisi 1.319 kandidat (≤3 kata, muncul di ≥3 file). Sebagian besar adalah opini pendek umum ("bagus", "ok", "berfungsi dengan baik"), jadi tidak ada yang dibuang. Label berakhiran `:` sudah dipisah secara struktural. Cara pakai: isi kolom `setuju` dengan `ya` untuk fragmen yang memang tag, lalu jalankan `--aspect-tag-vocab <file.csv>`.
4. **Frasa templat platform.** `preset_phrase_candidates.csv` berisi 139 segmen yang muncul di ≥10 file, mis. "Baterai tahan lama, Pengalaman Android yang lancar, …". Ini tampaknya frasa pilihan siap klik dari Shopee/Lazada, bukan tulisan bebas. Saat ini tetap masuk `review`. Dipertahankan?
5. **`seller_reply_suspect` (28 baris).** Ini sisa balasan penjual di kolom campuran (mis. `review4` Maniac Poco). Saat ini hanya ditandai dan tetap `include_for_inference=True`; sebagian adalah false positive. Keluarkan dari inferensi?
6. **Lazada tidak punya jam.** Duplikat teknis tidak terdeteksi untuk 25.002 baris Lazada. Ada 223 baris dengan teks + tanggal sama tanpa jam. Opsi `--allow-date-only-technical-dup` akan ikut menandainya. Diaktifkan?
7. **939 timestamp relatif** (NaT). Mau scrape ulang file dengan % NaT tinggi, atau biarkan? Saya tidak menurunkan tanggal dari epoch `web_scraper_order` karena itu sama dengan menebak.
8. **NFKC aktif** (735 baris berubah, mis. "𝙗𝙖𝙜𝙪𝙨" → "bagus", "…" → "..."). Bisa dimatikan dengan `--no-nfkc`.
9. **Stabilitas `review_id` saat scrape ulang.** ID bergantung pada nama file dan nomor baris. Jika file di-scrape ulang dengan nama sama tetapi urutan baris berubah, ID yang sama akan menunjuk ulasan lain. Perlu kunci lain (mis. hash teks + timestamp) bila hasil lintas-run akan digabung.

---

## 4. Temuan di luar tugas (kode lama, TIDAK diubah)

Diverifikasi dengan memanggil fungsi lama secara hanya-baca terhadap 403 file. Perannya dibandingkan dengan `column_map.csv`.

- `src/preprocessing.py` · `METADATA_COLUMNS` memuat `"data"`. Akibatnya 34 file yang kolom `data`-nya berisi ulasan kehilangan kolom itu.
- `detect_review_columns`: bila ada kolom bernama `review`, hanya kolom itu yang dipakai. Di 8 file, kolom teks lain (`review1`, `review2`, `review 2`) terlewat.
- Pola nama kolom kode lama ikut memilih kolom non-ulasan:
  - `XYk98l` (timestamp) di 93 file;
  - `img-more-text` ("+2") di 40 file;
  - `HcSdrS src` (URL gambar) di 32 file;
  - `review-attribute` (label aspek) di 29 file;
  - kolom tanggal `data3`/`data4` di 17 file;
  - kolom balasan penjual di 10 file.
- `normalize_text_series`: regex `(.)\1{2,}` juga memotong angka. Contoh: `"baterai 5000mAh"` → `"baterai 500mAh"`, `"harga 1000000"` → `"harga 100"`.
- `clean_timestamp`: `pd.to_datetime` tanpa format membaca `"07/03/2026"` sebagai 2026-07-03 (bulan lebih dulu). Format ini tidak muncul di data sekarang, tetapi berisiko pada data scrape ulang.
- `clean_chunk`: duplikat dibuang diam-diam per chunk (`drop_duplicates` product + review), tanpa catatan.
- `src/config.py`: `sentiment_model_name` didefinisikan dua kali; definisi kedua (`None`) menimpa yang pertama.

---

## 5. File output (`outputs/cleaned/`)

| File | Isi |
|---|---|
| `reviews_clean.parquet` | 64.699 baris × 27 kolom (skema bagian 5 tugas + `aspect_tags`, `timestamp_status`, `n_fragments`, `seller_reply_suspect`, `file_dup_group`, `schema_family`) |
| `reports/column_map.csv` | peran tiap file×kolom + profil + bukti + 5 contoh nilai |
| `reports/file_manifest.csv` | per file: baris sumber/kosong/keluar, cek seimbang, flag, % NaT, kolom yang dipakai; 2 file dikecualikan juga tercantum |
| `reports/cleaning_audit.csv` | tiap aturan: jumlah sebelum/sesudah/terdampak + alasan |
| `reports/fragment_counts.csv` | jumlah sel per peran kolom × jenis nilai |
| `reports/duplicate_files.csv` | grup hash identik |
| `reports/aspect_tag_candidates.csv` | kandidat tag aspek + kolom `setuju` kosong |
| `reports/preset_phrase_candidates.csv` | kandidat frasa templat platform |
| `reports/product_summary.csv` | per produk: jumlah include, duplikat, NaT, `lolos_3/5/10` (tidak memfilter) |
| `reports/timestamp_formats.csv` | pola timestamp per keluarga skema + status |
| `reports/excluded_files.csv`, `validation_checks.csv`, `output_hashes.csv`, `tabel_contoh*.csv` | — |

Keluaran cleaning berupa **satu tabel** `"outputs/cleaned/reviews_clean.parquet"`
dan folder laporan `"outputs/cleaned/reports/"`. Tidak ada lagi folder per produk,
`part_*.parquet`, atau `_meta.json` pada format baru. Pintu masuk resmi untuk
membaca tabel ini adalah `load_clean_reviews`:

```python
from src.cleaning import load_clean_reviews

reviews = load_clean_reviews()
semua_reviews = load_clean_reviews(only_included=False)
sampel = load_clean_reviews(columns=["nama_toko", "seq_in_product"]).head(3)
# Untuk lokasi khusus, gunakan path file atau CleaningConfig(output_dir=...).
reviews_lain = load_clean_reviews(path="outputs/cleaned_v2/reviews_clean.parquet")
```

Secara bawaan hanya baris `include_for_inference == True` yang dikembalikan.
Kedua mode diurutkan menurut `product_id`, lalu `seq_in_product`; indeks direset,
tetapi nomor urut asli, tipe kolom, dan timestamp `NaT` dipertahankan.
`columns=None` mengembalikan semua kolom. Jika `columns` diberikan, empat kolom
kontrak inferensi (`review_id`, `product_id`, `text_light`, `timestamp`) tetap
disertakan. `config` menerima `CleaningConfig`; `path` mengalahkan lokasi dalam
konfigurasi. File yang tidak tersedia menghasilkan `FileNotFoundError` yang
menjelaskan lokasi dan perintah cleaning. Loader hanya membaca hasil yang ada.

`DataPreprocessor` sudah digantikan oleh `src.cleaning`. Untuk folder yang
berisi `reviews_clean.parquet`, `clean_single_file()` menolak penulisan dengan
pesan `format baru terdeteksi; pakai python -m src.cleaning.run`; penolakan ini
juga diteruskan oleh `clean_all()` saat memanggil fungsi tersebut, termasuk
ketika `overwrite=True`. `get_cleaned_files()` dan `inspect_cleaned_data()`
menolak format baru dengan petunjuk memakai `src.cleaning.load_clean_reviews()`.
Perilaku format lama tetap tersedia jika file tunggal tersebut tidak ada.

Pemeriksaan 2026-10-10 menemukan 593 file `.done` dan tidak ada isi lain di
`"outputs/checkpoints/cleaning/"`. Setelah memastikan kode `src/cleaning` tidak
membaca atau menulis checkpoint tersebut, semua penanda dihapus per file:
**593 sebelum → 0 sesudah**, tanpa pemindahan ke `_to_delete/`. Skip pada kode
lama sebenarnya mensyaratkan `.done` sekaligus `_meta.json`; `get_cleaned_files`
membaca manifest lama, sedangkan `inspect_cleaned_data` menelusuri parquet.
Tidak ditemukan pembaca format per produk di notebook atau modul lain yang
memerlukan perubahan desain. `RM1_execution_report.md` dan `docs/audit_pipeline.md`
masih memuat uraian historis pipeline lama; keduanya di luar lingkup pembaruan ini.

Menjalankan ulang (folder output harus kosong atau belum ada; isi lama `outputs/cleaned` dari run 2026-10-03 sudah dihapus atas permintaan Asyifa):

```
python -m src.cleaning.run --output-dir "outputs/cleaned_v2" \
    [--aspect-tag-vocab "reports/aspect_tag_candidates.csv"] \
    [--exclude-duplicate-file-groups] [--allow-date-only-technical-dup] [--no-nfkc] \
    [--raw-dir "/path/data baru" --expected-mapping-rows none --expected-ok-files none]
```
