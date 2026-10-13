# Verifikasi parquet proposal v2 — kamus alay

## Artefak dan SHA-256

- input: "outputs/cleaned/reviews_clean (2).parquet"; SHA-256 `c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0`
- output: "outputs/cleaned/reviews_clean_v2.parquet"; SHA-256 `dc20870f56b1b8e3d85597f51eeb0021cfd3065896518da5d4c462cb729889bb`
- slang_csv: "references/slang.csv"; SHA-256 `f4320a27ec257a508cec6bdf825aef9f2a3232658f0d18b462693c5340b4f7e2`
- report: "reports/verifikasi_proposal_v2_alay.md"
- top_csv: "reports/top_penggantian_slang.csv"

## Terverifikasi

Nilai, dtype, indeks, nama serta urutan seluruh kolom lama dibandingkan persis. Keluaran proposal diregenerasi memakai fungsi preprocessing yang sama dengan jejak setiap penggantian slang aktual, lalu dibandingkan dengan parquet v2.

- n_baris: 64699
- n_berubah: 45082
- n_kosong: 58
- slang_dijalankan: True
- jumlah_slang_dinormalkan: 61003
- jumlah_pasangan_slang_terpakai: 1150
- n_kolom_lama: 27
- n_kolom_v2: 29
- jumlah_baris_identik: True
- urutan_review_id_seq_identik: True
- seluruh_kolom_lama_identik: True
- dtype_kolom_lama_identik: True
- text_light_identik: True
- text_lex_tersedia: False
- text_lex_identik: None
- regenerasi_text_proposal_identik: True
- flag_kosong_identik: True
- n_include_for_inference: 64332
- n_technical_duplicate: 347
- n_technical_duplicate_masih_include: 0
- n_text_duplicate: 4668
- n_text_duplicate_masih_include: 4647
- n_kosong_masih_include: 45
- sha256_input_sebelum: c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0
- sha256_input_sesudah: c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0
- sha256_input_tidak_berubah: True

Kolom lama yang diperiksa: `review_id`, `product_id`, `nama_toko`, `nama_produk`, `review`, `text_light`, `aspect_tags`, `timestamp`, `timestamp_has_time`, `timestamp_raw`, `timestamp_status`, `row_order`, `source_file`, `fragment_cols`, `n_fragments`, `n_chars`, `n_words`, `is_empty`, `is_short`, `is_technical_duplicate`, `is_text_duplicate`, `dup_group_size`, `seller_reply_suspect`, `include_for_inference`, `seq_in_product`, `file_dup_group`, `schema_family`.

`text_lex` tidak tersedia pada input maupun v2. Tidak ada kolom yang dibuat untuk menggantikannya, dan perbandingan nilainya tidak dapat dilakukan (`text_lex_identik=null`). Seluruh kolom lama yang benar-benar tersedia tetap dibandingkan persis.

## Per aturan

| Aturan | Baris berubah |
| --- | ---: |
| hapus_url | 6 |
| hapus_emoji_karakter_khusus | 27710 |
| ringkas_tanda_baca | 10478 |
| hapus_angka_panjang | 11 |
| normalisasi_typo_huruf_berulang | 4143 |
| normalisasi_slang | 26594 |
| rapikan_spasi | 28998 |

Satu ulasan dapat berubah pada beberapa aturan; hitungan baris antartahap tidak dijumlahkan.

## PROTECTED_TERMS

Pencocokan masing-masing istilah tidak peka kapital, pada batas kata, dan menerima spasi fleksibel. Setiap istilah dihitung sendiri: `ori` dalam `bukan ori` ikut dihitung. Kehilangan diperiksa per baris agar kenaikan pada ulasan lain tidak menutup kehilangan. Kenaikan dapat terjadi ketika pembersihan simbol/typo membentuk batas kata atau ejaan yang cocok; kenaikan tidak berarti istilah asal dihapus.

| Istilah | Sebelum | Sesudah | Hilang per baris | Baris kehilangan | Bertambah per baris |
| --- | ---: | ---: | ---: | ---: | ---: |
| kw | 230 | 232 | 0 | 0 | 2 |
| ori | 3485 | 3523 | 0 | 0 | 38 |
| original | 2880 | 2885 | 0 | 0 | 5 |
| palsu | 412 | 416 | 0 | 0 | 4 |
| replika | 16 | 16 | 0 | 0 | 0 |
| imitasi | 1 | 1 | 0 | 0 | 0 |
| hdc | 26 | 26 | 0 | 0 | 0 |
| bukan ori | 62 | 64 | 0 | 0 | 2 |
| sesuai deskripsi | 1009 | 1009 | 0 | 0 | 0 |

## Tinjauan manual penggantian slang

- CSV: "reports/top_penggantian_slang.csv"; 100 pasangan dari 1150 pasangan yang terpakai.
- Peringkat berdasarkan frekuensi penggantian aktual menurun, lalu slang dan baku menaik.
- Contoh berasal hanya dari review pengguna. Tiga posisi ulasan berbeda dipilih tanpa pengembalian memakai `random_state=42` per pasangan, setelah posisi diurutkan.
- Jika pasangan muncul pada kurang dari tiga ulasan, kolom contoh sisanya kosong; tidak ada konteks buatan. Posisi baris menggunakan indeks posisi mulai nol.
- Pasangan top-100 dengan kurang dari tiga ulasan: 0.
- Frekuensi menghitung kejadian; satu ulasan dapat menyumbang beberapa kejadian pasangan sama.

## Sumber dan batasan

Kamus lokal mengikuti "references/slang_PROVENANCE.md" dan profil "reports/profil_kamus_alay.md". Sumber: Salsabila dkk. (2018), ekspor loader SEACrowd; lisensi dinyatakan Unknown. Audit ini tidak mengunduh sumber atau memuat kolom context kamus. Audit kesetaraan data tidak membuktikan kualitas linguistik normalisasi; contoh perlu ditinjau.
