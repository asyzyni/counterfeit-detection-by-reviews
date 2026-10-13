# Integrasi kamus alay dan parquet proposal v2

Tanggal: 2026-10-13 (Asia/Jakarta).

## Ruang lingkup dan status awal

Lima berkas wajib dibaca sebelum edit: "src/cleaning/proposal.py", "src/cleaning/run_proposal.py",
"src/cleaning/config.py", "tests/cleaning/test_proposal.py", dan "docs/inferensi_sentimen.md".
Satu-satunya perintah Git adalah pemeriksaan status awal yang diminta pengguna:

```text
 M .DS_Store
?? docs/hmm_rm2.md
?? src/temporal/
?? tests/temporal/
```

Berkas tersebut sudah berubah/belum dilacak sebelum pengerjaan. Tidak ada commit, push, staging,
perubahan ".gitignore", penghapusan, pemindahan, atau penggantian nama input. Tidak ada jaringan/unduhan.
"src/cleaning/run_proposal.py" dan "src/cleaning/config.py" tidak diubah.
Kode transformasi proposal hanya ditambah callback opsional untuk jejak penggantian aktual;
keluaran diregenerasi dan dibandingkan persis oleh audit. Model, max-length, jalur B,
HMM, XGBoost, pelabelan, serta aturan include/duplikasi tidak diubah.

## Terverifikasi dari ekspor lokal

Kedua hash input cocok sebelum pekerjaan; hash diperiksa kembali setelah keluaran dibuat.
Kamus UTF-8 strict: 15.006 record CSV, 4.330 slang unik (casefold+trim), 4.409 pasangan unik,
71 kunci multi-baku dengan 150 pasangan. Ada 15.397 LF, termasuk newline dalam komentar berkutip;
jumlah record harus dibaca dengan parser CSV, bukan wc. Tidak ada slang/formal kosong atau identik,
non-ASCII, maupun penanda mojibake pada dua kolom tersebut.

Sebanyak 1.337 kunci muncul pada lebih dari satu record; 1.293 memiliki lebih dari satu context berbeda.
Angka ini memverifikasi pengulangan per konteks, tetapi tidak menjelaskan selisih terhadap angka 3.592
kata yang dilaporkan halaman HF. Arti in_dictionary tidak ditafsirkan: nilai literal True 13.722,
False 1.284. Categories memiliki 190 nilai teks berbeda, tanpa digunakan untuk menyaring.
Isi context Instagram tidak disalin ke laporan atau CSV turunan.

## Asumsi filter dan rekonsiliasi

Kunci casefold+whitespace; nilai baku hanya whitespace. Duplikat ditentukan setelah normalisasi.
Identitas tidak peka kapital. PROTECTED_TERMS dicocokkan sebagai substring literal pada kedua sisi,
termasuk ori dalam favorit/tutorial. Ambiguitas dihitung sebelum filter lainnya, tanpa memilih baku
berdasarkan frekuensi. Kunci 1–2 karakter dikeluarkan. Satu alasan utama per record menurut urutan
kosong, duplikat, identitas, protected_terms, multi_baku, satu karakter, dua karakter.

| Alasan utama | Record dibuang |
| --- | ---: |
| Kosong | 0 |
| Duplikat pasangan | 10.597 |
| Slang sama dengan baku | 0 |
| PROTECTED_TERMS | 9 |
| Multi-baku | 150 |
| Satu karakter | 7 |
| Dua karakter | 119 |
| **Total dibuang** | **10.882** |
| **Diterima** | **4.124** |

15.006 = 10.882 + 4.124; 285 pasangan unik tidak diterima. Kunci hilang **206/4.330 = 4,757506%**:
gerbang >30% **tidak terpicu**. Duplikasi konteks tidak dianggap kunci hilang. Profil sebelum filter
mempunyai 13 kunci satu karakter dan 131 dua karakter; sebagian mendapat alasan konflik/protected.
Seluruh kunci dua karakter terdaftar di "reports/profil_kamus_alay.md".

Header turunan terverifikasi persis `slang,baku` dan `slang,baku,alasan`; tidak ada kolom context.
Nol kunci diterima yang bentrok protected/multi-baku/pendek. Urutan sumber dibalik dan kedua CSV
terbukti dapat direproduksi byte demi byte; loader produksi membaca 4.124 pasangan.

## Perintah pembuatan dan keluaran asli

```bash
python -m src.cleaning.run_proposal --input "outputs/cleaned/reviews_clean (2).parquet" --slang-csv "references/slang.csv" --output "outputs/cleaned/reviews_clean_v2.parquet" --report "reports/preprocessing_proposal_v2.md"
```

Exit code 0. Stdout:

```text
n_baris: 64699
n_berubah: 45082
n_kosong: 58
slang_dijalankan: True
jumlah_slang_dinormalkan: 61003
baris_berubah_per_aturan: {'hapus_url': 6, 'hapus_emoji_karakter_khusus': 27710, 'ringkas_tanda_baca': 10478, 'hapus_angka_panjang': 11, 'normalisasi_typo_huruf_berulang': 4143, 'normalisasi_slang': 26594, 'rapikan_spasi': 28998}
n_include_for_inference: 64332
n_technical_duplicate: 347
n_technical_duplicate_masih_include: 0
n_text_duplicate: 4668
n_text_duplicate_masih_include: 4647
n_kosong_masih_include: 45
jumlah_baris_identik: True
urutan_review_id_seq_identik: True
seluruh_kolom_lama_identik: True
Keluaran: "outputs/cleaned/reviews_clean_v2.parquet"; laporan: "reports/preprocessing_proposal_v2.md"
```

PyArrow juga mengeluarkan empat pesan sysctlbyname Operation not permitted pada informasi cache/NEON;
proses tetap exit 0 dan round-trip parquet serta audit persis selesai. Pesan tersebut bukan hasil validasi data.

## Verifikasi artefak tersimpan

Audit "src/cleaning/audit_proposal.py" exit 0 dengan --expected-rows 64699 dan --expected-columns 27;
perintah lengkap ada di "docs/inferensi_sentimen.md". Laporan lama "reports/verifikasi_proposal_v2.md"
tidak ditimpa; laporan baru bernama "reports/verifikasi_proposal_v2_alay.md".

- 64.699 baris; 27 kolom lama utuh, identik nilai/dtype/indeks/urutan; 29 kolom pada v2.
- Urutan review_id dan seq_in_product identik; text_light identik.
- **text_lex tidak ada di input maupun v2**: tidak ada nilai yang dapat dibandingkan; kolom tidak dibuat.
  Audit awal berhenti atas asumsi kolom wajib ini sebelum menulis laporan. Audit diperbaiki agar
  menyatakan ketersediaannya false dan kesetaraan null, lalu dijalankan ulang pada artefak v2 yang sama.
- text_proposal diregenerasi dan identik dengan yang tersimpan; text_proposal_kosong sesuai hasil kosong.
- 45.082 ulasan berubah; 58 kosong; 61.003 penggantian dari 1.150 pasangan; slang mengubah 26.594 ulasan.
- Flags include/duplikasi merupakan bagian 27 kolom yang diperiksa persis dan tidak dihitung ulang.
- CSV top-100 memiliki 100 pasangan dan 300 contoh; setiap contoh diverifikasi cocok dengan posisi,
  review_id, review asli, serta text_proposal pada tabel. Tiga posisi berbeda per pasangan dipilih
  tanpa pengembalian memakai random_state=42. Tidak ada konteks dari kamus mentah.
- Top-100 mencakup 47.096 kejadian dari 61.003 total penggantian. Pemeringkatan memakai frekuensi,
  bukan pemilihan baku. Sampel tetap perlu tinjauan linguistik/domain; misalnya dah → deh terjadi
  1.003 kali sesuai kamus sumber, bukan pemetaan yang dikarang atau dikoreksi otomatis.

| PROTECTED_TERMS | Sebelum | Sesudah | Hilang per baris |
| --- | ---: | ---: | ---: |
| kw | 230 | 232 | 0 |
| ori | 3.485 | 3.523 | 0 |
| original | 2.880 | 2.885 | 0 |
| palsu | 412 | 416 | 0 |
| replika | 16 | 16 | 0 |
| imitasi | 1 | 1 | 0 |
| hdc | 26 | 26 | 0 |
| bukan ori | 62 | 64 | 0 |
| sesuai deskripsi | 1.009 | 1.009 | 0 |

Hitungan per istilah tidak peka kapital, batas kata, spasi fleksibel. Ori dalam bukan ori dihitung
untuk masing-masing istilah. Tidak ada penurunan pada baris asal; kenaikan tidak menghapus bukti asal.
Tahap pembersihan simbol/typo dapat menghasilkan kecocokan baru. text_proposal hanya untuk IndoBERT.

## Tes

Perintah: `python -m pytest tests -q -rs`. Keluaran asli lengkap (termasuk traceback dan alasan skip)
disimpan di "reports/hasil_tes_integrasi_alay.txt". Ringkasan asli run akhir:

```text
1 failed, 176 passed, 2 skipped in 6.48s
```

Exit code **1**, jadi seluruh suite belum hijau. Semua 42 tes baru (34 konverter + 8 audit) serta tes
proposal lama lulus. Kegagalan adalah TestHmmlearnParity.test_forward_matches_hmmlearn_score_and_posterior
pada "tests/temporal/test_pipeline.py": "src/temporal/fitting.py:120" menerima covars_ (2, 3, 3),
namun mengharapkan (2, 3). Berkas temporal sudah ada pada status awal dan tidak disentuh karena
HMM di luar lingkup. Dua tes model nyata dilewati karena INDOBERT_SENTIMENT_MODEL_DIR tidak ditetapkan.

## Keputusan pengguna dan usulan saja

- Pengguna telah memilih agar "references/slang.csv" tetap lokal dan diabaikan Git. Aturan `*.csv`
  yang sudah ada mengabaikan CSV mentah/turunan; tidak ada pengecualian untuk melacak kamus.
- Lisensi dicatat **Unknown** menurut informasi SEACrowd dari pengguna, tanpa verifikasi daring;
  provenance menyebut sitasi Salsabila dkk. (2018), asal loader, versi ekspor yang dilaporkan, dan hash terukur.
- Belum ada keputusan memasukkan kembali kunci dua karakter; 131 kunci tetap dikeluarkan sesuai aturan awal.
- Usulan berikut hanya ditampilkan, belum diterapkan pada ".gitignore":

```diff
--- a/.gitignore
+++ b/.gitignore
@@
 *.csv
+references/sumber/
```

## Berkas baru

- "src/cleaning/build_slang_dictionary.py"
- "src/cleaning/audit_proposal.py"
- "tests/cleaning/test_build_slang_dictionary.py"
- "tests/cleaning/test_audit_proposal.py"
- "references/slang.csv"
- "references/slang_dikecualikan.csv"
- "references/slang_PROVENANCE.md"
- "outputs/cleaned/reviews_clean_v2.parquet"
- "reports/profil_kamus_alay.md"
- "reports/preprocessing_proposal_v2.md"
- "reports/verifikasi_proposal_v2_alay.md"
- "reports/top_penggantian_slang.csv"
- "reports/hasil_tes_integrasi_alay.txt"
- "reports/integrasi_kamus_alay_v2.md"

## Berkas diubah

- "src/cleaning/proposal.py"
- "docs/inferensi_sentimen.md"

## SHA-256 akhir

| Berkas | SHA-256 |
| --- | --- |
| "references/sumber/colloquial-indonesian-lexicon.csv" | `48e7450875b75935cf15b200524e97927a8c27f8169e658cf4a73993c7818d76` |
| "references/slang.csv" | `f4320a27ec257a508cec6bdf825aef9f2a3232658f0d18b462693c5340b4f7e2` |
| "references/slang_dikecualikan.csv" | `21293d04e4cb71c66969cecde83055b1e321f1a8b62f1fd4d7cdd2fe3cb8a52a` |
| "outputs/cleaned/reviews_clean (2).parquet" | `c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0` |
| "outputs/cleaned/reviews_clean_v2.parquet" | `dc20870f56b1b8e3d85597f51eeb0021cfd3065896518da5d4c462cb729889bb` |
| "reports/top_penggantian_slang.csv" | `ef2b555936792549bcff69d0eb44493af2e75392d21094965416865c434a224a` |
