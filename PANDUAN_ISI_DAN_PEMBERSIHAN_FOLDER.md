# Panduan Isi dan Pembersihan Folder Kode TA

Tanggal pemeriksaan: **14 Oktober 2026**. Panduan ini menjelaskan fungsi setiap file yang ditemukan dan membantu menentukan mana yang perlu disimpan, dirapikan, diarsipkan, atau dibersihkan.

**Pembaruan konfigurasi 16 Oktober 2026:** panjang inferensi IndoBERT mengikuti `ExperimentConfig.sentiment_max_length=128`, sama dengan fine-tuning; kapasitas arsitektur tetap 512. Modul, CLI, dan notebook membaca sumber konfigurasi yang sama. Panjang berbeda memerlukan `--allow-length-mismatch`, disertai peringatan dan catatan metadata. v2 berjalan dengan 512, digantikan oleh keputusan 128; rincian v2 di inventaris historis di bawah tetap memakai angka eksekusi aslinya. Metadata v3/v4 mencatat 128. Prediksi v5 identik dengan v4 melalui `review_id`; provenance migrasi v5 tidak merekam `max_length` sendiri. Pada v4/v5, 123 dari 73.184 ulasan yang diinferensi terpotong (0,17%).

Inventaris mencakup **228 file yang sudah ada**, berukuran total **185.28 MiB**, di luar direktori internal `.git/` dan file panduan ini sendiri. File tersembunyi dan cache turut dicatat. `.git/` dijelaskan sebagai satu kesatuan karena merupakan basis data internal Git, bukan kumpulan berkas kerja yang perlu dibersihkan satu per satu. Ada **72 file cache/metadata sistem**, total **1.10 MiB**.

Dasar pemeriksaan: daftar berkas, isi kode dan notebook tanpa eksekusi, header/metadata data, laporan hasil, serta hash SHA-256 untuk duplikat. Pipeline, inferensi model, dan rangkaian tes tidak dijalankan untuk membuat panduan ini. Rekomendasi merupakan hasil peninjauan; **belum ada file yang dipindahkan atau dihapus**. Kondisi ini adalah snapshot, bukan inventaris yang diperbarui otomatis.

Navigasi: [prioritas pembersihan](#2-prioritas-pembersihan) · [peta folder](#3-peta-folder) · [setiap file kerja](#4-penjelasan-setiap-file-kerja) · [cache](#5-file-cache-dan-metadata-sistem) · [masalah penataan](#7-masalah-penataan-yang-perlu-dibereskan) · [usulan struktur](#8-usulan-struktur-setelah-ditata) · [checklist](#9-checklist-tindakan-berikutnya).

## 1. Cara membaca rekomendasi

| Rekomendasi | Arti dan tindakan |
| --- | --- |
| **SIMPAN** | Masih merupakan sumber, dependensi, data, hasil, atau bukti penelitian yang berguna. |
| **RAPIKAN** | Pertahankan isinya; benahi lokasi, nama, atau referensinya secara terkoordinasi. |
| **ARSIP BERSYARAT** | Dapat dipisahkan dari pekerjaan aktif setelah perannya selesai dan dependensinya diperiksa. Arsip bukan penghapusan. |
| **TINJAU** | Status penggunaan atau kelengkapan belum cukup untuk menyimpulkan aman dihapus. |
| **BERSIHKAN** | Cache/metadata yang dapat dibuat ulang, atau notebook kosong yang tidak menyimpan pekerjaan. |
| **DUPLIKAT** | Kontennya terbukti identik; boleh memilih satu salinan setelah semua referensi diarahkan ke salinan yang dipertahankan. |

Nama seperti `v2`, `v3`, `(2)`, `debug`, atau `uji` **tidak otomatis berarti file boleh dihapus**. Perhatikan hubungan sumber–hasil dan kebutuhan pelacakan eksperimen.

## 2. Prioritas pembersihan

1. **Mulai dari cache:** seluruh `__pycache__/`, `.pytest_cache/`, dan empat `.DS_Store`. Ini mengurangi 72 file (1.10 MiB) tanpa menghapus kode atau hasil eksperimen.
2. **Tinjau dua notebook 0 byte:** `check.ipynb` dan `notebooks/00_debug.ipynb`. Keduanya tidak berisi JSON notebook maupun sel kerja. Boleh dibuang jika tidak sedang menjadi placeholder. Sebaliknya, `.gitkeep` dan `__init__.py` kosong punya fungsi dan perlu dipertahankan.
3. **Konsolidasikan dua pasangan duplikat identik** pada bagian berikut. Penghematan satu salinan dari masing-masing pasangan sekitar 518.5 KiB; manfaat utamanya mengurangi kebingungan.
4. **Kelompokkan eksperimen lama:** notebook debug, uji inferensi 200 baris, dan laporan parsial dapat masuk arsip setelah tidak menjadi rujukan aktif. Pindahkan hasil uji bersama ringkasan/log terkait agar asal-usulnya tetap jelas.
5. **Pertahankan rangkaian hasil dasar, v2, dan v3 untuk sementara.** Hasil sekuens/audit yang ada berasal dari sentimen v2. v2 dan v3 juga memakai konfigurasi panjang token berbeda; keduanya bukan sekadar salinan file.
6. **Selaraskan dokumentasi dan path** sebelum memindahkan kode atau data. Beberapa nama di README sudah tidak cocok dengan file yang tersedia, dan cleaning merujuk sumber data di luar folder ini.

Dataset dan keluaran besar banyak diabaikan oleh `.gitignore`. **Diabaikan Git tidak berarti tidak penting**, dan Git belum tentu menyediakan salinan pemulihannya. Backup data serta hasil yang dipakai dalam TA sebelum pengarsipan atau penghapusan permanen.

### Duplikat yang sudah dibuktikan

Kesamaan di bawah diperiksa dari seluruh byte, bukan hanya kemiripan nama.

| File A | File B | Ukuran per file | Saran |
| --- | --- | --- | --- |
| [01_pipeline.png](01_pipeline.png) | [docs/01_pipeline.png](docs/01_pipeline.png) | 471.4 KiB | Jadikan versi di `docs/` sebagai salinan utama; periksa referensi gambar sebelum menghapus salinan root. |
| [data/raw/model data.csv](data/raw/model%20data.csv) | [data/raw/modeling data.csv](data/raw/modeling%20data.csv) | 47.1 KiB | Notebook cleaning masih menyebut `model data.csv`; periksa path yang dipakai sebelum memilih satu salinan. |

SHA-256 pasangan gambar: `f9d2e8dfbec9c09bc9137c6db420f67a5b954afc61f2a0a88358a125fd90bdbb`.

SHA-256 pasangan CSV: `57166821926edff3f42ca80c942af0f6fbbca1c78ee962e69b5bd0fe0c178c21`.

### Hubungan file yang penting sebelum menghapus

```text
Sumber cleaning: direktori eksternal yang dikonfigurasi
    → outputs/cleaned/reviews_clean (2).parquet  [hasil dasar yang tersedia]
        → reviews_clean_v2.parquet → sentimen_ulasan_v2.csv
                                      → sekuens_produk.csv
                                      → outputs/audit/*
        → reviews_clean_v3.parquet → sentimen_ulasan_v3.csv
```

- **`reviews_clean (2).parquet` perlu disimpan.** Hash-nya cocok dengan entri `reviews_clean.parquet` dalam `outputs/cleaned/reports/output_hashes.csv`, dan laporan proposal v2/v3 menggunakannya sebagai input. Nama `(2)` bukan bukti duplikasi yang boleh dibuang.
- **v2 masih menjadi sumber ekspor sekuens yang tersimpan.** Ini tercatat di `outputs/audit/ekspor_sekuens_ringkasan.json` dan `ekspor_sekuens_laporan.md`. Jangan membuang CSV sentimen v2 selama sekuens/audit tersebut masih digunakan.
- **v3 belum menggantikan semua keluaran hilir.** Preprocessing v3 menambahkan normalisasi Unicode; ringkasan inferensi v2 mencatat `max_length=512`, sedangkan v3 `max_length=128`. Perbandingan kedua hasil perlu mempertimbangkan dua perubahan ini.
- **Kode dan artefak tidak selalu berasal dari revisi yang sama.** Kode ekspor saat ini menangani `product_id_lama`, sementara audit CSV yang tersimpan belum memuat kolom tersebut. Jangan menganggap seluruh hasil lama sudah memenuhi kontrak kode terbaru tanpa verifikasi ulang.
- **Data lokal belum mencakup semua sumber cleaning.** Ada 35 CSV di `data/raw/`, sedangkan manifest cleaning mencatat 403 file diproses. Default `CleaningConfig.raw_dir` mengarah ke `/Users/asyzyni/Desktop/TA/Data Beneran`. Direktori eksternal tersebut tidak termasuk inventaris ini.
- **Ada path default yang belum cocok dengan berkas tersedia.** Loader cleaning menunjuk `outputs/cleaned/reviews_clean.parquet`, tetapi yang tersedia adalah `reviews_clean (2).parquet`, v2, dan v3. Jangan mengganti nama tanpa menyelaraskan pemanggil dan catatan sumber.
- **Default antar-tahap belum seragam.** `run_proposal.py` menulis v3, sedangkan `run_sentiment_inference.py` masih membaca v2 secara default. Pilih input secara eksplisit ketika menjalankan tahap berikutnya; keberadaan v3 tidak otomatis mengubah sumber inferensi.
- **Kode lama masih punya pemakai.** `src/preprocessing.py` masih diuji di `tests/cleaning/test_loader.py`; `src/inference.py` masih diuji di `tests/test_inference_smoke.py`. Menghapusnya memerlukan keputusan migrasi, bukan sekadar pembersihan folder.

## 3. Peta folder

| Lokasi | Fungsi utama | Arah penataan |
| --- | --- | --- |
| Root | README, dependensi, skrip EDA, diagram, notebook kosong. | Sisakan pintu masuk proyek dan panduan; pertimbangkan memindahkan skrip EDA ke `scripts/` setelah path diperbaiki. |
| `src/` | Implementasi cleaning, inferensi sentimen, ekspor sekuens, fitur leksikal, dan HMM. | Pertahankan struktur paket; bersihkan cache saja terlebih dahulu. |
| `tests/` | Tes unit/integrasi serta data sintetis untuk memeriksa kode. | Pertahankan file sumber tes dan helper. |
| `notebooks/` | Eksperimen interaktif. | Pisahkan notebook aktif dan arsip; jangan jalankan sekadar untuk inventaris. |
| `Lagi/` | Notebook debug dalam susunan folder tersendiri. | Kandidat konsolidasi ke arsip notebook setelah memeriksa path internal. |
| `data/raw/` | CSV lokal dari pengumpulan/eksperimen data. | Simpan sebagai sumber; tidak semuanya merupakan data mentah murni. |
| `data/processed/` | Lokasi yang disiapkan untuk data olahan; saat ini hanya `.gitkeep`. | Boleh dipertahankan sebagai placeholder; jelaskan bahwa hasil cleaning aktual berada di `outputs/cleaned/`. |
| `references/` | Kamus slang aktif, sumber asli, daftar pengecualian, dan provenance. | Pertahankan bersama agar asal-usul normalisasi dapat ditelusuri. |
| `outputs/` | Data olahan, prediksi, sekuens, audit, dan log eksperimen. | Kelompokkan menurut tahap/versi setelah dependency dan path diselaraskan. |
| `reports/` | Laporan preprocessing, integrasi kamus, verifikasi, dan inferensi. | Pertahankan hasil final; arsipkan laporan parsial bila sudah digantikan. |
| `docs/` | Diagram dan laporan eksekusi RM1. | Pusat dokumentasi; beberapa dokumen yang disebut README sedang tidak tersedia. |
| `.git/` | Riwayat commit, index, refs, objek Git, dan konfigurasi lokal. | **SIMPAN utuh.** Jangan menghapus isi manual sebagai pembersihan proyek. |
| `.pytest_cache/`, `__pycache__/` | Cache otomatis alat pengembangan. | Boleh dibersihkan. |

## 4. Penjelasan setiap file kerja

Ukuran menggunakan KiB/MiB (kelipatan 1024), dibulatkan untuk memudahkan pemindaian. Seluruh path relatif terhadap folder `Kode TA`. File cache dijelaskan satu per satu pada bagian 5.


### 4.1 File di root

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [.gitignore](.gitignore) | 1.6 KiB | Aturan Git mengabaikan cache, CSV/Parquet/JSONL, outputs, bobot model, lingkungan virtual, dan berkas OS/IDE; juga memuat path absolut lama src/smoke_test. | Simpan; rapikan aturan path absolut lama dan tinjau pengecualian untuk CSV penting seperti references/slang.csv. Diabaikan Git bukan berarti aman dihapus. |
| [01_pipeline.png](01_pipeline.png) | 471.4 KiB | Diagram rancangan: cleaning; jalur IndoBERT–HMM dan TF-IDF–leksikon; fitur produk, ablasi, XGBoost, evaluasi. Isi identik dengan docs/01_pipeline.png berdasarkan SHA-256. | DUPLIKAT — pilih docs/01_pipeline.png sebagai salinan utama; salinan root boleh dihapus setelah referensi diperbarui. |
| [README.md](README.md) | 3.4 KiB | Dokumentasi utama tentang riset IndoBERT–HMM, instalasi, struktur folder, dan cleaning; beberapa nama notebook dan tautannya tidak cocok dengan berkas yang sekarang ada. | Simpan dan perbarui: koreksi nama notebook, tautan docs yang hilang, serta lokasi keluaran cleaning sesuai konfigurasi aktif. |
| [check.ipynb](check.ipynb) | 0 B | Berkas 0 byte, sehingga belum berisi JSON notebook maupun sel. | BERSIHKAN — 0 byte, tidak ada sel/isi yang hilang; boleh dihapus jika tidak dibutuhkan sebagai placeholder. |
| [eda_phone_reviews.py](eda_phone_reviews.py) | 7.0 KiB | Fungsi EDA ulasan ponsel: membaca CSV tepat di folder skrip, menghitung statistik rating/tanggal/duplikasi/produk; mengembalikan dict. REPORT_PATH menunjuk EDA_REPORT.md tetapi tidak dipakai, dan tidak ada pemanggilan run_eda saat skrip dijalankan langsung. | Arsipkan jika EDA lama tidak dipakai; bila dipertahankan di scripts/, perbaiki DATA_DIR lebih dulu karena relatif terhadap lokasi file dan CSV root kini tidak ada. |
| [requirements.txt](requirements.txt) | 178 B | Daftar dependensi Python (numpy, pandas, torch, transformers, datasets, accelerate, scikit-learn, hmmlearn, tqdm, pyarrow, jupyter) dengan batas versi minimum. | Simpan di root sebagai acuan instalasi; versi minimum bukan penguncian lingkungan eksperimen. |


### 4.2 Notebook eksperimen dan debug

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [Lagi/src/debug.ipynb](Lagi/src/debug.ipynb) | 27.1 KiB | Notebook 34 sel untuk smoke test trainer IndoBERT: tokenisasi, dataset dummy, collator, forward pass, softmax, metrik, dan training satu epoch. Mengimpor src.config/src.sentiment/src.split dan menulis debug_sentiment. | Pindahkan ke notebooks/debug/ hanya setelah memperbaiki penentuan PROJECT_ROOT; saat dijalankan dari Lagi/src, pencarian root dapat berhenti pada Lagi yang hanya berisi notebook, sehingga impor src dapat gagal. |
| [notebooks/00_debug.ipynb](notebooks/00_debug.ipynb) | 0 B | Berkas notebook kosong 0 byte; tidak memiliki sel atau hasil eksperimen. | BERSIHKAN — 0 byte, tidak ada sel/isi yang hilang; boleh dihapus jika tidak dibutuhkan sebagai placeholder. |
| [notebooks/clean-process-data.ipynb](notebooks/clean-process-data.ipynb) | 17.2 KiB | Eksperimen cleaning awal beberapa toko: parsing tanggal, normalisasi kolom, 423 baris gabungan, label KKA/Maja, dan sampel seimbang 60 baris; menulis data siap TA.csv, model data.csv, scraping_data_balanced.csv. | Arsipkan sebagai eksperimen awal; jalur absolut Model-Try dan path data relatif bergantung lokasi kerja. Perbaiki path sebelum dipindah/dijalankan ulang, dan jangan anggap label toko lama sebagai hasil pipeline terkini. |
| [notebooks/pipeline_llm_hmm_xgb.ipynb](notebooks/pipeline_llm_hmm_xgb.ipynb) | 43.6 KiB | Eksperimen membandingkan Qwen, Gemma, dan Phi pada PRDECT-ID untuk probabilitas token A/B/C/D; keluaran tersimpan menunjukkan pembatalan saat memuat Qwen. Notebook tidak berisi implementasi HMM atau XGBoost meskipun namanya menyebut keduanya. | Arsipkan/ubah nama agar mencerminkan eksperimen LLM; membutuhkan akses dataset/model Hugging Face dan sumber daya model bila dijalankan ulang. |


### 4.3 Dokumentasi dan laporan penelitian

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [docs/01_pipeline.png](docs/01_pipeline.png) | 471.4 KiB | Diagram arsitektur riset yang menampilkan dua jalur, pemisahan produk train/test, HMM, agregasi fitur, regresi, dan validasi. Salinan identik juga berada di root. | Simpan sebagai diagram utama; diagram menyatakan rancangan, bukan bukti semua komponennya sudah diimplementasikan. |
| [docs/RM1_execution_report.md](docs/RM1_execution_report.md) | 5.9 KiB | Laporan historis 3 Oktober 2026: cleaning 593 CSV menghasilkan 15.783 baris, smoke test pemetaan kelas lulus, inferensi penuh saat itu terhalang model belum tersedia. | Simpan sebagai riwayat/arsip RM1; jangan gunakan status blocker lama untuk menyimpulkan kondisi terkini, karena laporan inferensi run 2 mencatat eksekusi kemudian. |
| [reports/hasil_tes_integrasi_alay.txt](reports/hasil_tes_integrasi_alay.txt) | 3.7 KiB | Log pytest historis integrasi alay: 176 passed, 2 skipped, 1 failed. Kegagalan pada uji parity hmmlearn karena bentuk covars_ (2,3,3) berbeda dari (2,3). | Simpan bersama laporan integrasi sebagai bukti kegagalan yang tercatat; bukan hasil pengujian ulang saat inventaris dibuat. |
| [reports/inferensi_lokal_run2.md](reports/inferensi_lokal_run2.md) | 10.8 KiB | Laporan inferensi lokal 13 Oktober 2026 memakai MPS, text_proposal v2, max_length 512: 64.699 baris, 58 teks kosong, durasi sekitar 161 detik; mencatat validasi, perbandingan run 1, checksum dan isu Unicode. | Simpan bersama hasil sentimen v2; path Downloads untuk pembanding run 1 dan path output perlu dicatat ulang bila artefak dipindah. |
| [reports/integrasi_kamus_alay_v2.md](reports/integrasi_kamus_alay_v2.md) | 9.5 KiB | Catatan integrasi kamus slang: profil dan penyaringan sumber, perintah pembuatan kamus/parquet v2, verifikasi artefak, hasil tes, daftar berkas, dan SHA-256. | Simpan sebagai riwayat reproduksi v2; tautan docs/inferensi_sentimen.md dan docs/hmm_rm2.md merujuk berkas yang kini tidak ada. |
| [reports/preprocessing_proposal_v2.md](reports/preprocessing_proposal_v2.md) | 6.5 KiB | Audit preprocessing final v2 berbasis references/slang.csv: 64.699 baris, 45.082 berubah, 58 kosong, 61.003 penggantian slang; memuat rincian aturan dan 20 contoh. | Simpan dengan reviews_clean_v2.parquet; v3 adalah versi berbeda sehingga laporan ini berguna untuk reproduksi, bukan duplikat otomatis. |
| [reports/preprocessing_proposal_v2_parsial.md](reports/preprocessing_proposal_v2_parsial.md) | 7.1 KiB | Audit awal v2 sebelum kamus tersedia: 36.378 baris berubah, normalisasi slang belum dijalankan, parquet final belum dibuat; menyertakan checksum keluaran lama. | Kandidat arsip historis setelah laporan final v2 terjaga; tandai parsial agar tidak tertukar dengan hasil final. |
| [reports/preprocessing_proposal_v3.md](reports/preprocessing_proposal_v3.md) | 12.5 KiB | Audit preprocessing v3 dengan NFKC dan penanganan pemisah Unicode: 45.114 baris berubah, 61.341 penggantian slang; memuat verifikasi artefak, hasil tes, serta audit sisa karakter non-Latin. | Simpan sebagai laporan versi v3; tetap hubungkan dengan reviews_clean_v3.parquet dan kamus yang digunakan. |
| [reports/profil_kamus_alay.md](reports/profil_kamus_alay.md) | 10.1 KiB | Profil sumber kamus 15.006 record: encoding, 4.330 slang unik, pasangan ambigu, distribusi kategori, aturan filter, dan daftar kunci pendek/protected terms yang dikecualikan. | Simpan bersama kamus dan provenance karena menjelaskan bagaimana kamus produksi dipilih. |
| [reports/top_penggantian_slang.csv](reports/top_penggantian_slang.csv) | 116.8 KiB | Tabel audit 100 pasangan slang→baku teratas (20 kolom), frekuensi, jumlah ulasan terdampak, dan sampai tiga contoh review beserta teks hasil normalisasi. | Simpan jika diperlukan untuk tinjauan kualitas normalisasi v2; bukan dataset mentah dan dapat diregenerasi hanya jika input, kamus, dan versi kode yang sama tersedia. |
| [reports/verifikasi_proposal_v2.md](reports/verifikasi_proposal_v2.md) | 5.9 KiB | Catatan implementasi 11 Oktober 2026: perubahan inferensi/cleaning, 112 tes lulus dan 2 dilewati; kamus belum tersedia dan keluaran v2 final masih tertunda saat laporan ditulis. | Kandidat arsip historis; status menunggu kamus sudah memiliki tindak lanjut di integrasi_kamus_alay_v2.md dan verifikasi_proposal_v2_alay.md. |
| [reports/verifikasi_proposal_v2_alay.md](reports/verifikasi_proposal_v2_alay.md) | 4.6 KiB | Audit kesetaraan parquet v2 setelah kamus alay tersedia: hash input/output/kamus, seluruh kolom lama identik, regenerasi teks identik, hitungan protected terms, dan rujukan top-100 slang. | Simpan sebagai bukti integritas v2; jangan gabungkan dengan laporan verifikasi awal karena cakupan dan status eksekusinya berbeda. |


### 4.4 Data mentah/lokal dan placeholder data

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [data/processed/.gitkeep](data/processed/.gitkeep) | 0 B | File kosong agar folder data/processed tetap dapat disimpan dalam Git. | SIMPAN; tidak memakan ruang berarti dan menjelaskan folder keluaran yang masih kosong. |
| [data/raw/ mutiara_xiami redmi.csv](data/raw/%20mutiara_xiami%20redmi.csv) | 57.7 KiB | Ekspor scraping ulasan mutiara_xiami redmi; 49 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. Nama diawali spasi; ubah hanya setelah semua pemetaan/referensi path diperbarui. |
| [data/raw/1688.shopp_Oppo A31.csv](data/raw/1688.shopp_Oppo%20A31.csv) | 12.3 KiB | Ekspor scraping ulasan 1688.shopp_Oppo A31; 44 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/1688.shopp_Oppo A3s.csv](data/raw/1688.shopp_Oppo%20A3s.csv) | 740.3 KiB | Ekspor scraping ulasan 1688.shopp_Oppo A3s; 2390 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/1688.shopp_Vivo Y17.csv](data/raw/1688.shopp_Vivo%20Y17.csv) | 252.7 KiB | Ekspor scraping ulasan 1688.shopp_Vivo Y17; 894 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/1688.shopp_Vivo Y97.csv](data/raw/1688.shopp_Vivo%20Y97.csv) | 30.3 KiB | Ekspor scraping ulasan 1688.shopp_Vivo Y97; 104 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/188.toko_oppo a31.csv](data/raw/188.toko_oppo%20a31.csv) | 313.7 KiB | Ekspor scraping ulasan 188.toko_oppo a31; 807 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/566store_Oppo A3s.csv](data/raw/566store_Oppo%20A3s.csv) | 104.1 KiB | Ekspor scraping ulasan 566store_Oppo A3s; 369 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/566store_Oppo A5s.csv](data/raw/566store_Oppo%20A5s.csv) | 93.0 KiB | Ekspor scraping ulasan 566store_Oppo A5s; 324 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/566store_Vivo Y17.csv](data/raw/566store_Vivo%20Y17.csv) | 145.0 KiB | Ekspor scraping ulasan 566store_Vivo Y17; 476 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/566store_oppo a31.csv](data/raw/566store_oppo%20a31.csv) | 159.9 KiB | Ekspor scraping ulasan 566store_oppo a31; 405 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/566store_vivo y20s.csv](data/raw/566store_vivo%20y20s.csv) | 14.7 KiB | Ekspor scraping ulasan 566store_vivo y20s; 36 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/888TOKO_Vivo Y17.csv](data/raw/888TOKO_Vivo%20Y17.csv) | 12.1 KiB | Ekspor scraping ulasan 888TOKO_Vivo Y17; 18 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/Amoli TOKO 3C.csv](data/raw/Amoli%20TOKO%203C.csv) | 50.5 KiB | Ekspor scraping ulasan Amoli TOKO 3C; 40 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/Berliancahh.csv](data/raw/Berliancahh.csv) | 3.8 KiB | Ekspor scraping ulasan Berliancahh; 3 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/CRB phone second.csv](data/raw/CRB%20phone%20second.csv) | 7.7 KiB | Ekspor scraping ulasan CRB phone second; 6 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/Ext Gadget_iphone 12 mini .csv](data/raw/Ext%20Gadget_iphone%2012%20mini%20.csv) | 50.9 KiB | Ekspor scraping ulasan Ext Gadget_iphone 12 mini; 150 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. Ada spasi sebelum .csv; rapikan nama bersamaan dengan referensinya. |
| [data/raw/Ext Gadget_iphone 13.csv](data/raw/Ext%20Gadget_iphone%2013.csv) | 129.8 KiB | Ekspor scraping ulasan Ext Gadget_iphone 13; 405 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/GALAXI. 29.csv](data/raw/GALAXI.%2029.csv) | 24.5 KiB | Ekspor scraping ulasan GALAXI. 29; 21 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/GALAXI.29_Samsung Galaxy A01.csv](data/raw/GALAXI.29_Samsung%20Galaxy%20A01.csv) | 4.6 KiB | Ekspor scraping ulasan GALAXI.29_Samsung Galaxy A01; 25 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/Galaxi_Samsung A01.csv](data/raw/Galaxi_Samsung%20A01.csv) | 27.6 KiB | Ekspor scraping ulasan Galaxi_Samsung A01; 23 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/Gemilang Cell_Infinix Hot 50i.csv](data/raw/Gemilang%20Cell_Infinix%20Hot%2050i.csv) | 5.8 KiB | Ekspor scraping ulasan Gemilang Cell_Infinix Hot 50i; 31 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/Gemilang Cell_Realme C51s.csv](data/raw/Gemilang%20Cell_Realme%20C51s.csv) | 47.1 KiB | Ekspor scraping ulasan Gemilang Cell_Realme C51s; 71 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/HP SAMSUNG J2 PRIME NORMAL SIAP PAKAI SECOND BERKUALITAS.csv](data/raw/HP%20SAMSUNG%20J2%20PRIME%20NORMAL%20SIAP%20PAKAI%20SECOND%20BERKUALITAS.csv) | 65.9 KiB | Ekspor scraping ulasan HP SAMSUNG J2 PRIME NORMAL SIAP PAKAI SECOND BERKUALITAS; 57 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/K_KA SELULER (2).csv](data/raw/K_KA%20SELULER%20%282%29.csv) | 31.1 KiB | Ekspor scraping ulasan K_KA SELULER (2); 27 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/K_KA SELULER (counterfeit).csv](data/raw/K_KA%20SELULER%20%28counterfeit%29.csv) | 6.9 KiB | Ekspor scraping ulasan K_KA SELULER (counterfeit); 6 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/Panda Elektronik Store.csv](data/raw/Panda%20Elektronik%20Store.csv) | 659.2 KiB | Ekspor scraping ulasan Panda Elektronik Store; 558 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/data siap TA.csv](data/raw/data%20siap%20TA.csv) | 46.2 KiB | Tabel ulasan yang sudah dirapikan (423 baris): nama_toko, ulasan, tanggal; belum memakai skema input model berlabel. | SIMPAN sebagai dataset olahan awal; dapat dipindah ke data/interim setelah referensi notebook diperbarui. |
| [data/raw/gadget.ind_Infinix Hot 30i.csv](data/raw/gadget.ind_Infinix%20Hot%2030i.csv) | 30.1 KiB | Ekspor scraping ulasan gadget.ind_Infinix Hot 30i; 153 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/gadget.ind_Realme Note 50.csv](data/raw/gadget.ind_Realme%20Note%2050.csv) | 11.1 KiB | Ekspor scraping ulasan gadget.ind_Realme Note 50; 60 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/gadget.ind_Samsung A04e.csv](data/raw/gadget.ind_Samsung%20A04e.csv) | 156.4 KiB | Ekspor scraping ulasan gadget.ind_Samsung A04e; 119 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/gadget.ind_TECNO Camon 40.csv](data/raw/gadget.ind_TECNO%20Camon%2040.csv) | 12.7 KiB | Ekspor scraping ulasan gadget.ind_TECNO Camon 40; 15 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/maja stori (counterfeit).csv](data/raw/maja%20stori%20%28counterfeit%29.csv) | 12.0 KiB | Ekspor scraping ulasan maja stori (counterfeit); 10 record CSV, dengan metadata web_scraper serta kolom ulasan/tanggal yang masih perlu dipetakan. | SIMPAN sebagai data sumber; arsipkan sebagai satu paket beserta pemetaan bila eksperimen lama telah ditinggalkan. |
| [data/raw/model data.csv](data/raw/model%20data.csv) | 47.1 KiB | Tabel pemodelan berisi 423 baris dengan shop_id, review_text, timestamp, dan label. | ARSIP BERSYARAT — identik byte dengan pasangan model data.csv/modeling data.csv; pilih satu nama baku setelah memeriksa referensi notebook/skrip, lalu arsipkan salinan. |
| [data/raw/modeling data.csv](data/raw/modeling%20data.csv) | 47.1 KiB | Tabel pemodelan berisi 423 baris dengan shop_id, review_text, timestamp, dan label. | ARSIP BERSYARAT — identik byte dengan pasangan model data.csv/modeling data.csv; pilih satu nama baku setelah memeriksa referensi notebook/skrip, lalu arsipkan salinan. |
| [data/raw/scraping_data_balanced.csv](data/raw/scraping_data_balanced.csv) | 6.2 KiB | Dataset kecil 60 baris dengan shop_id, review_text, timestamp, label; nama file menunjukkan sampel balanced, tetapi keseimbangan/asal label perlu ditinjau sebelum dipakai. | SIMPAN sampai fungsi sampel dan label terkonfirmasi; tempatkan bersama dataset eksperimen, terpisah dari ekspor mentah. |


### 4.5 Referensi kamus

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [references/slang.csv](references/slang.csv) | 60.8 KiB | Kamus normalisasi aktif 4.124 pasangan slang→baku untuk text_proposal, setelah penyaringan konflik dan istilah terlindungi. | SIMPAN lokal; dependensi praproses v2/v3, jangan hapus sebagai data contoh. |
| [references/slang_PROVENANCE.md](references/slang_PROVENANCE.md) | 6.0 KiB | Dokumentasi asal kamus, hash, aturan pembentukan, statistik penyaringan, perintah reproduksi, dan keputusan menyimpan CSV lokal. | SIMPAN; dokumentasi asal dan batas penggunaan sumber kamus. |
| [references/slang_dikecualikan.csv](references/slang_dikecualikan.csv) | 305.6 KiB | Audit 10.882 record kamus yang tidak dipakai, dengan alasan seperti duplikat, konflik baku, atau panjang karakter. | SIMPAN bersama sumber dan slang.csv agar keputusan penyaringan bisa direkonstruksi. |
| [references/sumber/colloquial-indonesian-lexicon.csv](references/sumber/colloquial-indonesian-lexicon.csv) | 3.01 MiB | Ekspor sumber kamus alay lokal, 15.006 record dengan slang, formal, in_dictionary, context, categories; dipakai membangun kamus turunan. | SIMPAN lokal sebagai sumber reproduksi kamus; catatan provenance menetapkan penyimpanan lokal dan lisensi upstream belum dipastikan. |


### 4.6 Kode sumber utama

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [src/__init__.py](src/__init__.py) | 77 B | Penanda paket src dan pintu impor ExperimentConfig dari src.config. | Simpan; mengikat API paket dan impor konfigurasi. |
| [src/config.py](src/config.py) | 11.1 KiB | ExperimentConfig pusat: lokasi data/references/outputs, parameter eksperimen leksikal, sentimen dan HMM, validasi, serta pembuatan direktori. | Simpan; masih dipakai CLI sentimen dan HMM. Rapikan konfigurasi lama secara terpisah: sentiment_model_name dideklarasikan dua kali; jangan pindah file tanpa memperbarui impor dan path berbasis __file__. |
| [src/export_sekuens.py](src/export_sekuens.py) | 29.2 KiB | CLI ekspor probabilitas per produk yang berurutan waktu; membentuk product_id nama_toko_nama_produk, memvalidasi pemetaan 1-ke-1, menyimpan seluruh baris audit, alasan pengecualian, laporan, ringkasan dan checksum. | Simpan; sumber reproduksi sekuens_produk.csv dan outputs/audit. Perubahan nama output atau product_id harus diselaraskan dengan analisis serta daftar split HMM. |
| [src/inference.py](src/inference.py) | 4.5 KiB | Helper inferensi model IndoBERT lokal: menafsirkan urutan label, softmax tiga kelas dan menambah p_negative/p_neutral/p_positive sambil mempertahankan baris teks kosong. | Simpan sementara sebagai API inferensi alternatif/lama; masih diuji tests/test_inference_smoke.py. Konsolidasikan ke sentiment_inference.py hanya setelah migrasi nama kolom Inggris ke Indonesia dan pemanggilnya. |
| [src/preprocessing.py](src/preprocessing.py) | 34.9 KiB | Pipeline cleaning lama per produk/chunk: deteksi kolom, gabung ulasan, bersihkan timestamp, tulis parquet/manifest/checkpoint dan inspeksi hasil; docstring menyatakan sudah digantikan src.cleaning. | Kandidat arsip bersyarat. Pastikan semua notebook telah migrasi dan perbarui tests/cleaning/test_loader.py yang masih menguji pengaman kompatibilitasnya; jangan hapus langsung. |
| [src/run_sentiment_inference.py](src/run_sentiment_inference.py) | 10.5 KiB | CLI RM 1 yang membaca tabel bersih, menjalankan pemeriksaan model dan inferensi pada text_proposal, lalu menulis CSV sentimen serta laporan JSON/Markdown tanpa menimpa berkas. | Simpan sebagai entry point. Default input masih reviews_clean_v2.parquet, sementara run_proposal.py kini menghasilkan v3; samakan lewat argumen eksplisit/konfigurasi sebelum merapikan versi output. |
| [src/sentiment.py](src/sentiment.py) | 5.0 KiB | Kelas IndoBERTSentimentTrainer untuk tokenisasi, fine-tuning tiga kelas dengan Hugging Face Trainer, evaluasi accuracy/F1, dan menyimpan model/tokenizer. | Kandidat arsip eksperimen bersyarat; Lagi/src/debug.ipynb masih mengimpornya. Memakai atribut sentiment_base_model, sentiment_model_dir, dan random_seed yang tidak didefinisikan di ExperimentConfig utama; perlu pemeriksaan sebelum dipakai ulang. |
| [src/sentiment_inference.py](src/sentiment_inference.py) | 27.4 KiB | Mesin inferensi RM 1: memuat model lokal atau revision Hugging Face yang dipin, memeriksa label/tokenizer/device, menghitung p_negatif/p_netral/p_positif dan diagnostik, serta membuat ringkasan. | Simpan sebagai modul inti RM 1; digunakan run_sentiment_inference.py dan tes proposal. Pemindahan memerlukan perubahan impor serta penyesuaian jalur cache/model di konfigurasi. |
| [src/split.py](src/split.py) | 7.2 KiB | ProductSplitter dan SplitResult untuk membagi DataFrame menjadi train/validation/test berdasarkan product_id, mengacak deterministik dan mengecek tidak ada produk lintas split. | Simpan sementara; dipakai notebook smoke_test_split dan Lagi/src/debug.ipynb. Berbeda kontrak dari temporal/splits.py yang memakai daftar train/test; jangan digabung hanya karena nama mirip. |


### 4.7 Kode cleaning

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [src/cleaning/__init__.py](src/cleaning/__init__.py) | 208 B | Penanda paket cleaning dan ekspor load_clean_reviews sebagai API pembacaan tabel ulasan baru. | Simpan; README dan tes memakai impor langsung dari src.cleaning. |
| [src/cleaning/audit.py](src/cleaning/audit.py) | 17.1 KiB | Membuat audit sebelum/sesudah aturan cleaning, statistik timestamp, kandidat tag aspek/frasa templat, ringkasan produk dan pemeriksaan konsistensi hasil. | Simpan; run.py memakainya untuk laporan dan validasi. Bukan berkas hasil audit yang bisa dibuang. |
| [src/cleaning/audit_proposal.py](src/cleaning/audit_proposal.py) | 15.0 KiB | CLI audit yang membaca parquet sebelum/sesudah proposal dan kamus, memverifikasi kolom asli/istilah terlindungi/hash serta menyusun top penggantian slang dan laporan Markdown tanpa mengubah tabel input. | Simpan sebagai alat verifikasi dan jejak reproducibility; nama/default laporan masih menyebut v2 sehingga perbarui argumen saat audit v3, bukan menganggap kode ini duplikat run_proposal.py. |
| [src/cleaning/build.py](src/cleaning/build.py) | 19.4 KiB | Menyusun reviews_clean dari seluruh CSV: ekstraksi fragmen review/aspek/timestamp, ID stabil, text_light, flag kosong/pendek/duplikat, include_for_inference, urutan produk dan jejak manifest. | Simpan sebagai inti cleaning; nama file sumber memengaruhi ID review, sehingga jangan mengganti nama CSV tanpa merencanakan regenerasi dan pencocokan hasil. |
| [src/cleaning/build_slang_dictionary.py](src/cleaning/build_slang_dictionary.py) | 15.5 KiB | Konversi offline ekspor SEACrowd menjadi kamus slang,baku, menyaring konflik/kunci pendek/istilah terlindungi, serta menulis daftar pengecualian dan profil kamus. | Simpan untuk reproduksi references/slang.csv; bergantung pada ekspor sumber lokal, tanpa jaringan. Kamus dan daftar pengecualian sebaiknya diarsipkan bersama sumber dan laporan profil. |
| [src/cleaning/config.py](src/cleaning/config.py) | 5.6 KiB | CleaningConfig untuk sumber CSV/pemetaan, output, aturan fragmen, timestamp, duplikasi, ambang profiling, serta PROTECTED_TERMS normalisasi proposal/slang. | Simpan; dipakai hampir seluruh modul cleaning. Default raw_dir menunjuk folder eksternal /Users/asyzyni/Desktop/TA/Data Beneran, sehingga pemindahan folder data membutuhkan pembaruan argumen/config. |
| [src/cleaning/export.py](src/cleaning/export.py) | 2.3 KiB | Mengekspor sampel tabel Nama Toko/Nama Produk/Review/Timestamp dengan pemilihan acak deterministik dan opsi anonimisasi khusus contoh. | Simpan; run.py memakainya untuk tabel_contoh.csv dan tabel_contoh_anonim.csv. |
| [src/cleaning/file_map.py](src/cleaning/file_map.py) | 6.3 KiB | Membaca dan memvalidasi pemetaan file→toko/produk, membaca CSV berbagai encoding, mendeteksi keluarga skema dan kelompok file yang hash isinya sama. | Simpan; fondasi build.py dan run.py. Pemetaan mentah serta nama file sumber harus tetap konsisten saat data dirapikan. |
| [src/cleaning/loader.py](src/cleaning/loader.py) | 2.4 KiB | Pembaca parquet cleaning yang opsional memfilter include_for_inference, mempertahankan kolom wajib dan mengurutkan per product_id/seq_in_product tanpa mengolah data mentah. | Simpan sebagai API akses hasil cleaning; dipakai paket src.cleaning dan tes kompatibilitas. |
| [src/cleaning/proposal.py](src/cleaning/proposal.py) | 13.4 KiB | Transformasi text_proposal sesuai Bab 3.3 dengan Unicode, URL/simbol/tanda baca/angka/typo/slang dan istilah terlindungi; menambah dua kolom tanpa mengubah data asli serta menghasilkan audit. | Simpan; digunakan CLI proposal dan audit_proposal. Kamus slang.csv merupakan input penting dan tidak boleh dihapus hanya karena merupakan hasil konversi. |
| [src/cleaning/run.py](src/cleaning/run.py) | 8.7 KiB | CLI cleaning utama: validasi input/pemetaan, build_all, tulis reviews_clean.parquet, laporan CSV/checksum serta ringkasan pemeriksaan; berhenti jika folder output sudah berisi file. | Simpan sebagai entry point cleaning; gunakan output baru untuk menjalankan ulang setelah perapian, bukan menghapus hasil lama hanya agar CLI dapat berjalan. |
| [src/cleaning/run_proposal.py](src/cleaning/run_proposal.py) | 3.3 KiB | CLI yang mewajibkan kamus slang,baku, membaca reviews_clean.parquet, menambah text_proposal/text_proposal_kosong dan memverifikasi round-trip parquet sebelum menulis reviews_clean_v3.parquet/laporan. | Simpan; entry point proposal terkini. Sesuaikan konsumen yang masih default v2 sebelum mengarsipkan salah satu versi parquet. |
| [src/cleaning/schema.py](src/cleaning/schema.py) | 14.2 KiB | Profil isi/header kolom CSV dan klasifikasi peran review, timestamp, rating, aspek, metadata, media atau balasan penjual; menyimpan bukti/keyakinan pemetaan kolom. | Simpan; penentu data yang masuk cleaning dan laporan column_map.csv. Mengubah heuristik akan mengubah hasil tabel bersih. |
| [src/cleaning/text_clean.py](src/cleaning/text_clean.py) | 4.4 KiB | Fungsi pembersihan ringan: menyatukan fragmen, membuang HTML/URL/kontrol, NFKC, mereduksi huruf berulang dan spasi; menghitung jumlah perubahan tiap langkah. | Simpan; digunakan build.py, audit.py dan proposal.py. Berbeda tujuan dari praproses leksikal dan proposal. |


### 4.8 Kode fitur leksikal

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [src/features/__init__.py](src/features/__init__.py) | 250 B | Pintu impor kelas praproses leksikal dan pembentuk kamus semantic orientation dalam paket features. | Simpan bersama kedua modul features; bila fitur leksikal diarsipkan, pindahkan satu paket beserta pemanggilnya. |
| [src/features/lexical_preprocessing.py](src/features/lexical_preprocessing.py) | 7.4 KiB | Membentuk text_lex: normalisasi kamus publik/domain, penggabungan frasa seed, penandaan negasi, stopword dan ambang frekuensi dengan perlindungan istilah seed. | Simpan sebagai komponen eksperimen leksikal atau arsipkan satu paket setelah konfirmasi cakupan penelitian; bukan pengganti text_light/text_proposal. Belum terlihat sebagai bagian entry point RM 1/RM 2 saat ini. |
| [src/features/lexicon.py](src/features/lexicon.py) | 7.4 KiB | SemanticOrientationLexicon menghitung PMI kata terhadap seed genuine/counterfeit, bootstrap dokumen dan interval kepercayaan, lalu menghasilkan daftar kata signifikan tiap kelas. | Simpan jika eksperimen leksikal masih dibutuhkan; kandidat arsip terpisah bila penelitian hanya memakai RM 1/RM 2. Membutuhkan ExperimentConfig dan seed, sehingga jangan hapus referensi pasangannya tanpa memeriksa notebook. |


### 4.9 Kode HMM/temporal

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [src/temporal/__init__.py](src/temporal/__init__.py) | 767 B | API paket RM 2 yang mengekspor konfigurasi HMM, pembentukan sekuens/split, fitting, fitur dan pipeline; tidak menerima label genuine/counterfeit. | Simpan; dipakai tes dan pemanggil paket src.temporal. |
| [src/temporal/config.py](src/temporal/config.py) | 5.5 KiB | HMMConfig mengatur nama kolom, urutan [positif, netral, negatif], representasi prob/logit/ALR/CLR, state/restart/backend, minimum ulasan dan lokasi output; dapat dibuat dari ExperimentConfig. | Simpan; dipakai seluruh modul HMM. Path output model/fitur/meta perlu diperbarui bersama bila outputs dipindahkan. |
| [src/temporal/features.py](src/temporal/features.py) | 2.3 KiB | Ekstraksi fitur per produk dari HMM beku: log_lik/T, prop_state1, pos_transisi dan kolom opsional idx_transisi/occ_state1. | Simpan; hasilnya merupakan masukan fitur temporal lanjutan, dengan state1 berarti sentimen negatif tinggi dan bukan label counterfeit. |
| [src/temporal/fitting.py](src/temporal/fitting.py) | 10.1 KiB | Melatih HMM unsupervised pada Train saja, memakai inisialisasi proposal dan restart acak dengan backend hmmlearn/NumPy, memilih likelihood terbaik dan mengurutkan state berdasarkan massa p_negatif. | Simpan; inti fitting pipeline RM 2. Memindahkan/menghapusnya membuat model tidak dapat direproduksi. |
| [src/temporal/gaussian_hmm.py](src/temporal/gaussian_hmm.py) | 10.3 KiB | Implementasi Gaussian diagonal HMM berbasis NumPy/SciPy: forward/backward di log-space, posterior/likelihood, pengurutan state dan EM Baum–Welch multi-sekuens. | Simpan; dipakai untuk inferensi fitur pada semua backend dan sebagai backend fitting cadangan. Model pickle keluaran menyimpan kelas modul ini, sehingga memindahkannya berisiko memutus pemuatan model lama. |
| [src/temporal/pipeline.py](src/temporal/pipeline.py) | 10.9 KiB | Orkestrasi RM 2: cegah kebocoran Train/Test/sintetis, fit Train, bekukan model, transformasi fitur, uji stabilitas opsional, simpan model pickle/meta/fitur dan muat model untuk data baru. | Simpan sebagai inti RM 2; keluaran model, metadata dan fitur merupakan satu set yang sebaiknya diarsipkan bersama. |
| [src/temporal/run_hmm.py](src/temporal/run_hmm.py) | 4.1 KiB | CLI RM 2 dari tabel RM 1 dan daftar split, mengatur backend/representasi/parameter, menjalankan pipeline dan menulis hmm_model.pkl, hmm_meta.json serta hmm_features.parquet. | Simpan sebagai entry point HMM. Memerlukan --split-ids atau --temporary-split eksplisit; sesuaikan path ketika hasil eksperimen ditata ulang. |
| [src/temporal/sequences.py](src/temporal/sequences.py) | 8.1 KiB | Menyaring dan mengurutkan ulasan valid per produk, membentuk SequenceSet/X/lengths tanpa menyambung batas produk dan mengubah probabilitas/logit ke representasi observasi HMM. | Simpan; input inti fitting/pipeline. Urutan kolom probabilitas dan aturan tie-break harus konsisten dengan RM 1 serta ekspor sekuens. |
| [src/temporal/splits.py](src/temporal/splits.py) | 2.3 KiB | Membaca daftar product_id/split train/test dari CSV/parquet dan menyediakan split acak sementara yang ditandai bukan untuk hasil pelaporan. | Simpan; dipakai run_hmm.py. Berkas split final perlu dipertahankan agar eksperimen dapat direproduksi. |


### 4.10 Notebook smoke test di dalam src

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [src/smoke_test/smoke_test_split.ipynb](src/smoke_test/smoke_test_split.ipynb) | 14.9 KiB | Notebook 16 sel yang memeriksa impor/config, membuat 60 ulasan sintetis dari 20 produk, dan menginisialisasi ProductSplitter; belum memanggil metode pembagian dataset. | RAPIKAN — pindahkan ke notebooks/debug/ bila masih digunakan; perbaiki PROJECT_ROOT absolut yang memuat spasi setelah TA. Arsipkan jika sudah digantikan tes otomatis. |


### 4.11 File pengujian

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [tests/cleaning/__init__.py](tests/cleaning/__init__.py) | 0 B | Penanda paket pengujian cleaning; sengaja kosong agar impor relatif antartes berfungsi. | SIMPAN — ukuran 0 byte di sini normal; berbeda dari notebook kosong. |
| [tests/cleaning/conftest.py](tests/cleaning/conftest.py) | 6.8 KiB | Fixture pytest: membuat CSV sintetis tiga skema, pemetaan file, dan konfigurasi cleaning sementara. | SIMPAN — dipakai tes cleaning; bukan data mentah produksi. |
| [tests/cleaning/test_audit_proposal.py](tests/cleaning/test_audit_proposal.py) | 8.3 KiB | Menguji jejak penggantian slang, perlindungan istilah, frekuensi penggantian, hash input, dan penolakan overwrite pada audit proposal. | SIMPAN — memeriksa konsistensi audit preprocessing. |
| [tests/cleaning/test_build_slang_dictionary.py](tests/cleaning/test_build_slang_dictionary.py) | 9.4 KiB | Menguji konversi kamus slang: ambiguitas, duplikasi, istilah terlindungi, determinisme, hash, dan penolakan penimpaan. | SIMPAN — menjaga kontrak kamus references/. |
| [tests/cleaning/test_cleaning.py](tests/cleaning/test_cleaning.py) | 11.6 KiB | Menguji cleaning tiga skema CSV, deduplikasi, timestamp, pemetaan, ekspor contoh anonim, serta determinisme. | SIMPAN — tes utama tahap cleaning. |
| [tests/cleaning/test_loader.py](tests/cleaning/test_loader.py) | 6.5 KiB | Menguji pembacaan Parquet, filter dan urutan inferensi, pemilihan kolom, serta pengaman pipeline preprocessing lama. | SIMPAN — masih mengimpor DataPreprocessor dari src/preprocessing.py. |
| [tests/cleaning/test_proposal.py](tests/cleaning/test_proposal.py) | 10.7 KiB | Menguji preprocessing proposal v3, normalisasi Unicode, kamus slang, istilah terlindungi, kolom asli, dan CLI. | SIMPAN — tes untuk perilaku proposal yang sedang dikembangkan. |
| [tests/label_readiness/synthetic_label_readiness.py](tests/label_readiness/synthetic_label_readiness.py) | 1.8 KiB | Harness audit kesiapan Jalur B: membuat 320 ulasan sintetis dan mengecek impor lexicon serta tfidf; belum mengikat enam pemeriksaan perilaku ke API nyata. | TINJAU — src/features/tfidf.py belum ada, sehingga harness belum menjadi tes jalur lengkap; arsipkan hanya jika rencana Jalur B ditinggalkan. |
| [tests/temporal/__init__.py](tests/temporal/__init__.py) | 0 B | Penanda paket tes HMM; sengaja kosong untuk impor relatif helpers. | SIMPAN — bukan file sampah meskipun 0 byte. |
| [tests/temporal/helpers.py](tests/temporal/helpers.py) | 2.0 KiB | Generator probabilitas sentimen sintetis untuk produk baik, memburuk, dan buruk, tanpa model IndoBERT. | SIMPAN — digunakan pengujian HMM. |
| [tests/temporal/test_gaussian_hmm.py](tests/temporal/test_gaussian_hmm.py) | 3.7 KiB | Menguji forward/posterior terhadap enumerasi jalur, pemrosesan batch, pemisahan sekuens, dan validitas pembelajaran Baum–Welch. | SIMPAN — verifikasi matematis implementasi HMM. |
| [tests/temporal/test_pipeline.py](tests/temporal/test_pipeline.py) | 11.5 KiB | Menguji urutan sekuens, pemisahan train/test, pencegahan kebocoran, fitur HMM, penyimpanan model, dan kecocokan dengan hmmlearn bila tersedia. | SIMPAN — tes integrasi pipeline temporal. |
| [tests/test_export_sekuens.py](tests/test_export_sekuens.py) | 13.2 KiB | Menguji filter ulasan, batas tiga ulasan per produk, urutan deterministik, ID produk, rekonstruksi tabel ekspor dari audit, checksum, dan CLI. | SIMPAN — mendampingi src/export_sekuens.py. |
| [tests/test_inference_smoke.py](tests/test_inference_smoke.py) | 1.6 KiB | Menguji pemetaan label dan inferensi sepuluh kalimat dengan model lokal; bagian model dilewati bila INDOBERT_SENTIMENT_MODEL_DIR tidak tersedia. | SIMPAN — masih mengimpor src/inference.py; perlu model lokal untuk uji penuh. |
| [tests/test_sentiment_inference.py](tests/test_sentiment_inference.py) | 23.3 KiB | Menguji revision model, pemetaan label, tokenizer/device, batching, probabilitas, ringkasan, dan CLI dengan model palsu; mencakup smoke test model nyata opsional. | SIMPAN — tes utama inferensi, bukan bukti bahwa semua uji model nyata sudah dijalankan. |


### 4.12 Keluaran cleaning dan laporannya

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [outputs/cleaned/reports/aspect_tag_candidates.csv](outputs/cleaned/reports/aspect_tag_candidates.csv) | 139.5 KiB | 1.319 kandidat fragmen label aspek beserta frekuensi, asal kolom, dan isian persetujuan untuk peninjauan cleaning. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/cleaning_audit.csv](outputs/cleaned/reports/cleaning_audit.csv) | 2.8 KiB | 23 catatan perubahan jumlah file, baris, fragmen, teks, duplikasi, dan timestamp pada cleaning dasar. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/column_map.csv](outputs/cleaned/reports/column_map.csv) | 973.8 KiB | Pemetaan peran 2.035 kolom sumber: metadata, ulasan, tanggal, atau balasan penjual; disertai statistik dan bukti. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/duplicate_files.csv](outputs/cleaned/reports/duplicate_files.csv) | 5.8 KiB | 42 entri file yang masuk grup duplikat berdasarkan hash pada dataset sumber cleaning; bukan daftar otomatis untuk dihapus. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/excluded_files.csv](outputs/cleaned/reports/excluded_files.csv) | 235 B | Dua entri yang dikeluarkan dari cleaning: CHIC PHONE.csv dan file pemetaan yang merujuk dirinya sendiri. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/file_manifest.csv](outputs/cleaned/reports/file_manifest.csv) | 129.4 KiB | Manifest 405 entri pemetaan file sumber, skema, hash, jumlah baris, kolom review, status, dan statistik kualitas; 403 diproses. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/fragment_counts.csv](outputs/cleaned/reports/fragment_counts.csv) | 344 B | 13 ringkasan jumlah sel/fragmen menurut peran kolom dan jenis nilai. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/output_hashes.csv](outputs/cleaned/reports/output_hashes.csv) | 1.3 KiB | 14 hash keluaran cleaning dasar untuk memeriksa integritas; nama reviews_clean.parquet di sini cocok byte dengan reviews_clean (2).parquet yang tersedia. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/preset_phrase_candidates.csv](outputs/cleaned/reports/preset_phrase_candidates.csv) | 13.9 KiB | 139 kandidat frasa ulasan bawaan/berulang beserta frekuensinya, bahan tinjauan aturan cleaning. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/product_summary.csv](outputs/cleaned/reports/product_summary.csv) | 55.9 KiB | Ringkasan 401 produk: jumlah ulasan, kelayakan inferensi, duplikasi, timestamp hilang, dan ambang minimal ulasan. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/tabel_contoh.csv](outputs/cleaned/reports/tabel_contoh.csv) | 333 B | Tiga contoh hasil cleaning berformat Nama Toko, Nama Produk, Review, Timestamp; masih memuat nama sebenarnya. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/tabel_contoh_anonim.csv](outputs/cleaned/reports/tabel_contoh_anonim.csv) | 262 B | Tiga contoh hasil cleaning dengan nama toko/produk dianonimkan untuk penulisan laporan. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/timestamp_formats.csv](outputs/cleaned/reports/timestamp_formats.csv) | 742 B | 12 profil format tanggal per keluarga skema, status parsing, jumlah baris, dan contoh. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reports/validation_checks.csv](outputs/cleaned/reports/validation_checks.csv) | 975 B | 11 hasil validasi cleaning dasar, termasuk rekonsiliasi jumlah file/baris, ID unik, dan urutan sekuens produk. | SIMPAN sebagai bukti cleaning dasar; jika diarsipkan, pindahkan seluruh paket laporan bersama parquet sumbernya. |
| [outputs/cleaned/reviews_clean (2).parquet](outputs/cleaned/reviews_clean%20%282%29.parquet) | 7.09 MiB | Tabel cleaning dasar: 64.699 ulasan, 27 kolom. Hash sesuai entri reviews_clean.parquet pada output_hashes.csv; masukan eksplisit pembentukan v2 dan v3. | SIMPAN; (2) bukan bukti duplikasi. Nama dapat dibakukan hanya bersama semua path dan catatan asalnya. |
| [outputs/cleaned/reviews_clean_v2.parquet](outputs/cleaned/reviews_clean_v2.parquet) | 9.70 MiB | Versi praproses proposal v2, 64.699 baris/29 kolom; menambahkan text_proposal dan penanda teks kosong menggunakan kamus slang. Masukan inferensi v2. | SIMPAN bersama sentimen v2 dan audit sekuens yang masih memakai v2; arsipkan satu paket setelah alur pengganti diverifikasi. |
| [outputs/cleaned/reviews_clean_v3.parquet](outputs/cleaned/reviews_clean_v3.parquet) | 9.70 MiB | Versi praproses proposal v3, 64.699 baris/29 kolom; memperbaiki NFKC, zero-width, dan pemisah Unicode. Masukan inferensi v3 14 Oktober 2026. | SIMPAN; keluaran praproses terbaru yang tercatat, dengan laporan preprocessing_proposal_v3.md. |


### 4.13 Keluaran inferensi sentimen dan lognya

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [outputs/sentiment/logs/run_penuh_lokal.log](outputs/sentiment/logs/run_penuh_lokal.log) | 3.0 KiB | Log konsol dan durasi inferensi penuh lokal v2; mencatat keluaran sentimen_ulasan_v2.csv serta distribusi hasil. | ARSIP BERSYARAT bersama run v2; ringkasan sudah memuat statistik utama, tetapi log berguna untuk menelusuri eksekusi. |
| [outputs/sentiment/logs/run_penuh_lokal.start](outputs/sentiment/logs/run_penuh_lokal.start) | 20 B | Penanda waktu mulai run penuh lokal: 13 Oktober 2026 pukul 14:56:06. | ARSIP BERSYARAT; informasi juga ada di ringkasan v2, dapat dilepas setelah kebutuhan pencatatan waktu selesai. |
| [outputs/sentiment/logs/uji_200_lokal.log](outputs/sentiment/logs/uji_200_lokal.log) | 1.9 KiB | Log konsol dan durasi uji lokal 200 ulasan v2, keluaran uji_200_lokal.csv. | ARSIP BERSYARAT bersama hasil uji_200_lokal; bukan masukan pipeline. |
| [outputs/sentiment/sekuens_produk.csv](outputs/sentiment/sekuens_produk.csv) | 12.60 MiB | Sekuens probabilitas sentimen yang diurutkan per produk: 63.348 baris, 400 produk, minimal 3 ulasan; laporan menegaskan sumber sentimen_ulasan_v2.csv. | SIMPAN bersama paket v2; belum ada bukti file ini sudah diperbarui dari v3. |
| [outputs/sentiment/sentimen_ulasan_v2.csv](outputs/sentiment/sentimen_ulasan_v2.csv) | 44.75 MiB | Hasil inferensi penuh v2 dari reviews_clean_v2.parquet: 64.699 baris, 64.641 teks diinferensi, max_length=512; memuat probabilitas, logits, token, dan label bantu. | SIMPAN; v2 masih menjadi sumber sekuens dan audit yang tersedia. Arsipkan hanya bersama seluruh hasil turunannya. |
| [outputs/sentiment/sentimen_ulasan_v2_ringkasan.json](outputs/sentiment/sentimen_ulasan_v2_ringkasan.json) | 4.0 KiB | Metadata dan statistik mesin untuk inferensi penuh v2 (13 Oktober 2026), termasuk model/revisi, parameter max_length=512, distribusi, dan hasil pemeriksaan. | SIMPAN; v2 masih menjadi sumber sekuens dan audit yang tersedia. Arsipkan hanya bersama seluruh hasil turunannya. |
| [outputs/sentiment/sentimen_ulasan_v2_ringkasan.md](outputs/sentiment/sentimen_ulasan_v2_ringkasan.md) | 2.7 KiB | Laporan terbaca manusia untuk inferensi penuh v2 (13 Oktober 2026), parameter, distribusi keluaran, sanity check, dan keterbatasan evaluasi. | SIMPAN; v2 masih menjadi sumber sekuens dan audit yang tersedia. Arsipkan hanya bersama seluruh hasil turunannya. |
| [outputs/sentiment/sentimen_ulasan_v3.csv](outputs/sentiment/sentimen_ulasan_v3.csv) | 44.74 MiB | Hasil inferensi penuh v3 dari reviews_clean_v3.parquet: 64.699 baris, 64.641 teks diinferensi, max_length=128; memuat probabilitas, logits, token, dan label bantu. | SIMPAN; run inferensi penuh terbaru yang tercatat. Belum terbukti menjadi sumber sekuens_produk.csv saat ini. |
| [outputs/sentiment/sentimen_ulasan_v3_ringkasan.json](outputs/sentiment/sentimen_ulasan_v3_ringkasan.json) | 4.0 KiB | Metadata dan statistik mesin untuk inferensi penuh v3 (14 Oktober 2026), termasuk model/revisi, parameter max_length=128, distribusi, dan hasil pemeriksaan. | SIMPAN; run inferensi penuh terbaru yang tercatat. Belum terbukti menjadi sumber sekuens_produk.csv saat ini. |
| [outputs/sentiment/sentimen_ulasan_v3_ringkasan.md](outputs/sentiment/sentimen_ulasan_v3_ringkasan.md) | 2.7 KiB | Laporan terbaca manusia untuk inferensi penuh v3 (14 Oktober 2026), parameter, distribusi keluaran, sanity check, dan keterbatasan evaluasi. | SIMPAN; run inferensi penuh terbaru yang tercatat. Belum terbukti menjadi sumber sekuens_produk.csv saat ini. |
| [outputs/sentiment/uji200_v3.csv](outputs/sentiment/uji200_v3.csv) | 116.6 KiB | Hasil uji 200 ulasan pada 14 Oktober 2026 dari parquet v3, max_length=128. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji200_v3_ringkasan.json](outputs/sentiment/uji200_v3_ringkasan.json) | 3.9 KiB | Metadata, hitungan, distribusi, dan pemeriksaan terstruktur untuk uji 200 ulasan pada 14 Oktober 2026 dari parquet v3, max_length=128. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji200_v3_ringkasan.md](outputs/sentiment/uji200_v3_ringkasan.md) | 2.7 KiB | Laporan Markdown untuk uji 200 ulasan pada 14 Oktober 2026 dari parquet v3, max_length=128. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji_200.csv](outputs/sentiment/uji_200.csv) | 89.3 KiB | Hasil uji awal 200 ulasan pada 10 Oktober 2026, max_length=128, sebelum kolom text_proposal; path input lama reviews_clean.parquet. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji_200_lokal.csv](outputs/sentiment/uji_200_lokal.csv) | 103.6 KiB | Hasil uji lokal 200 ulasan pada 13 Oktober 2026 dari parquet v2, max_length=512. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji_200_lokal_ringkasan.json](outputs/sentiment/uji_200_lokal_ringkasan.json) | 3.9 KiB | Metadata, hitungan, distribusi, dan pemeriksaan terstruktur untuk uji lokal 200 ulasan pada 13 Oktober 2026 dari parquet v2, max_length=512. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji_200_lokal_ringkasan.md](outputs/sentiment/uji_200_lokal_ringkasan.md) | 2.7 KiB | Laporan Markdown untuk uji lokal 200 ulasan pada 13 Oktober 2026 dari parquet v2, max_length=512. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji_200_ringkasan.json](outputs/sentiment/uji_200_ringkasan.json) | 2.2 KiB | Metadata, hitungan, distribusi, dan pemeriksaan terstruktur untuk uji awal 200 ulasan pada 10 Oktober 2026, max_length=128, sebelum kolom text_proposal; path input lama reviews_clean.parquet. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |
| [outputs/sentiment/uji_200_ringkasan.md](outputs/sentiment/uji_200_ringkasan.md) | 1.9 KiB | Laporan Markdown untuk uji awal 200 ulasan pada 10 Oktober 2026, max_length=128, sebelum kolom text_proposal; path input lama reviews_clean.parquet. | ARSIP BERSYARAT — hasil smoke test; pindahkan CSV dan kedua ringkasannya bersama setelah run penuh terkait tervalidasi. Simpan bila dibutuhkan untuk pembandingan. |


### 4.14 Audit ekspor sekuens

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [outputs/audit/CHECKSUMS.txt](outputs/audit/CHECKSUMS.txt) | 454 B | Empat checksum SHA-256 untuk input sentimen v2, sekuens, audit lengkap, dan daftar ulasan yang dikeluarkan. | SIMPAN bersama paket audit; saat memindahkan file pertahankan informasi path asal. |
| [outputs/audit/ekspor_sekuens_dikeluarkan.csv](outputs/audit/ekspor_sekuens_dikeluarkan.csv) | 166.4 KiB | Daftar 1.351 ID ulasan yang dikeluarkan dari sekuens v2 beserta alasan dan flag kelayakan. | SIMPAN bersama audit ekspor v2 agar pengurangan data dapat ditelusuri. |
| [outputs/audit/ekspor_sekuens_laporan.md](outputs/audit/ekspor_sekuens_laporan.md) | 3.9 KiB | Laporan ekspor sekuens v2 tanggal 13 Oktober 2026: perintah, jumlah baris/produk, alasan pengecualian, validasi, dan hash. | SIMPAN; bukti utama asal sekuens_produk.csv dan parameter run. |
| [outputs/audit/ekspor_sekuens_ringkasan.json](outputs/audit/ekspor_sekuens_ringkasan.json) | 4.0 KiB | Versi terstruktur ringkasan ekspor sekuens v2: parameter, hitungan, invarian, hash, dan hasil reproduksibilitas. | SIMPAN bersama laporan Markdown; berguna untuk pemeriksaan otomatis, bukan salinan format yang sia-sia. |
| [outputs/audit/sekuens_audit.csv](outputs/audit/sekuens_audit.csv) | 45.37 MiB | Audit 64.699 baris input v2 dengan status masuk sekuens, alasan dikeluarkan, dan nomor urut; versi tersimpan belum punya product_id_lama yang ada di kode exporter saat ini. | SIMPAN sebagai bukti run v2; arsipkan bersama sekuens, laporan, dan checksum. Jangan menganggap identik dengan keluaran versi kode terbaru. |


### 4.15 Keluaran lain dan placeholder

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [outputs/.gitkeep](outputs/.gitkeep) | 0 B | File kosong penjaga folder outputs dalam Git. | SIMPAN; bukan data eksperimen dan tidak perlu dibersihkan. |
| [outputs/logs/cleaning_manifest.csv](outputs/logs/cleaning_manifest.csv) | 71.2 KiB | Log pipeline cleaning lain/lama dengan 593 entri file, status, encoding, jumlah baris masuk/keluar, bagian, dan galat; skemanya berbeda dari manifest cleaning dasar. | ARSIP BERSYARAT — setelah dipastikan eksperimen terkait tidak lagi aktif; jangan digabung dengan outputs/cleaned/reports/file_manifest.csv seolah berasal dari run yang sama. |

## 5. File cache dan metadata sistem

Semua file di bawah merupakan kandidat pembersihan. Menghapus cache tidak sama dengan menghapus file `.py` sumbernya. Cache dapat muncul lagi sesudah Python, pytest, atau Finder digunakan. Akhiran `cpython-310`, `cpython-311`, dan `cpython-313` menunjukkan cache dari versi Python berbeda, bukan tiga versi kode sumber yang harus disimpan.


### 5.1 Metadata Finder

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [.DS_Store](.DS_Store) | 10.0 KiB | Metadata tampilan folder dari Finder macOS; bukan bagian kode atau dataset. | BERSIHKAN — dapat dibuat lagi oleh Finder. |
| [outputs/.DS_Store](outputs/.DS_Store) | 8.0 KiB | Metadata tampilan folder dari Finder macOS; bukan bagian kode atau dataset. | BERSIHKAN — dapat dibuat lagi oleh Finder. |
| [outputs/cleaned/.DS_Store](outputs/cleaned/.DS_Store) | 6.0 KiB | Metadata tampilan folder dari Finder macOS; bukan bagian kode atau dataset. | BERSIHKAN — dapat dibuat lagi oleh Finder. |
| [src/.DS_Store](src/.DS_Store) | 6.0 KiB | Metadata tampilan folder dari Finder macOS; bukan bagian kode atau dataset. | BERSIHKAN — dapat dibuat lagi oleh Finder. |


### 5.2 Cache pytest

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [.pytest_cache/.gitignore](.pytest_cache/.gitignore) | 37 B | Aturan otomatis pytest agar isi direktori cache tidak ikut Git. | BERSIHKAN — boleh menghapus seluruh .pytest_cache/; riwayat cache tes akan hilang. |
| [.pytest_cache/CACHEDIR.TAG](.pytest_cache/CACHEDIR.TAG) | 191 B | Penanda direktori cache untuk alat backup dan pengelola berkas. | BERSIHKAN — boleh menghapus seluruh .pytest_cache/; riwayat cache tes akan hilang. |
| [.pytest_cache/README.md](.pytest_cache/README.md) | 302 B | Penjelasan otomatis tentang cache pytest, bukan dokumentasi proyek. | BERSIHKAN — boleh menghapus seluruh .pytest_cache/; riwayat cache tes akan hilang. |
| [.pytest_cache/v/cache/lastfailed](.pytest_cache/v/cache/lastfailed) | 116 B | Daftar tes yang gagal pada eksekusi pytest terdahulu; tidak membuktikan status tes saat ini. | BERSIHKAN — boleh menghapus seluruh .pytest_cache/; riwayat cache tes akan hilang. |
| [.pytest_cache/v/cache/nodeids](.pytest_cache/v/cache/nodeids) | 21.1 KiB | Daftar identitas tes yang pernah dikumpulkan pytest. | BERSIHKAN — boleh menghapus seluruh .pytest_cache/; riwayat cache tes akan hilang. |


### 5.3 Cache kode sumber

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [src/__pycache__/__init__.cpython-310.pyc](src/__pycache__/__init__.cpython-310.pyc) | 230 B | Bytecode/cache Python untuk `src/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/__init__.cpython-311.pyc](src/__pycache__/__init__.cpython-311.pyc) | 270 B | Bytecode/cache Python untuk `src/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/config.cpython-310.pyc](src/__pycache__/config.cpython-310.pyc) | 7.4 KiB | Bytecode/cache Python untuk `src/config.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/config.cpython-311.pyc](src/__pycache__/config.cpython-311.pyc) | 11.6 KiB | Bytecode/cache Python untuk `src/config.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/export_sekuens.cpython-311.pyc](src/__pycache__/export_sekuens.cpython-311.pyc) | 53.8 KiB | Bytecode/cache Python untuk `src/export_sekuens.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/inference.cpython-311.pyc](src/__pycache__/inference.cpython-311.pyc) | 8.2 KiB | Bytecode/cache Python untuk `src/inference.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/inference.cpython-313.pyc](src/__pycache__/inference.cpython-313.pyc) | 6.9 KiB | Bytecode/cache Python untuk `src/inference.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/preprocessing.cpython-311.pyc](src/__pycache__/preprocessing.cpython-311.pyc) | 27.3 KiB | Bytecode/cache Python untuk `src/preprocessing.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/preprocessing.cpython-313.pyc](src/__pycache__/preprocessing.cpython-313.pyc) | 23.6 KiB | Bytecode/cache Python untuk `src/preprocessing.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/run_sentiment_inference.cpython-311.pyc](src/__pycache__/run_sentiment_inference.cpython-311.pyc) | 18.2 KiB | Bytecode/cache Python untuk `src/run_sentiment_inference.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/sentiment.cpython-311.pyc](src/__pycache__/sentiment.cpython-311.pyc) | 6.7 KiB | Bytecode/cache Python untuk `src/sentiment.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/sentiment_inference.cpython-311.pyc](src/__pycache__/sentiment_inference.cpython-311.pyc) | 42.0 KiB | Bytecode/cache Python untuk `src/sentiment_inference.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/__pycache__/split.cpython-311.pyc](src/__pycache__/split.cpython-311.pyc) | 10.4 KiB | Bytecode/cache Python untuk `src/split.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/__init__.cpython-311.pyc](src/cleaning/__pycache__/__init__.cpython-311.pyc) | 430 B | Bytecode/cache Python untuk `src/cleaning/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/audit.cpython-311.pyc](src/cleaning/__pycache__/audit.cpython-311.pyc) | 25.9 KiB | Bytecode/cache Python untuk `src/cleaning/audit.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/audit_proposal.cpython-311.pyc](src/cleaning/__pycache__/audit_proposal.cpython-311.pyc) | 26.2 KiB | Bytecode/cache Python untuk `src/cleaning/audit_proposal.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/build.cpython-311.pyc](src/cleaning/__pycache__/build.cpython-311.pyc) | 25.0 KiB | Bytecode/cache Python untuk `src/cleaning/build.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/build_slang_dictionary.cpython-311.pyc](src/cleaning/__pycache__/build_slang_dictionary.cpython-311.pyc) | 30.5 KiB | Bytecode/cache Python untuk `src/cleaning/build_slang_dictionary.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/config.cpython-311.pyc](src/cleaning/__pycache__/config.cpython-311.pyc) | 5.0 KiB | Bytecode/cache Python untuk `src/cleaning/config.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/export.cpython-311.pyc](src/cleaning/__pycache__/export.cpython-311.pyc) | 4.3 KiB | Bytecode/cache Python untuk `src/cleaning/export.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/file_map.cpython-311.pyc](src/cleaning/__pycache__/file_map.cpython-311.pyc) | 9.4 KiB | Bytecode/cache Python untuk `src/cleaning/file_map.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/loader.cpython-311.pyc](src/cleaning/__pycache__/loader.cpython-311.pyc) | 3.6 KiB | Bytecode/cache Python untuk `src/cleaning/loader.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/proposal.cpython-311.pyc](src/cleaning/__pycache__/proposal.cpython-311.pyc) | 23.4 KiB | Bytecode/cache Python untuk `src/cleaning/proposal.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/run.cpython-311.pyc](src/cleaning/__pycache__/run.cpython-311.pyc) | 14.4 KiB | Bytecode/cache Python untuk `src/cleaning/run.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/run_proposal.cpython-311.pyc](src/cleaning/__pycache__/run_proposal.cpython-311.pyc) | 6.1 KiB | Bytecode/cache Python untuk `src/cleaning/run_proposal.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/schema.cpython-311.pyc](src/cleaning/__pycache__/schema.cpython-311.pyc) | 20.0 KiB | Bytecode/cache Python untuk `src/cleaning/schema.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/cleaning/__pycache__/text_clean.cpython-311.pyc](src/cleaning/__pycache__/text_clean.cpython-311.pyc) | 6.2 KiB | Bytecode/cache Python untuk `src/cleaning/text_clean.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/features/__pycache__/__init__.cpython-311.pyc](src/features/__pycache__/__init__.cpython-311.pyc) | 457 B | Bytecode/cache Python untuk `src/features/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/features/__pycache__/lexical_preprocessing.cpython-311.pyc](src/features/__pycache__/lexical_preprocessing.cpython-311.pyc) | 13.2 KiB | Bytecode/cache Python untuk `src/features/lexical_preprocessing.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/features/__pycache__/lexicon.cpython-311.pyc](src/features/__pycache__/lexicon.cpython-311.pyc) | 9.3 KiB | Bytecode/cache Python untuk `src/features/lexicon.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/__init__.cpython-310.pyc](src/temporal/__pycache__/__init__.cpython-310.pyc) | 897 B | Bytecode/cache Python untuk `src/temporal/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/__init__.cpython-311.pyc](src/temporal/__pycache__/__init__.cpython-311.pyc) | 1.0 KiB | Bytecode/cache Python untuk `src/temporal/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/config.cpython-310.pyc](src/temporal/__pycache__/config.cpython-310.pyc) | 4.2 KiB | Bytecode/cache Python untuk `src/temporal/config.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/config.cpython-311.pyc](src/temporal/__pycache__/config.cpython-311.pyc) | 6.3 KiB | Bytecode/cache Python untuk `src/temporal/config.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/features.cpython-310.pyc](src/temporal/__pycache__/features.cpython-310.pyc) | 2.4 KiB | Bytecode/cache Python untuk `src/temporal/features.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/features.cpython-311.pyc](src/temporal/__pycache__/features.cpython-311.pyc) | 3.9 KiB | Bytecode/cache Python untuk `src/temporal/features.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/fitting.cpython-311.pyc](src/temporal/__pycache__/fitting.cpython-311.pyc) | 17.4 KiB | Bytecode/cache Python untuk `src/temporal/fitting.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/gaussian_hmm.cpython-310.pyc](src/temporal/__pycache__/gaussian_hmm.cpython-310.pyc) | 9.1 KiB | Bytecode/cache Python untuk `src/temporal/gaussian_hmm.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/gaussian_hmm.cpython-311.pyc](src/temporal/__pycache__/gaussian_hmm.cpython-311.pyc) | 17.7 KiB | Bytecode/cache Python untuk `src/temporal/gaussian_hmm.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/pipeline.cpython-311.pyc](src/temporal/__pycache__/pipeline.cpython-311.pyc) | 20.1 KiB | Bytecode/cache Python untuk `src/temporal/pipeline.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/sequences.cpython-311.pyc](src/temporal/__pycache__/sequences.cpython-311.pyc) | 15.8 KiB | Bytecode/cache Python untuk `src/temporal/sequences.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [src/temporal/__pycache__/splits.cpython-311.pyc](src/temporal/__pycache__/splits.cpython-311.pyc) | 4.3 KiB | Bytecode/cache Python untuk `src/temporal/splits.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |


### 5.4 Cache pengujian

| File | Ukuran | Fungsi / isi | Rekomendasi |
| --- | ---: | --- | --- |
| [tests/__pycache__/test_export_sekuens.cpython-311-pytest-9.1.1.pyc](tests/__pycache__/test_export_sekuens.cpython-311-pytest-9.1.1.pyc) | 79.2 KiB | Bytecode/cache Python untuk `tests/test_export_sekuens.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/__pycache__/test_inference_smoke.cpython-311-pytest-9.1.1.pyc](tests/__pycache__/test_inference_smoke.cpython-311-pytest-9.1.1.pyc) | 7.1 KiB | Bytecode/cache Python untuk `tests/test_inference_smoke.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/__pycache__/test_inference_smoke.cpython-311.pyc](tests/__pycache__/test_inference_smoke.cpython-311.pyc) | 2.7 KiB | Bytecode/cache Python untuk `tests/test_inference_smoke.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/__pycache__/test_inference_smoke.cpython-313.pyc](tests/__pycache__/test_inference_smoke.cpython-313.pyc) | 2.6 KiB | Bytecode/cache Python untuk `tests/test_inference_smoke.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/__pycache__/test_sentiment_inference.cpython-311-pytest-9.1.1.pyc](tests/__pycache__/test_sentiment_inference.cpython-311-pytest-9.1.1.pyc) | 55.6 KiB | Bytecode/cache Python untuk `tests/test_sentiment_inference.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/__pycache__/test_sentiment_inference.cpython-311.pyc](tests/__pycache__/test_sentiment_inference.cpython-311.pyc) | 55.4 KiB | Bytecode/cache Python untuk `tests/test_sentiment_inference.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/__init__.cpython-311.pyc](tests/cleaning/__pycache__/__init__.cpython-311.pyc) | 189 B | Bytecode/cache Python untuk `tests/cleaning/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/conftest.cpython-311-pytest-9.1.1.pyc](tests/cleaning/__pycache__/conftest.cpython-311-pytest-9.1.1.pyc) | 6.9 KiB | Bytecode/cache Python untuk `tests/cleaning/conftest.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/test_audit_proposal.cpython-311-pytest-9.1.1.pyc](tests/cleaning/__pycache__/test_audit_proposal.cpython-311-pytest-9.1.1.pyc) | 39.9 KiB | Bytecode/cache Python untuk `tests/cleaning/test_audit_proposal.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/test_build_slang_dictionary.cpython-311-pytest-9.1.1.pyc](tests/cleaning/__pycache__/test_build_slang_dictionary.cpython-311-pytest-9.1.1.pyc) | 58.2 KiB | Bytecode/cache Python untuk `tests/cleaning/test_build_slang_dictionary.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/test_build_slang_dictionary.cpython-311.pyc](tests/cleaning/__pycache__/test_build_slang_dictionary.cpython-311.pyc) | 18.2 KiB | Bytecode/cache Python untuk `tests/cleaning/test_build_slang_dictionary.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/test_cleaning.cpython-311-pytest-9.1.1.pyc](tests/cleaning/__pycache__/test_cleaning.cpython-311-pytest-9.1.1.pyc) | 64.6 KiB | Bytecode/cache Python untuk `tests/cleaning/test_cleaning.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/test_loader.cpython-311-pytest-9.1.1.pyc](tests/cleaning/__pycache__/test_loader.cpython-311-pytest-9.1.1.pyc) | 32.5 KiB | Bytecode/cache Python untuk `tests/cleaning/test_loader.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/cleaning/__pycache__/test_proposal.cpython-311-pytest-9.1.1.pyc](tests/cleaning/__pycache__/test_proposal.cpython-311-pytest-9.1.1.pyc) | 51.9 KiB | Bytecode/cache Python untuk `tests/cleaning/test_proposal.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/temporal/__pycache__/__init__.cpython-310.pyc](tests/temporal/__pycache__/__init__.cpython-310.pyc) | 170 B | Bytecode/cache Python untuk `tests/temporal/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/temporal/__pycache__/__init__.cpython-311.pyc](tests/temporal/__pycache__/__init__.cpython-311.pyc) | 189 B | Bytecode/cache Python untuk `tests/temporal/__init__.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/temporal/__pycache__/helpers.cpython-311.pyc](tests/temporal/__pycache__/helpers.cpython-311.pyc) | 3.5 KiB | Bytecode/cache Python untuk `tests/temporal/helpers.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/temporal/__pycache__/test_gaussian_hmm.cpython-310.pyc](tests/temporal/__pycache__/test_gaussian_hmm.cpython-310.pyc) | 4.4 KiB | Bytecode/cache Python untuk `tests/temporal/test_gaussian_hmm.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/temporal/__pycache__/test_gaussian_hmm.cpython-311-pytest-9.1.1.pyc](tests/temporal/__pycache__/test_gaussian_hmm.cpython-311-pytest-9.1.1.pyc) | 9.3 KiB | Bytecode/cache Python untuk `tests/temporal/test_gaussian_hmm.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/temporal/__pycache__/test_pipeline.cpython-310.pyc](tests/temporal/__pycache__/test_pipeline.cpython-310.pyc) | 13.1 KiB | Bytecode/cache Python untuk `tests/temporal/test_pipeline.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |
| [tests/temporal/__pycache__/test_pipeline.cpython-311-pytest-9.1.1.pyc](tests/temporal/__pycache__/test_pipeline.cpython-311-pytest-9.1.1.pyc) | 27.7 KiB | Bytecode/cache Python untuk `tests/temporal/test_pipeline.py`; dibuat saat modul diimpor atau diuji. | BERSIHKAN — Python membuat ulang saat diperlukan; file sumber .py tetap disimpan. |

## 6. Direktori kosong dan metadata Git

Direktori kosong yang ditemukan:

- `outputs/features/`
- `outputs/labels/`
- `outputs/models/`
- `outputs/predictions/`
- `outputs/checkpoints/cleaning/`
- `outputs/checkpoints/sentiment/`

Direktori ini tidak menyimpan hasil eksperimen saat pemeriksaan. Boleh dipertahankan sebagai struktur yang disiapkan konfigurasi atau dibuang jika mengganggu; periksa apakah kode mengharuskan direktori tersebut ada atau membuatnya otomatis. Menghapus direktori kosong hampir tidak menghemat ruang.

Isi internal `.git/` tidak dijabarkan per objek: nama objek berbasis hash tidak menggambarkan fungsi riset dan dikelola Git. File seperti `HEAD`, `index`, konfigurasi, refs, log, hooks, dan objek riwayat tetap dipertahankan sebagai satu kesatuan. Inventaris ini tidak memeriksa isi konfigurasi sensitif atau mengusulkan penghapusan riwayat.

## 7. Masalah penataan yang perlu dibereskan

### Dokumentasi yang belum sesuai folder

- README masih menyebut `notebooks/01_clean_process_data.ipynb` dan `notebooks/02_pipeline_llm_hmm_xgb.ipynb`. Nama yang tersedia adalah `notebooks/clean-process-data.ipynb` dan `notebooks/pipeline_llm_hmm_xgb.ipynb`.
- README menautkan `docs/pipeline.md` dan `docs/laporan_cleaning.md`, yang tidak ada saat pemeriksaan.
- Status Git menunjukkan `docs/audit_pipeline.md`, `docs/hmm_rm2.md`, `docs/inferensi_sentimen.md`, `docs/laporan_cleaning.md`, `docs/metodologi.md`, `docs/pipeline.md`, serta `RM1_execution_report.md` di root dalam keadaan dihapus pada working tree. `docs/RM1_execution_report.md` tersedia. Daftar ini dicatat sebagai konteks, **bukan file yang masih ada** dan bukan perintah untuk memulihkannya.
- Beberapa kode dan tes sudah memiliki perubahan lokal ketika pemeriksaan dimulai. Panduan ini tidak mengembalikan, mengganti, atau merapikan perubahan tersebut.

### Penamaan dan path

- `data/raw/ mutiara_xiami redmi.csv` memiliki **spasi di awal nama file**. Pertahankan ejaan persis saat mengacu ke file; rename perlu disertai pembaruan mapping/path terkait.
- Nama CSV memiliki variasi kapital, spasi, serta ejaan toko/produk. Jangan menyamaratakannya sebelum memeriksa mekanisme pemetaan identitas produk dan daftar file yang dikecualikan.
- `.gitignore` memiliki entri path absolut lokal di bagian akhir; rapikan menjadi pola relatif bila memang masih diperlukan. Aturan ignore untuk CSV, Parquet, dan `outputs/` perlu dipahami saat menyiapkan backup.
- `notebooks/pipeline_llm_hmm_xgb.ipynb` berisi eksperimen probabilitas token LLM, belum implementasi HMM/XGBoost. Nama notebook sebaiknya diperjelas. Diagram arsitektur juga merupakan rancangan, bukan bukti seluruh tahap telah dijalankan.

## 8. Usulan struktur setelah ditata

Usulan ini hanya rancangan. Lokasi saat ini tetap berlaku sampai kode, notebook, dokumentasi, dan catatan asal data diperbarui bersama.

```text
Kode TA/
├── README.md
├── PANDUAN_ISI_DAN_PEMBERSIHAN_FOLDER.md
├── requirements.txt
├── .gitignore
├── src/                       # paket kode utama, tetap sesuai impor
├── tests/                     # tes dan helper sintetis
├── scripts/                   # kandidat lokasi eda_phone_reviews.py
├── notebooks/                 # notebook aktif
│   └── archive/               # debug/eksperimen yang sudah ditutup
├── data/
│   ├── raw/                   # sumber lokal yang perlu dipertahankan
│   └── processed/             # tetapkan peran agar tidak membingungkan
├── references/                # kamus aktif + sumber + provenance
├── docs/                      # panduan metodologi dan diagram utama
├── reports/                   # laporan eksperimen yang dirujuk
│   └── archive/               # laporan parsial atau telah digantikan
└── outputs/
    ├── cleaned/               # hasil dasar, v2, v3 dan audit cleaning
    ├── sentiment/             # keluaran penuh beserta ringkasan
    │   └── trials/            # kandidat lokasi uji 200 baris
    ├── audit/                 # audit, ringkasan, dan checksum sekuens
    └── logs/                  # log yang masih perlu dipertahankan
```

Untuk keluaran penelitian, lebih baik mencatat pasangan **input → konfigurasi → output → laporan** daripada hanya mengganti nama menjadi `final`. Jika hasil sekuens v3 dibuat nanti, berikan lokasi/versi tersendiri agar hasil v2 dan bukti auditnya tetap dapat ditelusuri.

## 9. Checklist tindakan berikutnya

- [ ] Backup data lokal, referensi, hasil inferensi, dan laporan yang dipakai dalam naskah TA.
- [ ] Bersihkan `.DS_Store`, `__pycache__/`, dan `.pytest_cache/` bila diinginkan.
- [ ] Putuskan apakah dua notebook 0 byte masih dibutuhkan sebagai placeholder.
- [ ] Pilih satu salinan untuk setiap pasangan duplikat; perbarui referensi sebelum menghapus salinannya.
- [ ] Tandai eksperimen aktif dan arsipkan notebook/log/uji kecil yang sudah selesai bersama ringkasannya.
- [ ] Tentukan hubungan hasil v2 dan v3 serta apakah ekspor sekuens baru diperlukan sebelum menghapus keluaran lama.
- [ ] Selaraskan sumber cleaning eksternal, file dasar `(2)`, dan default loader dengan kondisi yang benar-benar digunakan.
- [ ] Perbaiki tautan README dan putuskan status dokumentasi yang sedang dihapus di working tree.
- [ ] Jika nanti memindahkan kode atau mengubah path, jalankan pemeriksaan/tes yang relevan untuk perubahan tersebut.

File panduan ini sendiri (`PANDUAN_ISI_DAN_PEMBERSIHAN_FOLDER.md`) berfungsi sebagai indeks dan catatan keputusan pembersihan. **SIMPAN dan perbarui** setelah penataan dilakukan; jumlah 228 file di awal tidak memasukkan file panduan ini.
