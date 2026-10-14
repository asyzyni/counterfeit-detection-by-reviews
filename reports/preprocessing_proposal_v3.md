# Audit preprocessing proposal Bab 3.3

- Masukan: "outputs/cleaned/reviews_clean (2).parquet"
- Keluaran: "outputs/cleaned/reviews_clean_v3.parquet"
- Kamus: "references/slang.csv"

- n_baris: 64699
- n_berubah: 45114
- n_kosong: 58
- slang_dijalankan: True
- jumlah_slang_dinormalkan: 61341
- n_include_for_inference: 64332
- n_technical_duplicate: 347
- n_technical_duplicate_masih_include: 0
- n_text_duplicate: 4668
- n_text_duplicate_masih_include: 4647
- n_kosong_masih_include: 45
- jumlah_baris_identik: True
- urutan_review_id_seq_identik: True
- seluruh_kolom_lama_identik: True

## Per aturan (perbandingan sebelum/sesudah langkah tersebut)

Satu baris dapat berubah pada lebih dari satu aturan; hitungan tidak dijumlahkan.

| Aturan | Baris berubah |
| --- | ---: |
| nfkc | 735 |
| normalisasi_pemisah_unicode | 60 |
| hapus_url | 6 |
| hapus_emoji_karakter_khusus | 27477 |
| ringkas_tanda_baca | 10536 |
| hapus_angka_panjang | 11 |
| normalisasi_typo_huruf_berulang | 4143 |
| normalisasi_slang | 26689 |
| rapikan_spasi | 28854 |

## Asumsi dan penyimpangan

Sumber teks adalah review asli + NFKC pada tabel lama. NFKC merupakan langkah pertama, sebelum penghapusan URL; jumlah baris yang berubah ditampilkan pada aturan nfkc. Setelah NFKC, U+200B/U+200C/U+200D/U+2060/U+FEFF dihapus, sedangkan U+2028/U+2029/NBSP diganti spasi biasa pada aturan normalisasi_pemisah_unicode. NBSP yang telah menjadi spasi pada NFKC dihitung pada aturan nfkc. Kolom review, text_light, dan seluruh kolom lama dipertahankan apa adanya. Karakter di luar huruf Unicode, digit desimal, spasi, dan . , ! ? ' - diganti spasi agar kata tidak tergabung. Tanda baca yang sama dan berurutan diringkas; kombinasi seperti !? tetap. Huruf berulang dihitung sebagai normalisasi typo; tidak ada lowercase paksa.

Asumsi angka konservatif: hanya deretan >= 6 digit yang berdiri sendiri dihapus. Angka pendek dan alfanumerik (128gb, 5g, a07, iphone 17) tetap. Nomor yang dipisah spasi/tanda baca tidak digabung untuk dihapus.

Pencocokan slang tidak peka kapital, sekali jalan pada batas kata/frasa; kapital semua/awal token dipertahankan. Frasa terlindungi diprioritaskan. Mempertahankan kw adalah PENYIMPANGAN disengaja dari contoh normalisasi proposal 3.3.

PROTECTED_TERMS: `kw`, `ori`, `original`, `palsu`, `replika`, `imitasi`, `hdc`, `bukan ori`, `sesuai deskripsi`

## Semantik duplikat dan rekomendasi

is_technical_duplicate: produk + text_light + timestamp berjam identik; kemunculan setelah yang pertama dikecualikan dari include_for_inference. is_text_duplicate: teks sama dalam produk, masih >= 2 ulasan setelah duplikat teknis disisihkan; flag dapat menempel juga pada baris teknis dalam grup tersebut. Teks sama tidak otomatis spam.

Rekomendasi: pertahankan flag saat ini dan tinjau sampel duplikat sebelum mengubah aturan. CLI sentimen tetap menginferensi semua teks nonkosong, termasuk flag include=False, seperti versi lama; downstream dapat memfilter. Tidak ada perubahan include_for_inference. Teks proposal kosong tetap menjadi baris dengan probabilitas NaN.

## 20 contoh acak (seluruh tabel, random_state=42)

| review_id | review | text_proposal |
| --- | --- | --- |
| gudang phoncell&#124;infinix smart 20 ram 8+256 gb (extend ram 4+4 gb)__5ab2dd57__280 | mantap sesuai pesanan | mantap sesuai pesanan |
| ext gadget&#124;iphone 13__21f9d384__33 | barang sampai dengan aman, bagus, kondisi sesuai dengan deskripsi dan semoga awet terimakasih | barang sampai dengan aman, bagus, kondisi sesuai dengan deskripsi dan semoga awet terimakasih |
| panda elektronik store&#124;oppo a3s hp ram 6:128g__bf9fea35__153 | bagus bagus | bagus bagus |
| king77.toko&#124;vivo y71__a0ea9a1a__156 | naik &#124; Cuman Casnya lambat mengisi<br>Selain itu semuanya bagus<br>Mantap &#124; awet | naik Cuman Casnya lambat mengisi Selain itu semuanya bagus Mantap awet |
| gudang phoncell&#124;infinix hot 70 nfc 12+256 gb__c53210ab__1650 | Desain yang elegan, suka sekali!, &#124; bagus | Desain yang elegan, suka sekali!, bagus |
| digimap official shop&#124;iphone 17__572a69ce__460 | Dah nyampeeee semoga awet terimakasih🫶🫶🫶🫶 | Deh sampai semoga awet terimakasih |
| leon_cell99&#124;infinix itel a80 8+128 gb__64b92700__334 | bagus,hpnya mantap pengirimannya cepat.kurirnya langganan.ok | bagus,hpnya mantap pengirimannya cepat.kurirnya langganan.ok |
| digimap official shop&#124;iphone 15__8f9d7173__887 | Paket cepet sampai dan hp sesuai dengan diskripsi dan pesenan &#124; blue &#124; original | Paket cepat sampai dan hp sesuai dengan diskripsi dan pesenan blue original |
| gemilang cell&#124;oppo a58__8bf1e5ea__1486 | ok brang bagus sesuai dgn harga ny | ok brang bagus sesuai dengan harga ny |
| gudang phoncell&#124;infinix hot 70 nfc 12+256 gb__c53210ab__327 | top markotop.... | top markotop. |
| space mobile official store&#124;samsung galaxy a17 lte__681f7070__20 | Alhamdulilah barang mendarat dengan selamat, terima kasih buat penjual yang amanah, semoga sukses selalu. | Alhamdulilah barang mendarat dengan selamat, terima kasih buat penjual yang amanah, semoga sukses selalu. |
| yukehpstore1&#124;vivo y17__b7ac7dcf__57 | kirain beneran murah ternyata HP KW (palsu) 😩 &#124; minus voicenya &#124; warna sesuai | mengira benaran murah ternyata HP KW palsu minus voicenya warna sesuai |
| gemilang cell&#124;oppo a58__8bf1e5ea__971 | tidak sesuai warna gold yg datang black, tapi oke pengiriman cepet, semoga awet. | tidak sesuai warna gold yg datang black, tapi oke pengiriman cepat, semoga awet. |
| abassystore.id&#124;vivo y71__21d61d5d__57 | sesuai semoga awet | sesuai semoga awet |
| gemilang cell&#124;infinix smart 20 nfc 8+256__b5ddf306__511 | Alhamdulillah barang smpai dengan selamat,, paking rapi GK ada cacat, top pokonya amanah,, trimakasih juga buat kurirnya yg gercep, 👍🏻👍🏻👍🏻 | Alhamdulillah barang sampai dengan selamat, paking rapi GK ada cacat, top pokoknya amanah, trimakasih juga buat kurirnya yg gercep, |
| 1688.shopp&#124;oppo a3s__d69de855__818 | sempet panik karena gk bisa login ke email, akhir nya chat admin alhamdulillah sudah bisa sekarang,, &#124; barang dan kelengkapan Oke semua | sempat panik karena gk bisa login ke email, akhir nya chat admin alhamdulillah sudah bisa sekarang, barang dan kelengkapan Oke semua |
| gudang phoncell&#124;infinix smart 20 8+256 gb__ba9ff761__989 | Koneksi 4G cepat dan handal., | Koneksi 4G cepat dan handal., |
| gemilang cell&#124;oppo a58__8bf1e5ea__973 | Memori yang dapat diperluas untuk penyimpanan lebih banyak, Prosesor cepat dan efisien, | Memori yang dapat diperluas untuk penyimpanan lebih banyak, Prosesor cepat dan efisien, |
| hp mart&#124;infinix hot 50 8:128gb - mediatek helio__9b392c7a__9 | walaupun di proses nya sangat lama tapi barang klop mantap hatur nhun lazada sangat amanah | walaupun di proses nya sangat lama tapi barang klop mantap hatur nhun lazada sangat amanah |
| digimap official shop&#124;iphone 16__98bfc4d9__174 | berfungsi dengan baik &#124; baik &#124; pengiriman cepat respon penjual cepat | berfungsi dengan baik baik pengiriman cepat respon penjual cepat |

## Verifikasi artefak v3

- Asumsi masukan: gunakan `outputs/cleaned/reviews_clean (2).parquet`, sumber yang sama dengan laporan v2, dan `references/slang.csv` yang ada.
- SHA256 `reviews_clean_v3.parquet`: `f413ed0d6b9ae3cd4b18438d53940c3308ace55e876ffc90ff0f1be04d0ca690`.
- Jumlah baris: 64699; `text_proposal_kosong=True`: 58; kosong dengan `include_for_inference=True`: 45.
- NFKC mengubah 735 baris; pembersihan zero-width/pemisah Unicode berikutnya mengubah 60 baris. Hitungan antaraturan dapat tumpang tindih.
- `text_proposal` berbeda dari v2 pada 668 baris.
- Seluruh 27 kolom lama, nilai, dtype, indeks, urutan baris, dan flag identik dengan sumber serta v2. `verify_original_columns` dipertahankan dan lulus sebelum/sesudah round-trip parquet. Kolom `text_lex` tidak tersedia pada masukan produksi; invariansinya dicakup tes sintetis.
- Kelima zero-width dan U+2028/U+2029/NBSP tidak ditemukan lagi pada `text_proposal` keluaran. `nama_toko` dan `nama_produk` hanya dipindai, tidak dinormalisasi.
- SHA256 sumber, v2, kamus, dan konfigurasi dibandingkan sebelum/sesudah dan tetap sama:
  - `outputs/cleaned/reviews_clean (2).parquet`: `c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0`
  - `outputs/cleaned/reviews_clean_v2.parquet`: `dc20870f56b1b8e3d85597f51eeb0021cfd3065896518da5d4c462cb729889bb`
  - `references/slang.csv`: `f4320a27ec257a508cec6bdf825aef9f2a3232658f0d18b462693c5340b4f7e2`
  - `src/cleaning/config.py`: `6490d79525efc5249ce10dc81edaee89364c71c6c89fdd06447e90fbd68c4c44`

Perintah pembuatan:

```sh
python -m src.cleaning.run_proposal --input 'outputs/cleaned/reviews_clean (2).parquet' --output outputs/cleaned/reviews_clean_v3.parquet --report reports/preprocessing_proposal_v3.md --slang-csv references/slang.csv
```

## Validasi tes

- `python -m pytest tests/cleaning/test_proposal.py -q`: 44 lulus.
- `python -m pytest -q`: 215 lulus, 2 dilewati, 1 gagal pada `tests/temporal/test_pipeline.py::TestHmmlearnParity::test_forward_matches_hmmlearn_score_and_posterior`.
- Kegagalan yang sama direproduksi pada salinan bersih HEAD `99422837a03711d83fe2bdc022adc8473162ccad`, dengan interpreter dan `hmmlearn==0.3.3` yang sama: `src/temporal/fitting.py:120`, bentuk kovarians `(2, 3, 3)`, diharapkan `(2, 3)`. Baseline tidak memuat modul cleaning dan memakai data sintetis; kegagalan sudah ada sebelum perubahan proposal.
- Tidak ada commit atau push.

## Audit Unicode text_proposal v3

Sumber: `/Users/asyzyni/Desktop/TA/Kode dan Eksperimen/Kode TA/outputs/cleaned/reviews_clean_v3.parquet`; 64,699 baris.

### Asumsi

- Huruf Latin beraksen yang dikecualikan: nama Unicode mengandung LATIN, kategori Lu/Ll/Lt, dan NFKD terdiri atas satu huruf ASCII diikuti satu atau lebih combining mark. Huruf nondekomposisi seperti ø/ß/æ tetap ditandai agar definisi konservatif dan eksplisit.
- Hitungan karakter adalah setiap kemunculan dalam setiap sel, tanpa deduplikasi baris, nama toko, atau nama produk. Jumlah baris dihitung terpisah untuk setiap kolom; nilai kosong diabaikan.
- Sepuluh teratas diurutkan berdasarkan jumlah kemunculan gabungan menurun, lalu code point menaik. Konteks adalah kemunculan pertama per karakter per kolom; row_position_1based adalah posisi data mulai 1.
- Karakter tak terlihat (kategori C, Z, atau mark) ditampilkan sebagai U+XXXX di konteks; satu spasi ASCII biasa tetap spasi. Parquet hanya dibaca.

### Ringkasan

| Kolom | Kemunculan | Baris terdampak | Karakter berbeda | Latin beraksen dikecualikan |
| --- | ---: | ---: | ---: | ---: |
| text_proposal | 329 | 8 | 45 | 33 |
| nama_toko | 0 | 0 | 0 | 0 |
| nama_produk | 0 | 0 | 0 | 0 |

### Sepuluh karakter teratas gabungan

| Karakter | Code point dan nama Unicode | Total | text_proposal | nama_toko | nama_produk | Contoh |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ل | U+0644 ARABIC LETTER LAM | 42 | 42 | 0 | 0 | text_proposal, baris 11846: ا لس ل ام ع ل ي ك م و ر ح م ة الل ـه و… |
| ᥲ | U+1972 TAI LE LETTER TONE-4 | 34 | 34 | 0 | 0 | text_proposal, baris 51387: ძіᥣᥙᥲr ძᥙgᥲᥲᥒ.һ ᑲᥲgᥙs mᥲᥒtᥙᥣ ძᥲᥒ sᥲᥒgᥲt … |
| ا | U+0627 ARABIC LETTER ALEF | 27 | 27 | 0 | 0 | text_proposal, baris 11846: ا لس ل ام ع ل ي ك م و ر ح م ة الل ـه… |
| م | U+0645 ARABIC LETTER MEEM | 24 | 24 | 0 | 0 | text_proposal, baris 11846: ا لس ل ام ع ل ي ك م و ر ح م ة الل ـه و ب ر ك… |
| ᥒ | U+1952 TAI LE LETTER NGA | 15 | 15 | 0 | 0 | text_proposal, baris 51387: ძіᥣᥙᥲr ძᥙgᥲᥲᥒ.һ ᑲᥲgᥙs mᥲᥒtᥙᥣ ძᥲᥒ sᥲᥒgᥲt mᥱmᥙᥲskᥲ… |
| і | U+0456 CYRILLIC SMALL LETTER BYELORUSSIAN-UKRAINIAN I | 13 | 13 | 0 | 0 | text_proposal, baris 51387: ძіᥣᥙᥲr ძᥙgᥲᥲᥒ.һ ᑲᥲgᥙs mᥲᥒtᥙᥣ ძᥲᥒ sᥲᥒg… |
| ᥱ | U+1971 TAI LE LETTER TONE-3 | 13 | 13 | 0 | 0 | text_proposal, baris 51387: … ძᥙgᥲᥲᥒ.һ ᑲᥲgᥙs mᥲᥒtᥙᥣ ძᥲᥒ sᥲᥒgᥲt mᥱmᥙᥲskᥲᥒ,tᥱrіmᥲ ksіһ ᑲᥙᥲt t᥆k᥆ᥒᥡᥲ ᑲᥙ… |
| ه | U+0647 ARABIC LETTER HEH | 12 | 12 | 0 | 0 | text_proposal, baris 11846: ا لس ل ام ع ل ي ك م و ر ح م ة الل ـه و ب ر كات ه ا لل ه م ص ل ع ل ى س ي… |
| ᥙ | U+1959 TAI LE LETTER PA | 10 | 10 | 0 | 0 | text_proposal, baris 51387: ძіᥣᥙᥲr ძᥙgᥲᥲᥒ.һ ᑲᥲgᥙs mᥲᥒtᥙᥣ ძᥲᥒ sᥲᥒgᥲt… |
| ح | U+062D ARABIC LETTER HAH | 9 | 9 | 0 | 0 | text_proposal, baris 11846: ا لس ل ام ع ل ي ك م و ر ح م ة الل ـه و ب ر كات ه ا لل ه م ص … |

Aksara non-Latin yang tersisa dicatat untuk audit; NFKC tidak melakukan transliterasi.
