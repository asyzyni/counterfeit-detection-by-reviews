# Audit preprocessing proposal Bab 3.3

- Masukan: "outputs/cleaned/reviews_clean.parquet"
- Keluaran parquet v2 belum ditulis.
- PARSIAL: kamus slang belum tersedia; slang belum dijalankan. Angka di bawah belum final.

- n_baris: 64699
- n_berubah: 36378
- n_kosong: 58
- slang_dijalankan: False
- jumlah_slang_dinormalkan: 0
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
| hapus_url | 6 |
| hapus_emoji_karakter_khusus | 27710 |
| ringkas_tanda_baca | 10478 |
| hapus_angka_panjang | 11 |
| normalisasi_typo_huruf_berulang | 4143 |
| normalisasi_slang | 0 |
| rapikan_spasi | 28998 |

## Asumsi dan penyimpangan

Sumber teks adalah review asli pada tabel lama, bukan text_light. Karakter di luar huruf Unicode, digit desimal, spasi, dan . , ! ? ' - diganti spasi agar kata tidak tergabung. Tanda baca yang sama dan berurutan diringkas; kombinasi seperti !? tetap. Huruf berulang dihitung sebagai normalisasi typo; tidak ada lowercase paksa.

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
| digimap official shop&#124;iphone 17__572a69ce__460 | Dah nyampeeee semoga awet terimakasih🫶🫶🫶🫶 | Dah nyampe semoga awet terimakasih |
| leon_cell99&#124;infinix itel a80 8+128 gb__64b92700__334 | bagus,hpnya mantap pengirimannya cepat.kurirnya langganan.ok | bagus,hpnya mantap pengirimannya cepat.kurirnya langganan.ok |
| digimap official shop&#124;iphone 15__8f9d7173__887 | Paket cepet sampai dan hp sesuai dengan diskripsi dan pesenan &#124; blue &#124; original | Paket cepet sampai dan hp sesuai dengan diskripsi dan pesenan blue original |
| gemilang cell&#124;oppo a58__8bf1e5ea__1486 | ok brang bagus sesuai dgn harga ny | ok brang bagus sesuai dgn harga ny |
| gudang phoncell&#124;infinix hot 70 nfc 12+256 gb__c53210ab__327 | top markotop.... | top markotop. |
| space mobile official store&#124;samsung galaxy a17 lte__681f7070__20 | Alhamdulilah barang mendarat dengan selamat, terima kasih buat penjual yang amanah, semoga sukses selalu. | Alhamdulilah barang mendarat dengan selamat, terima kasih buat penjual yang amanah, semoga sukses selalu. |
| yukehpstore1&#124;vivo y17__b7ac7dcf__57 | kirain beneran murah ternyata HP KW (palsu) 😩 &#124; minus voicenya &#124; warna sesuai | kirain beneran murah ternyata HP KW palsu minus voicenya warna sesuai |
| gemilang cell&#124;oppo a58__8bf1e5ea__971 | tidak sesuai warna gold yg datang black, tapi oke pengiriman cepet, semoga awet. | tidak sesuai warna gold yg datang black, tapi oke pengiriman cepet, semoga awet. |
| abassystore.id&#124;vivo y71__21d61d5d__57 | sesuai semoga awet | sesuai semoga awet |
| gemilang cell&#124;infinix smart 20 nfc 8+256__b5ddf306__511 | Alhamdulillah barang smpai dengan selamat,, paking rapi GK ada cacat, top pokonya amanah,, trimakasih juga buat kurirnya yg gercep, 👍🏻👍🏻👍🏻 | Alhamdulillah barang smpai dengan selamat, paking rapi GK ada cacat, top pokonya amanah, trimakasih juga buat kurirnya yg gercep, |
| 1688.shopp&#124;oppo a3s__d69de855__818 | sempet panik karena gk bisa login ke email, akhir nya chat admin alhamdulillah sudah bisa sekarang,, &#124; barang dan kelengkapan Oke semua | sempet panik karena gk bisa login ke email, akhir nya chat admin alhamdulillah sudah bisa sekarang, barang dan kelengkapan Oke semua |
| gudang phoncell&#124;infinix smart 20 8+256 gb__ba9ff761__989 | Koneksi 4G cepat dan handal., | Koneksi 4G cepat dan handal., |
| gemilang cell&#124;oppo a58__8bf1e5ea__973 | Memori yang dapat diperluas untuk penyimpanan lebih banyak, Prosesor cepat dan efisien, | Memori yang dapat diperluas untuk penyimpanan lebih banyak, Prosesor cepat dan efisien, |
| hp mart&#124;infinix hot 50 8:128gb - mediatek helio__9b392c7a__9 | walaupun di proses nya sangat lama tapi barang klop mantap hatur nhun lazada sangat amanah | walaupun di proses nya sangat lama tapi barang klop mantap hatur nhun lazada sangat amanah |
| digimap official shop&#124;iphone 16__98bfc4d9__174 | berfungsi dengan baik &#124; baik &#124; pengiriman cepat respon penjual cepat | berfungsi dengan baik baik pengiriman cepat respon penjual cepat |

## Integritas keluaran lama

- "outputs/cleaned/reviews_clean.parquet" SHA-256: `c15e1ab7c1bf4de37d61e430fbe4d54b314938050cbd3ac3d294bf063a13ccb0`
- "outputs/sentiment/uji_200.csv" SHA-256: `c51e8954c9bcd1da6bae355f119a9ebf0a09ebb38472e91d5302db181523e202`
- "outputs/sentiment/uji_200_ringkasan.json" SHA-256: `b1e288823a9d07b174084bcd928c4cd5715bca6a3816f5fe07ad6ae47fef9046`
- "outputs/sentiment/uji_200_ringkasan.md" SHA-256: `4d11c7ac7adedd0420270951507c52d7bc406ce97cf36d04dd8af8c86f4efbf7`

Parquet v2 final belum dibuat: menunggu CSV kamus slang,baku dari pengguna.
