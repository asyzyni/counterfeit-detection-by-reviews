# Provenance kamus slang untuk text_proposal

Tanggal pencatatan: **2026-10-13 (Asia/Jakarta)**.

## Sumber dan batas verifikasi

- Sitasi yang diberikan pengguna: **Salsabila dkk. (2018), Colloquial Indonesian Lexicon**.
- Data lokal merupakan **hasil ekspor loader SEACrowd/kamus_alay, konfigurasi `*_source`, melalui `load_dataset`**,
  bukan salinan byte CSV GitHub. Versi ekspor yang dilaporkan pengguna: `datasets 3.6.0`, `seacrowd 0.2.2`,
  `Python 3.13.1`. Proses ekspor dan versi alat tersebut tidak dijalankan ulang dalam pekerjaan ini.
- URL data asal yang dipakai loader, menurut informasi pengguna:
  [CSV nasalsabila/kamus-alay](https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv).
- Lisensi: **Unknown**, menurut halaman SEACrowd sebagaimana disampaikan pengguna. Tidak diasumsikan sebagai
  izin redistribusi. Halaman daring dan lisensi upstream tidak diperiksa ulang karena pekerjaan ini tanpa unduhan/jaringan.
- Yang diverifikasi langsung: byte/hash ekspor lokal, decoding UTF-8 strict, struktur dan isi statistik CSV,
  serta hash kedua CSV turunan. Tidak ada kamus yang diunduh atau entri baru yang dikarang.

## Berkas dan SHA-256

| Berkas lokal | SHA-256 |
| --- | --- |
| "references/sumber/colloquial-indonesian-lexicon.csv" | `48e7450875b75935cf15b200524e97927a8c27f8169e658cf4a73993c7818d76` |
| "references/slang.csv" | `f4320a27ec257a508cec6bdf825aef9f2a3232658f0d18b462693c5340b4f7e2` |
| "references/slang_dikecualikan.csv" | `21293d04e4cb71c66969cecde83055b1e321f1a8b62f1fd4d7cdd2fe3cb8a52a` |

CSV ekspor: header `slang,formal,in_dictionary,context,categories`; **15.006 record**, **4.330 slang unik**
(casefold+trim, juga sama setelah merapikan spasi), **4.409 pasangan unik**, **71 kunci multi-baku / 150 pasangan**.
Ada 15.397 karakter LF; LF di field komentar berkutip menyebabkan hitungan baris fisik berbeda dari record CSV.
Angka 3.592 kata pada halaman HF yang disebut pengguna tidak sama dengan hasil ekspor lokal ini;
penyebabnya tidak diasumsikan. Sebanyak 1.337 kunci muncul berulang dan 1.293 memiliki lebih dari satu context berbeda.

`in_dictionary` hanya teramati sebagai `True` (13.722) dan `False` (1.284); arti kolom tidak disimpulkan.
`categories` memiliki 190 nilai teks berbeda. Kedua kolom tidak dipakai menyaring.
Seluruh isi slang/formal ASCII; tidak ditemukan penanda mojibake pada dua kolom ini.

## Aturan turunan (asumsi yang dapat direvisi pengguna)

1. Kunci: casefold, trim dan rapikan seluruh whitespace menjadi satu spasi. Baku: hanya rapikan whitespace,
   kapital dipertahankan. Baris dengan sisi kosong dibuang. Pasangan identik setelah normalisasi dideduplikasi.
2. Slang sama dengan baku tanpa peka kapital dibuang.
3. Slang atau baku yang memuat substring literal tanpa peka kapital dari `PROTECTED_TERMS` dibuang:
   `kw`, `ori`, `original`, `palsu`, `replika`, `imitasi`, `hdc`, `bukan ori`, `sesuai deskripsi`.
   Ini sengaja mencakup substring insidental, misalnya `ori` dalam `favorit` dan `tutorial`.
4. Konflik dihitung dari seluruh pasangan nonkosong sebelum filter lainnya: kunci yang memiliki lebih dari satu
   baku berbeda dikeluarkan seluruhnya. Tidak memilih baku berdasarkan frekuensi.
5. Kunci panjang satu dan dua karakter dikeluarkan; belum ada whitelist dua karakter.
6. Satu alasan utama per record, dengan prioritas: kosong → duplikat pasangan → identitas → protected_terms →
   multi_baku → satu karakter → dua karakter. Konflik tetap diketahui sebelum menerapkan prioritas tersebut.
7. Keluaran diurutkan deterministik. `context`, `categories`, dan `in_dictionary` tidak disalin ke CSV turunan.
   Kamus hanya `slang,baku`; pengecualian hanya `slang,baku,alasan` (termasuk baris duplikat yang dibuang).

| Alasan utama | Record dibuang | Kunci unik hilang |
| --- | ---: | ---: |
| Kosong | 0 | 0 |
| Duplikat pasangan | 10.597 | 0 secara langsung |
| Slang sama dengan baku | 0 | 0 |
| PROTECTED_TERMS | 9 | 9 |
| Multi-baku | 150 | 71 |
| Satu karakter | 7 | 7 |
| Dua karakter | 119 | 119 |
| **Total** | **10.882** | **206** |

Kamus akhir: **4.124 pasangan / kunci**. Rekonsiliasi record: **15.006 = 10.882 + 4.124**.
Sebanyak **285 pasangan unik** tidak diterima. Profil awal mempunyai 13 kunci satu karakter dan 131 kunci dua karakter;
sebagian mendapat alasan utama konflik/protected_terms, sehingga berbeda dari hitungan alasan di tabel.

Gerbang kehilangan kunci: **206 / 4.330 = 4,757506%**, **tidak melebihi 30%**.
Duplikasi konteks tidak dihitung sebagai kehilangan kunci. Bila gerbang terpicu pada sumber lain, konverter menulis
bahan tinjauan lalu exit 2; pembuatan v2 harus menunggu keputusan pengguna.

Profil lengkap dan daftar seluruh kunci dua karakter: "reports/profil_kamus_alay.md".

## Reproduksi lokal

```bash
python -m src.cleaning.build_slang_dictionary \
  --input "references/sumber/colloquial-indonesian-lexicon.csv" \
  --output "references/slang.csv" \
  --excluded "references/slang_dikecualikan.csv" \
  --profile "reports/profil_kamus_alay.md" \
  --expected-sha256 "48e7450875b75935cf15b200524e97927a8c27f8169e658cf4a73993c7818d76"
```

Konverter menolak file yang sudah ada; gunakan ketiga nama keluaran baru untuk reproduksi berikutnya.
Tidak ada jaringan, perubahan file sumber, atau penggunaan komentar Instagram dalam preprocessing ulasan.
`text_proposal` hanya untuk masukan IndoBERT; `text_light`, `text_lex`, dan kolom lama dipertahankan.

## Keputusan publikasi dan usulan saja

Pengguna memutuskan pada 2026-10-13: **"references/slang.csv" tetap lokal dan diabaikan Git**.
Aturan `*.csv` pada ".gitignore" yang sudah ada mengabaikan kedua CSV turunan dan CSV ekspor.
Tidak ada perubahan ".gitignore", staging, commit, atau push dalam pekerjaan ini.
Usulan penegasan untuk melindungi seluruh folder mentah, termasuk format selain CSV:

```diff
--- a/.gitignore
+++ b/.gitignore
@@
 *.csv
+references/sumber/
```

Usulan ini belum diterapkan. Ketidakjelasan lisensi tetap perlu diselesaikan sebelum mempertimbangkan redistribusi;
keputusan saat ini adalah menyimpan data kamus lokal. Kunci dua karakter tetap dikeluarkan sampai ada keputusan
spesifik setelah pengguna meninjau daftar.
