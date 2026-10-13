# Profil dan penyaringan kamus alay

- Sumber lokal: "references/sumber/colloquial-indonesian-lexicon.csv"; SHA-256: `48e7450875b75935cf15b200524e97927a8c27f8169e658cf4a73993c7818d76`.
- Encoding terverifikasi: UTF-8 strict; parser CSV strict dengan newline dipertahankan.
- Header terverifikasi: `slang,formal,in_dictionary,context,categories`.
- Record CSV: **15006**; karakter LF: **15397**.
  Hitungan LF bukan jumlah record: field komentar berkutip dapat memuat newline.
- Slang unik casefold+trim: **4330**; setelah rapikan seluruh spasi: **4330**.
- Pasangan unik mentah: **4410**; setelah normalisasi: **4409**.
- Kunci multi-baku: **71**; pasangan terkait: **150**.
- Kunci sama dengan baku (tidak peka kapital): **0**.
- Kunci 1 karakter: **13**; 2 karakter: **131** (sebelum filter lain).
- Baris slang/formal kosong setelah normalisasi: **0**.
- Kunci yang muncul >1 record: **1337**; kunci dengan >1 context berbeda: **1293**.
  Ini memverifikasi adanya pengulangan kunci dalam berbagai konteks; tidak membuktikan satu record selalu satu konteks unik. Isi context tidak disalin.
- Angka 3.592 kata dari halaman HF (informasi pengguna) tidak sama dengan ekspor lokal ini; penyebab perbedaan tidak disimpulkan tanpa bukti sumber lain.

## Encoding dan penanda mojibake

- Pasangan slang/formal non-ASCII: 0.
- Pasangan dengan penanda ('�', 'Ã', 'Â', 'â€', 'ðŸ', 'ï»¿'): 0.
- Pemeriksaan penanda bersifat heuristik. Untuk ekspor ini, bila seluruh slang/formal ASCII, tidak ditemukan persoalan decoding non-ASCII pada dua kolom tersebut.

## Nilai in_dictionary yang teramati

Makna kolom belum dibuktikan oleh isi CSV; nilai tidak digunakan menyaring.

| Nilai literal | Record |
| --- | ---: |
| False | 1284 |
| True | 13722 |

## Categories yang teramati

190 nilai teks berbeda; 17 label di antara tanda petik tunggal: `abreviasi`, `afiksasi`, `akronim`, `anaptiksis`, `coinage`, `elongasi`, `homofon`, `metatesis`, `modifikasi vokal`, `monoftongisasi`, `naturalisasi`, `pungtuasi`, `reduplikasi`, `salah ketik`, `subtitusi`, `word-value letter`, `zeroisasi`.
Format berisi representasi daftar/array, misalnya `['afiksasi' 'elongasi']`; tidak diparsing sebagai Python dan tidak digunakan menyaring. Ejaan label berasal dari berkas; arti kategori tidak ditebak.

| 10 nilai paling sering | Record |
| --- | ---: |
| ['abreviasi'] | 6366 |
| ['zeroisasi'] | 1890 |
| ['elongasi'] | 1772 |
| ['afiksasi'] | 974 |
| ['modifikasi vokal'] | 515 |
| ['reduplikasi'] | 433 |
| ['anaptiksis'] | 285 |
| ['abreviasi' 'zeroisasi'] | 281 |
| ['monoftongisasi'] | 197 |
| ['akronim'] | 183 |

## 15 contoh kunci multi-baku (urut alfabet)

| Slang | Semua baku berbeda |
| --- | --- |
| anterin | antar; antarkan |
| bagiin | bagikan; membagikan |
| beliin | belikan; membelikan |
| bikinin | bikin; membikin |
| bkn | bikin; bukan |
| buatin | buatkan; membuat |
| buktiin | buktikan; membuktikan |
| ciptain | ciptakan; menciptakan |
| cobain | coba; mencoba |
| coment | komen; komentar |
| d | ada; di |
| da | ada; dah; sudah |
| daah | dah; deh |
| dahh | dah; deh |
| dapetin | mendapati; mendapatkan |

## Aturan operasional (asumsi pengguna)

1. Kunci casefold dan spasi dirapikan; baku hanya dirapikan spasinya (kapital dipertahankan).
2. Baris kosong dibuang; pasangan identik setelah normalisasi dideduplikasi. Duplikat tetap dicatat sebagai satu baris pengecualian per record yang dibuang.
3. Identitas slang/baku dibandingkan tanpa peka kapital dan dibuang.
4. PROTECTED_TERMS dicocokkan sebagai substring literal tanpa peka kapital pada kedua sisi. Ini juga membuang substring insidental seperti ori dalam favorit/tutorial; bukan pencocokan batas kata.
5. Konflik baku dihitung dari SEMUA pasangan nonkosong sebelum filter lain. Seluruh pemetaan kunci konflik dibuang; frekuensi tidak memilih pemenang.
6. Kunci panjang satu dan dua karakter dibuang secara default; tidak ada whitelist diam-diam.
7. Satu alasan utama per record, prioritas: kosong → duplikat_pasangan → slang_sama_baku → protected_terms → multi_baku → kunci_satu_huruf → kunci_dua_huruf. Karena alasan dapat tumpang tindih, hitungan alasan pendek berbeda dari profil awal.
8. Keluaran diurutkan deterministik; tidak memakai in_dictionary, categories, atau context untuk filter.
PROTECTED_TERMS: `kw`, `ori`, `original`, `palsu`, `replika`, `imitasi`, `hdc`, `bukan ori`, `sesuai deskripsi`.

## Rekonsiliasi penyaringan

| Alasan utama | Record dibuang |
| --- | ---: |
| kosong | 0 |
| duplikat_pasangan | 10597 |
| slang_sama_baku | 0 |
| protected_terms | 9 |
| multi_baku | 150 |
| kunci_satu_huruf | 7 |
| kunci_dua_huruf | 119 |
| **Total dibuang** | **10882** |
| **Diterima (pasangan/kunci)** | **4124** |

Rekonsiliasi: 15006 = 10882 + 4124. Pasangan unik = 4409; setelah deduplikasi ada 285 pasangan unik tidak diterima.

Gerbang: **206/4330 = 4.757506% kunci unik dikeluarkan**. Status >30%: **False**.
Penyebut adalah kunci nonkosong unik sebelum filter; pembilang adalah kunci yang tidak memiliki pemetaan diterima. Duplikasi konteks tidak dihitung sebagai kehilangan kunci.
Kamus: "references/slang.csv" (slang,baku); pengecualian: "references/slang_dikecualikan.csv" (slang,baku,alasan). Kedua CSV tidak memuat context.

## Seluruh kunci dua karakter untuk tinjauan

Termasuk kunci yang alasan utamanya multi-baku/protected_terms. Semua ditolak pada run ini.

| Slang | Baku | Alasan utama pasangan nonduplikat |
| --- | --- | --- |
| ad | ada | kunci_dua_huruf |
| ae | saja | kunci_dua_huruf |
| aj | saja | kunci_dua_huruf |
| ak | aku | kunci_dua_huruf |
| am | sama | kunci_dua_huruf |
| ap | apa | kunci_dua_huruf |
| aq | aku | kunci_dua_huruf |
| b2 | berdua | kunci_dua_huruf |
| b3 | bertiga | kunci_dua_huruf |
| bc | baca | kunci_dua_huruf |
| bg | bang | kunci_dua_huruf |
| br | baru | kunci_dua_huruf |
| bs | bisa | kunci_dua_huruf |
| bt | buat | kunci_dua_huruf |
| bw | bawa | kunci_dua_huruf |
| bz | bisa | kunci_dua_huruf |
| cb | coba | kunci_dua_huruf |
| cd | celana dalam | kunci_dua_huruf |
| cm | cuma | kunci_dua_huruf |
| co | cowok | kunci_dua_huruf |
| cr | cara | kunci_dua_huruf |
| cw | cewek | kunci_dua_huruf |
| da | ada | multi_baku |
| da | dah | multi_baku |
| da | sudah | multi_baku |
| de | deh | multi_baku |
| de | dek | multi_baku |
| dg | dengan | kunci_dua_huruf |
| dh | dah | multi_baku |
| dh | deh | multi_baku |
| dl | dulu | kunci_dua_huruf |
| dn | dan | kunci_dua_huruf |
| do | di | kunci_dua_huruf |
| dr | dari | kunci_dua_huruf |
| dy | dia | kunci_dua_huruf |
| ea | ya | kunci_dua_huruf |
| fi | di | kunci_dua_huruf |
| ga | enggak | kunci_dua_huruf |
| gg | enggak | kunci_dua_huruf |
| gk | enggak | kunci_dua_huruf |
| gn | begini | kunci_dua_huruf |
| gt | begitu | kunci_dua_huruf |
| gw | gue | kunci_dua_huruf |
| gx | enggak | kunci_dua_huruf |
| hr | hari | kunci_dua_huruf |
| ht | hati | kunci_dua_huruf |
| hy | hai | kunci_dua_huruf |
| in | ini | kunci_dua_huruf |
| it | itu | kunci_dua_huruf |
| iu | itu | kunci_dua_huruf |
| iy | iya | kunci_dua_huruf |
| ja | saja | kunci_dua_huruf |
| jd | jadi | kunci_dua_huruf |
| jg | juga | kunci_dua_huruf |
| jh | saja | kunci_dua_huruf |
| jk | jika | kunci_dua_huruf |
| jt | juta | kunci_dua_huruf |
| k' | kak | kunci_dua_huruf |
| ka | kak | kunci_dua_huruf |
| kh | kah | kunci_dua_huruf |
| kk | kakak | kunci_dua_huruf |
| kl | kalau | multi_baku |
| kl | kalo | multi_baku |
| km | kami | multi_baku |
| km | kamu | multi_baku |
| kn | kan | kunci_dua_huruf |
| ko | kok | kunci_dua_huruf |
| kq | kok | kunci_dua_huruf |
| kr | kira | kunci_dua_huruf |
| kt | kita | kunci_dua_huruf |
| kw | kau | protected_terms |
| kx | kak | kunci_dua_huruf |
| ky | kaya | kunci_dua_huruf |
| la | lah | kunci_dua_huruf |
| le | boleh | kunci_dua_huruf |
| lg | lagi | kunci_dua_huruf |
| lh | boleh | multi_baku |
| lh | lah | multi_baku |
| lt | lantai | kunci_dua_huruf |
| lw | kalau | multi_baku |
| lw | lu | multi_baku |
| ma | sama | kunci_dua_huruf |
| mb | mbak | kunci_dua_huruf |
| mf | maaf | kunci_dua_huruf |
| mg | moga | kunci_dua_huruf |
| mh | mah | kunci_dua_huruf |
| mk | maka | kunci_dua_huruf |
| mn | mana | kunci_dua_huruf |
| mo | mau | kunci_dua_huruf |
| ms | masih | kunci_dua_huruf |
| mw | mau | kunci_dua_huruf |
| mz | mas | kunci_dua_huruf |
| na | nya | kunci_dua_huruf |
| nd | indak | kunci_dua_huruf |
| ne | ini | kunci_dua_huruf |
| nh | nih | kunci_dua_huruf |
| ni | ini | multi_baku |
| ni | nih | multi_baku |
| nm | nama | kunci_dua_huruf |
| nt | nanti | kunci_dua_huruf |
| ny | nya | kunci_dua_huruf |
| or | orang | kunci_dua_huruf |
| pa | apa | kunci_dua_huruf |
| pd | pada | multi_baku |
| pd | percaya diri | multi_baku |
| pg | pagi | kunci_dua_huruf |
| pi | tapi | kunci_dua_huruf |
| pk | pakai | kunci_dua_huruf |
| ps | pas | kunci_dua_huruf |
| pu | punya | kunci_dua_huruf |
| px | punya | kunci_dua_huruf |
| py | punya | kunci_dua_huruf |
| qm | kamu | kunci_dua_huruf |
| qo | kok | kunci_dua_huruf |
| qt | kita | kunci_dua_huruf |
| qu | ku | kunci_dua_huruf |
| rb | ribu | kunci_dua_huruf |
| si | sih | kunci_dua_huruf |
| sj | saja | kunci_dua_huruf |
| sk | suka | kunci_dua_huruf |
| sm | sama | kunci_dua_huruf |
| so | sok | kunci_dua_huruf |
| sp | sapa | multi_baku |
| sp | siapa | multi_baku |
| sy | saya | kunci_dua_huruf |
| t4 | tempat | kunci_dua_huruf |
| td | tadi | kunci_dua_huruf |
| th | tahu | multi_baku |
| th | tahun | multi_baku |
| th | teh | multi_baku |
| tk | tak | kunci_dua_huruf |
| to | tapi | kunci_dua_huruf |
| tp | tapi | kunci_dua_huruf |
| tu | itu | kunci_dua_huruf |
| tw | tau | kunci_dua_huruf |
| tx | tak | kunci_dua_huruf |
| ud | sudah | kunci_dua_huruf |
| wk | waktu | kunci_dua_huruf |
| yg | yang | kunci_dua_huruf |
| yh | ya | kunci_dua_huruf |
| yo | ya | kunci_dua_huruf |
| yq | yang | kunci_dua_huruf |
| yu | yuk | kunci_dua_huruf |
| za | saja | kunci_dua_huruf |

## Kunci satu karakter

`c`, `d`, `g`, `j`, `k`, `m`, `q`, `s`, `u`, `w`, `x`, `y`, `z`

## Pasangan terkena PROTECTED_TERMS

| Slang | Baku |
| --- | --- |
| fav | favorit |
| favoriteeee | favorit |
| gakwad | enggak kuat |
| konsekwensi | konsekuensi |
| kw | kau |
| kwatir | kawatir |
| ngelaporin | melaporkan |
| ori | orisinal |
| tutorialx | tutorialnya |
