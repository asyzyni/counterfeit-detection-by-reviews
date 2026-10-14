# Inferensi sentimen RM 1 — run 2 lokal (MPS, text_proposal, max_length 512)

Tanggal: 2026-10-13. Semua angka di bawah disalin dari keluaran perintah yang benar-benar dijalankan.

## Lingkungan

| Item | Nilai |
|---|---|
| Mesin | Apple M2, RAM 8 GB, macOS 26.2 |
| Python | 3.11.7 (`/Users/asyzyni/.global-venv/bin/python`) |
| torch | 2.12.1 (MPS available=True, built=True) |
| transformers | 5.12.1 |
| huggingface_hub | 1.14.0 |
| safetensors | 0.8.0 |
| pyarrow | 24.0.0 |
| pandas | 2.3.3 |
| Model | `asyzyni/indobert-nlu-smsa-counterfeit-ta` @ `5366aae2262a010df28ea88789abc3aeb4df897c` (dimuat dari cache lokal HF) |
| Env | `INDOBERT_SENTIMENT_REVISION` = hash di atas; `PYTORCH_ENABLE_MPS_FALLBACK` tidak diatur; `HF_TOKEN` tidak ada di environment (file token HF tersimpan ada, isinya tidak dibaca; model sudah ada di cache) |

## Input

- `outputs/cleaned/reviews_clean_v2.parquet`: 64.699 baris × 29 kolom; `text_proposal_kosong.sum()` = 58.
- SHA-256: `dc20870f56b1b8e3d85597f51eeb0021cfd3065896518da5d4c462cb729889bb` (cocok dengan nilai acuan).

## Pemeriksaan awal

- `python -m pytest tests/test_sentiment_inference.py tests/cleaning -q`: **153 lulus, 1 dilewati, 0 gagal**. Yang dilewati adalah tes model nyata (`tests/test_sentiment_inference.py:463`) karena `INDOBERT_SENTIMENT_MODEL_DIR` tidak diatur.
- Ruang disk bebas: 185 GiB. Semua file keluaran target belum ada sebelum dijalankan.

## Perintah

Keduanya dijalankan dengan `export INDOBERT_SENTIMENT_REVISION=5366aae2262a010df28ea88789abc3aeb4df897c`.

Uji 200 baris (log: `outputs/sentiment/logs/uji_200_lokal.log`):

```
/usr/bin/time -p python -m src.run_sentiment_inference --device mps --batch-size 16 --limit 200 \
  --input "outputs/cleaned/reviews_clean_v2.parquet" --text-col text_proposal --max-length 512 \
  --output "outputs/sentiment/uji_200_lokal.csv"
```

Run penuh (log: `outputs/sentiment/logs/run_penuh_lokal.log`):

```
nohup caffeinate -i /usr/bin/time -p python -m src.run_sentiment_inference --device mps --batch-size 16 \
  --input "outputs/cleaned/reviews_clean_v2.parquet" --text-col text_proposal --max-length 512 \
  --output "outputs/sentiment/sentimen_ulasan_v2.csv" --save-logits --save-diagnostics \
  > "outputs/sentiment/logs/run_penuh_lokal.log" 2>&1 &
```

`/usr/bin/time -p` ditambahkan hanya untuk mengukur durasi; argumen CLI sama dengan yang diinstruksikan.

## Pemeriksaan model (run penuh; uji 200 baris memberi status dan probabilitas tiga kalimat yang sama)

| Pemeriksaan | Status | Detail |
|---|---|---|
| id2label | OK | {0: negatif, 1: netral, 2: positif} |
| tokenizer.model_max_length (informasi) | OK | 128; max_length inferensi eksplisit 512 |
| max_length ≤ max_position_embeddings | OK | 512 ≤ 512 |
| Kalimat positif "Barangnya bagus banget, sesuai deskripsi, pengiriman cepat" | OK | neg 0,0011 / net 0,0008 / pos 0,9982 → positif |
| Kalimat netral "Barang sudah sampai, belum saya coba" | OK | neg 0,0765 / net 0,9016 / pos 0,0219 → netral |
| Kalimat negatif "Barang rusak, tidak sesuai foto, kecewa berat" | OK | neg 0,9971 / net 0,0006 / pos 0,0023 → negatif |
| Tidak ada NaN/Inf (16 teks) | OK | — |
| Jumlah probabilitas = 1 | OK | deviasi maks 2,22e-16 (uji 200: 1,11e-16) |
| Batch = satu per satu (≤ 1e-4) | OK | selisih maks 4,28e-07 (uji 200: 3,68e-07) |

Uji 200 baris: 200 diinferensi, 0 terpotong; argmax negatif 26 (13,0%), netral 9 (4,5%), positif 165 (82,5%); durasi 10,42 detik.

## Durasi run penuh

Mulai 14:56:06, `real 160.58` detik (≈2 menit 41 detik), 4.041 batch, device `mps`, batch 16, tanpa OOM.

## Validasi hasil (dihitung ulang secara independen dari CSV yang tertulis)

| Cek | Hasil |
|---|---|
| Jumlah baris | 64.699 (= input) |
| review_id & urutan identik dengan input | Ya; review_id unik |
| Kolom input yang hilang | Tidak ada |
| Kolom tambahan | p_negatif, p_netral, p_positif, logit_negatif, logit_netral, logit_positif, n_token, terpotong, label_argmax |
| Baris NaN probabilitas | 58, tepat sama dengan `text_proposal_kosong` (logit juga 58) |
| Deviasi maks p_neg+p_net+p_pos dari 1 | 5,55e-16 |
| Probabilitas di luar [0,1] | 0 |
| `label_argmax` = argmax probabilitas | Ya, seluruh 64.641 baris |
| Terpotong (> 512 token) | 0 |
| n_token median / p95 / maks | 15 / 48 / 447 |
| Sangat pendek (≤ 2 kata, teks tidak kosong) | 7.346 |
| Rata-rata p_netral | 0,0473 |

Distribusi argmax (64.641 baris yang diinferensi):

| Kelas | Run 2 (lokal, text_proposal, 512) | Run 1 (Kaggle, text_light, 128) |
|---|---|---|
| negatif | 7.789 (12,05%) | 12,07% |
| netral | 2.604 (4,03%) | 4,05% |
| positif | 54.248 (83,92%) | 83,87% |
| terpotong | 0 | 152 (0,23%) |

Ini hanya gambaran; tanpa label emas, tidak ada kesimpulan akurasi yang bisa diambil.

## Perbandingan per review_id dengan run 1

File run 1 yang dipakai: `/Users/asyzyni/Downloads/sentimen_ulasan.csv` (SHA-256 `064b179167a11e0aac517a6901517e90c36dffa5761459294609b83a2b03e657`; 64.699 baris, kolom `text_light`, 152 terpotong, distribusi argmax 12,07/4,05/83,87%, sehingga cocok dengan angka run 1). Argmax run 1 dihitung dari kolom probabilitasnya.

- Dibandingkan: 64.641 review_id (baris yang text_proposal-nya tidak kosong).
- Argmax berubah: **1.266 (1,96%)**.

Matriks run 1 (baris) × run 2 (kolom):

| run1 \ run2 | negatif | netral | positif |
|---|---|---|---|
| negatif | 7.389 | 88 | 312 |
| netral | 78 | 2.308 | 218 |
| positif | 345 | 225 | 53.678 |

20 contoh acak yang berubah (`random_state=42`; teks dipotong 90 karakter):

| review_id | run1 → run2 | text_light | text_proposal |
|---|---|---|---|
| xiaomi\|poco c71 (4+4gb:128gb)__464cac7f__980 | netral → positif | mantan pengiriman cepat👍 | mantan pengiriman cepat |
| smart hub store\|vivo y17__265d036e__221 | negatif → positif | baguss. normal semua | bagus normal semua |
| gudang phoncell\|infinix smart 20 8+256 gb__ba9ff761__683 | positif → negatif | terpaksa karna salah pikir :( aku pikir 6 GB RAM Asli … | terpaksa karena salah pikir aku pikir 6 GB RAM Asli … |
| vivo official store\|vivo v50 lite 5g__87fa4d60__313 | positif → negatif | Pertama kali beli HP online dan sebenere agak takut kalo bodong/ga sampe. … | … bodong ga sampai. Eh ternyata ben… |
| panda elektronik store\|oppo a3s__8a058636__411 | positif → negatif | goodd jobb gandooss | godd jobb gandos |
| mhi official store\|samsung galaxy s25 fe__fc5f3631__331 | positif → negatif | Gokil lengkap bangett paketnya. Bagus bangett. Tp sadly screen protector … | Gila lengkap banget paketnya. … |
| 1688.shopp\|oppo a3s__d69de855__899 | positif → netral | sangat baik. layar jernih dan terang toko nya sangat amanah … | 𝙨𝙖𝙣𝙜𝙖𝙩 𝙗𝙖𝙞𝙠 … (huruf Unicode matematis) |
| gudang phoncell\|infinix hot 70 nfc 12+256 gb__c53210ab__1158 | netral → positif | 📸Kualitas Kamera: | Kualitas Kamera |
| 188.toko\|oppo a31__2f67963b__253 | netral → positif | Barang datang aman👍🏽,cuma salh warn pemesanan.. Semoga awet👍🏽 | Barang datang aman ,cuma salh warn pemesanan. Semoga awet |
| infinix official shop\|infinix xpad 20 pro__7795cefd__176 | positif → negatif | tidak loading. lancar | tidak loading lancar |
| vivo official store\|vivo y21d__bb53b3ce__137 | positif → negatif | berfungsi dengan baik. blm tau | berfungsi dengan baik belum tau |
| infinix official shop\|infinix hot 70__5e85acac__322 | negatif → positif | baik. Jngan ragu belanja d toko ini … | baik Jangan ragu belanja d toko ini … |
| vivo official store\|vivo y21d__bb53b3ce__34 | negatif → positif | berfungsi dengan baik. kualitas kamera standar sihh. Sukaa🥰 … | berfungsi dengan baik kualitas kamera standar sih Suka … |
| gemilang cell\|infinix smart 20 nfc 8+256__b5ddf306__1707 | netral → positif | barang sampai dengan slmt | barang sampai dengan selamat |
| gudang phoncell\|infinix hot 70 nfc 12+256 gb__c53210ab__262 | positif → netral | realpick bangett... ho nya nyala dengan sempuran … | 𝚛𝚎𝚊𝚕𝚙𝚒𝚌𝚔 𝚋𝚊𝚗𝚐𝚎𝚝𝚝 … (huruf Unicode matematis) |
| vivo official store\|vivo y21d__bb53b3ce__166 | positif → netral | oke. sip. datang dengan selamat. terimakasih | oke sip datang dengan selamat. terimakasih |
| hp mart\|infinix hot 50 8:128gb - mediatek helio__9b392c7a__63 | negatif → positif | mantul boos q!!! | mantul boos q! |
| agres gadget official\|samsung galaxy a57 5g__816183be__139 | positif → negatif | mantap masih segel. mantap. Awalnya ragu sampe takut tertipu … | … sampai takut tertipu … |
| jiu_phoncell\|infinix hot 60i__795484fe__190 | positif → negatif | maaf peseny warna hitam yg datang warna silver,,tp GK apa" | maaf peseny warna hitam yg datang warna silver,tp GK apa |
| xiaomi\|poco c71 (4+4gb:128gb)__464cac7f__1707 | netral → positif | 📸Kualitas Kamera: | Kualitas Kamera |

## SHA-256 keluaran

| File | SHA-256 |
|---|---|
| `outputs/sentiment/sentimen_ulasan_v2.csv` (46.920.427 byte) | `fceb4d5a9e85ba52b656420982aca769e334b70c6f7bb0c1c223929399052040` |
| `outputs/sentiment/sentimen_ulasan_v2_ringkasan.md` | `6f9a0ebf662a910b810c2ca6cd4e0c61015843c2b58e54594bc9e0630b9f4a40` |
| `outputs/sentiment/sentimen_ulasan_v2_ringkasan.json` | `ecd44d6b450aa5e5a32d2eef65d05af1731ef5a9f97557ad80b6932648fedecd` |
| `outputs/sentiment/uji_200_lokal.csv` | `af6319fc50639d7b2afdf4c2902b33aef6ec4e4df91dcebdd78e12450d4e94f3` |
| `outputs/sentiment/uji_200_lokal_ringkasan.md` | `98293d6f21637b0f695c5124da903f1baf1fe7d29e60f327ddbafb0e4af7e20d` |
| `outputs/sentiment/uji_200_lokal_ringkasan.json` | `fd3893f504a077e0a0ef651f4fb884ec2bca8f67805d9e85c377ef89d184ba16` |

## Kendala dan temuan

1. **Huruf Unicode "Mathematical Alphanumeric" tidak dinormalisasi di `text_proposal`.** Ada 80 baris dengan karakter U+1D400–U+1D7FF (mis. 𝙨𝙖𝙣𝙜𝙖𝙩). Di `text_light` karakter ini sudah menjadi huruf biasa. Argmax pada 80 baris ini: netral 48, positif 27, negatif 5. Porsi netral (60%) jauh di atas porsi netral keseluruhan (4%), sehingga kemungkinan besar token tidak dikenali tokenizer (dugaan, belum diverifikasi dengan melihat token). Ini menyangkut preprocessing (kemungkinan perlu NFKC), bukan inferensi. Kode tidak diubah.
2. Contoh yang berubah menunjukkan model peka terhadap perbedaan kecil, seperti titik/emoji yang dihapus, pengulangan huruf, dan normalisasi slang (mis. "tidak loading. lancar" positif → "tidak loading lancar" negatif). Ini hanya pengamatan, bukan penilaian benar atau salah.
3. Tidak ada OOM dan tidak ada fallback ke CPU. Log melaporkan `Model siap di device: mps`.

## Asumsi

- File di `Downloads/sentimen_ulasan.csv` dianggap sebagai keluaran run 1 karena jumlah baris, kolom `text_light`, jumlah terpotong, dan distribusinya cocok dengan angka acuan. Asal-usulnya tidak diverifikasi lebih jauh.
- "Sangat pendek" dihitung sebagai ≤ 2 kata (split spasi) pada `text_proposal` yang tidak kosong. Hasilnya sama dengan angka CLI (7.346).
- Durasi diukur dengan `/usr/bin/time -p` (waktu nyata proses, termasuk memuat model dan menulis CSV).
