# Inferensi sentimen RM 1

Modul: `src/sentiment_inference.py` (fungsi) dan `src/run_sentiment_inference.py` (CLI).
Model: `asyzyni/indobert-nlu-smsa-counterfeit-ta` (IndoBERT IndoNLU, fine-tune SmSA, 3 kelas, max_length 128).
Kolom keluaran: `p_negatif`, `p_netral`, `p_positif` (urutan kolom diturunkan dari `config.id2label`).

## Yang harus disiapkan
1. Hash commit **lengkap (40 karakter)** model di Hugging Face. Hash singkat (mis. `5366aae`) ditolak, karena tidak dipin.
2. Token Hugging Face lewat variabel lingkungan `HF_TOKEN` (Kaggle: Secrets). Jangan menulisnya di kode, notebook, atau log.
3. `pip install -r requirements.txt` (butuh torch, transformers, huggingface_hub, safetensors, pyarrow).

## Menjalankan (Mac M2, MPS)
    export HF_TOKEN=...
    export INDOBERT_SENTIMENT_REVISION=<hash 40 karakter>
    python -m src.run_sentiment_inference --device mps --limit 200 --output outputs/sentiment/uji_200.csv   # uji kecil dulu
    python -m src.run_sentiment_inference --device mps --batch-size 16                                      # data penuh

Input bawaan `outputs/cleaned/reviews_clean.parquet` (kolom teks `text_light`). Semua baris dipertahankan;
teks kosong mendapat probabilitas NaN dan kolom `include_for_inference` dari tahap cleaning tetap ikut,
jadi tahap berikutnya bisa memfilter. Keluaran: CSV + `*_ringkasan.json` + `*_ringkasan.md`. File yang sudah ada tidak ditimpa.

Opsi: `--save-logits` (logit mentah, untuk temperature scaling/ALR/CLR), `--save-diagnostics` (`n_token`, `terpotong`),
`--model-dir` (folder model lokal, tanpa jaringan), `--device cuda` (Kaggle), `--no-argmax`.
Device tidak pernah jatuh ke CPU secara diam-diam: `auto` memilih CUDA lalu MPS, dan berhenti dengan pesan bila keduanya tidak ada.

## Pemeriksaan otomatis sebelum run penuh
id2label = {0:negatif, 1:netral, 2:positif}; `tokenizer.model_max_length == 128`; tanpa NaN/Inf; jumlah probabilitas = 1 (<=1e-6);
hasil batch = hasil satu-satu (<=1e-4). Bila ada yang gagal, run berhenti.

## Tes
    python -m pytest tests/test_sentiment_inference.py         # logika, tanpa model (model palsu)
    INDOBERT_SENTIMENT_MODEL_DIR="<folder model>" INDOBERT_SENTIMENT_DEVICE=mps python -m pytest tests/test_sentiment_inference.py -s   # model nyata, 10 kalimat

## Keterbatasan (tulis juga di naskah)
SmSA adalah ulasan umum, bukan marketplace. Kualitas pada ulasan ponsel belum terukur dan butuh sampel berlabel manual
(rencana 150 baris, bukan dari rating bintang). Distribusi argmax dan rata-rata `p_netral` di ringkasan adalah
gambaran keluaran model, bukan akurasi. Test set SmSA hanya boleh dipakai sekali setelah model final dipastikan.
