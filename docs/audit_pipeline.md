# Audit Pipeline Suspicion-Score (HMM vs. Statistik Statis Sentimen)

**Lingkup file yang dibaca penuh sebelum audit ini ditulis:**
`src/config.py`, `src/preprocessing.py`, `src/sentiment.py`, `src/split.py`,
`docs/pipeline.md`, `notebooks/pipeline_llm_hmm_xgb.ipynb`,
`notebooks/clean-process-data.ipynb`, `src/smoke_test/smoke_test_split.ipynb`,
`Lagi/src/debug.ipynb`, struktur `data/raw`, dan hasil grep repo-wide untuk
`hmmlearn|xgboost|TfidfVectorizer|cosine_similarity|spearman|lexicon` (nol hasil).

**Temuan paling penting duluan:** sebagian besar arsitektur di dokumen desain
(`docs/pipeline.md`) **belum punya implementasi kode sama sekali**. Kode yang
ada baru mencakup tahap 1 (preprocessing), fine-tuning IndoBERT (bukan
inference/skoring), dan product-level splitter generik. Tidak ada TF-IDF,
leksikon, HMM, feature union, model ablasi A/B/C, atau evaluator ranking di
manapun di repo. Karena itu, poin A–I di brief sebagian besar **tidak bisa
diverifikasi dari kode aktual** — ditandai sebagai "belum diimplementasikan",
bukan "lolos" atau "bug", karena kode yang bisa diberi bug belum ada.

---

## 1. Sesuai desain

| Bagian | Lokasi | Catatan |
|---|---|---|
| Preprocessing → cleaned_reviews | `src/preprocessing.py:319-412` (`clean_chunk`), `:416-935` (`DataPreprocessor`) | Membaca CSV per produk (nama file = `product_id`), mendeteksi kolom review/timestamp secara heuristik, membersihkan teks, memfilter panjang minimum, menyimpan ke parquet per produk dengan checkpoint (`_meta.json`, `.done` marker). Sesuai dengan tahap 1 di diagram (`docs/pipeline.md:12-17`). |
| Split di level produk (bukan level ulasan) | `src/split.py:63-113` | `ProductSplitter.split()` mengambil daftar `product_id` unik (`:71-76`), mengacak dengan `np.random.default_rng(random_state)` (`:81-83`), lalu mem-partisi *daftar produk* menjadi train/val/test (`:85-87, 162-208`) sebelum memfilter baris (`:89-105`). Ini secara struktural benar: tidak ada baris ulasan yang jadi unit split. |
| Guard anti product-overlap | `src/split.py:264-297` (`_validate_no_leakage`) | Mengecek irisan set `product_id` antar split dan raise `RuntimeError` bila ada overlap. Ini adalah pengaman yang baik untuk poin **A** (di level produk, bukan di level fit-parameter). |
| Reproducibility parsial | `src/split.py:27,54-58,81`; `src/sentiment.py:110` (`seed=self.config.random_seed`); `src/config.py:20,277` | `random_state`/`random_seed` dipropagasi ke splitter, `np.random.default_rng`, dan `TrainingArguments.seed`. Konsisten sejauh yang ada. |
| Skor kontinu tidak dipaksa jadi biner di kode yang ada | seluruh repo | Tidak ditemukan `argmax`/thresholding pada suspicion_score karena suspicion_score sendiri belum diimplementasikan — tapi juga tidak ada kode yang secara prematur membuatnya biner. |

---

## 2. Deviasi / bug pada kode yang sudah ada

Catatan: karena HMM/TF-IDF/leksikon/model ablasi belum ada, poin C, D, F, G
tidak bisa dicek pada kode nyata (lihat bagian 3). Bagian ini fokus pada bug
di kode yang **sudah** ditulis (preprocessing, sentiment, split).

### 2.1 `DataPreprocessor.clean_all` memotong pipeline setelah file pertama — **BUG KRITIS, bukan soal leakage tapi soal korektnes dasar**
- **Lokasi:** `src/preprocessing.py:788-826`
- **Masalah:** Di dalam loop `for filepath in tqdm(files, ...)`, `return manifest` (`:826`) dipanggil **di dalam iterasi pertama**, bukan setelah loop selesai. Akibatnya `clean_all()` hanya pernah memproses **file CSV pertama** lalu langsung keluar dari method — semua produk lain di `data/raw/` tidak pernah dibersihkan meskipun terdaftar oleh `discover_files()`.
- Selain itu di dalam loop yang sama ada pemanggilan `save_manifest` dua kali dengan argumen yang salah tipe: `self.save_manifest(results)` (list akumulatif, benar) lalu langsung ditimpa oleh `self.save_manifest(result)` (`:816-820`) — `result` di sini adalah **satu dict** dari `clean_single_file`, bukan `results` (list). `pd.DataFrame(single_dict)` akan raise `ValueError: If using all scalar values, you must pass an index` atau setidaknya menghasilkan manifest yang salah bentuk.
- **Kenapa ini penting untuk brief:** Bagian I (kualitas umum) — ini bukan soal leakage, tapi ini berarti **cleaned_reviews di root arsitektur (langkah 1) kemungkinan besar tidak lengkap** kalau `clean_all()` benar-benar dijalankan sebagaimana adanya. Semua tahap berikutnya (IndoBERT, TF-IDF, HMM, split) akan mewarisi dataset yang timpang ini.
- **Perbaikan konkret:**
```python
def clean_all(self) -> pd.DataFrame:
    files = self.discover_files()
    results = []

    for filepath in tqdm(files, desc="cleaning Files"):
        result = self.clean_single_file(filepath)
        results.append(result)

    manifest = self.save_manifest(results)
    self.print_summary(manifest)
    return manifest
```

### 2.2 `inspect_cleaned_data` mereferensikan variabel yang tidak pernah didefinisikan
- **Lokasi:** `src/preprocessing.py:938-990`
- **Masalah:** `indices` dihitung dari `rng.choice(...)` (`:968-970`) tapi loop di `:974` memakai `selected_files` — nama yang tidak pernah didefinisikan di fungsi ini. Fungsi ini akan selalu `NameError` saat dipanggil.
- **Perbaikan:**
```python
selected_files = [parquet_files[i] for i in indices]
for filepath in selected_files:
    ...
```
- Fungsi ini juga dideklarasikan mengembalikan `tuple[dict, pd.DataFrame]` tapi hanya `return sample` (satu nilai) — signature dan return tidak konsisten; QA report sebaiknya juga mengembalikan dict statistik seperti dijanjikan tipe.

### 2.3 `clean_single_file`: metadata & return terjadi *di dalam* loop chunk, bukan setelah semua chunk diproses
- **Lokasi:** `src/preprocessing.py:589-722`
- **Masalah:** Blok `metadata = {...}`, `json.dump(...)`, `done_marker.touch()`, dan `return {...}` semuanya ada di dalam badan `for chunk in reader:` dengan indentasi yang sama seperti `cleaned.to_parquet(...)`. Artinya untuk file apa pun yang lebih besar dari `chunk_size` (50.000 baris, `config.py:33`), method ini akan **berhenti setelah chunk pertama** — `rows_input`/`rows_output` yang dilaporkan hanya mencakup chunk pertama, dan sisa file tidak pernah dibaca/ditulis ke parquet.
- **Kenapa ini penting:** Sama seperti 2.1 — ini bug integritas data di tahap paling awal pipeline (langkah 1: cleaned_reviews), bukan soal leakage HMM, tapi bisa membuat cleaned_reviews per produk terpotong secara diam-diam untuk produk dengan ulasan banyak.
- **Perbaikan:** Pindahkan blok metadata/return ke luar `for chunk in reader:` (setelah loop selesai), akumulasikan `part_number` dan hitung `rows_removed` sekali di akhir.

### 2.4 Import mati/tidak dipakai & indikasi kode belum dibersihkan
- **Lokasi:** `src/preprocessing.py:1-5` (`from numpy import sort`, `from posixpath import sep`, `from numpy import average`, `from typing_extensions import runtime` — tidak satupun dipakai); `src/split.py:2` (`from typing_extensions import runtime` tidak dipakai); `src/sentiment.py:2` (`from os import stat` tidak dipakai).
- **Masalah:** Bukan bug fungsional, tapi ini pertanda kredibel bahwa file-file ini belum melalui linting/review menyeluruh — menambah alasan untuk tidak mengasumsikan bagian lain (2.1–2.3) "pasti sudah dites."
- **Perbaikan:** hapus import yang tidak dipakai; jalankan `ruff`/`flake8` sebagai gate sebelum lanjut ke modul HMM/TF-IDF.

### 2.5 `ExperimentConfig` — duplikasi field yang saling menimpa
- **Lokasi:** `src/config.py:24-26` dan `:259-261` (dua definisi `sentiment_base_model`, sama persis); `src/config.py:36` dan implicit re-declare `overwrite` di dua tempat (`:22`, `:36`); `sentiment_max_length` didefinisikan di `:55` (`=512`) lalu ditimpa lagi di `:263` (`=256`); `sentiment_batch_size` (`:54`) tidak pernah dipakai — yang dipakai `sentiment_train_batch_size`/`sentiment_eval_batch_size` (`:265,267`).
- **Masalah:** Ini bukan soal leakage (poin A–I), tapi risiko nyata: nilai final `sentiment_max_length` yang efektif adalah **256**, bukan 512 seperti yang mungkin diasumsikan pembaca yang membaca baris 55 duluan. Karena `@dataclass` mengevaluasi field berurutan, definisi kedua menang. Untuk reproducibility (poin I) ini rawan salah baca saat orang lain (atau kamu 6 bulan lagi) mengaudit config.
- **Perbaikan:** Gabungkan jadi satu blok field per topik, hapus duplikat, tambahkan test kecil yang mengecek `ExperimentConfig()` bisa diinstansiasi dan mencetak field-field kunci (max_length, batch size, seed) supaya regresi konfigurasi terdeteksi otomatis.

### 2.6 `IndoBERTSentimentTrainer` bukan alur "inference IndoBERT → p_neg/p_neutral/p_pos" yang dibutuhkan Jalur A
- **Lokasi:** `src/sentiment.py:39-243`
- **Masalah:** Kelas ini adalah **fine-tuning trainer** (butuh `label_id` berlabel, dilatih dengan `Trainer.train()`) untuk tugas SMSA 3-kelas, bukan modul untuk menjalankan **inference** IndoBERT pada `cleaned_reviews` guna menghasilkan probabilitas `p_negative/p_neutral/p_positive` per ulasan seperti didefinisikan di langkah 2A arsitektur (`docs/pipeline.md:22-24`). Tidak ada method `predict_proba`/`infer` yang mengonsumsi teks tak berlabel dan mengembalikan tiga probabilitas per baris.
- **Kenapa penting:** Ini bukan bug per se (trainer memang dibutuhkan untuk *punya* model IndoBERT sentimen), tapi ini menandakan **komponen inference (2A) belum ada**, jadi seluruh downstream (mean/std, observasi HMM) belum bisa dibangun dari kode saat ini. Lihat bagian 3.
- **Rekomendasi:** Tambahkan fungsi terpisah, mis. `src/inference.py::predict_sentiment_probs(df, model_dir) -> pd.DataFrame` yang mengembalikan kolom `p_negative, p_neutral, p_positive` per `review_id`, memakai model yang sudah di-`save()` (`sentiment.py:218-242`), tanpa butuh label.

### 2.7 `ProductSplitter` tidak mendukung *stratifikasi kuantil suspicion_score* yang diwajibkan poin 4/arsitektur
- **Lokasi:** `src/split.py:18-113`
- **Masalah:** Constructor tidak menerima parameter target/skor untuk stratifikasi (`:19-58`), dan `split()` (`:63-113`) hanya melakukan **random permutation murni** atas daftar produk (`np.random.default_rng(...).permutation(products)`, `:81-83`) lalu memotongnya secara proporsional (`_split_products`, `:162-208`). Tidak ada langkah yang membagi produk ke bin/kuantil `suspicion_score` sebelum alokasi ke train/val/test.
- **Kenapa ini masalah:** Arsitektur eksplisit meminta "stratifikasi kuantil suspicion_score" (poin 4 & `docs/pipeline.md:39-41`) — ini penting supaya distribusi skor kecurigaan di train dan test sebanding (menghindari test set yang kebetulan didominasi produk skor ekstrem/rendah, yang akan membuat metrik ranking seperti Spearman/precision@k tidak stabil / bias). Implementasi saat ini **tidak memenuhi requirement ini sama sekali** — ini gap desain, bukan cuma "belum lengkap".
- **Perbaikan konkret** (menambahkan stratifikasi kuantil tanpa mengubah kontrak split level-produk):
```python
def split(self, df: pd.DataFrame, suspicion_score_col: str = "suspicion_score", n_strata: int = 5) -> SplitResult:
    self._validate_dataframe(df)
    data = df.copy()

    # satu skor per produk (agregasi harus sudah dilakukan sebelum split)
    product_scores = (
        data.groupby(self.product_col)[suspicion_score_col]
        .first()  # asumsikan sudah agregat level-produk
    )

    # bin kuantil -> tiap stratum displit proporsional train/val/test
    strata = pd.qcut(product_scores, q=n_strata, duplicates="drop")

    rng = np.random.default_rng(self.random_state)
    train_products, val_products, test_products = [], [], []

    for _, group in product_scores.groupby(strata, observed=True):
        products_in_stratum = rng.permutation(group.index.to_numpy())
        tr, va, te = self._split_products(products_in_stratum)
        train_products.extend(tr); val_products.extend(va); test_products.extend(te)

    ...  # sisanya sama, filter data berdasarkan tiga daftar product_id ini
```
Catatan: fungsi ini butuh `suspicion_score` **sudah** dihitung & diagregasi ke level produk sebelum dipanggil — sesuai urutan di arsitektur (Jalur B selesai dulu → baru split). Perlu penanganan kasus stratum terlalu kecil (`min_reviews_per_product`/jumlah produk per bin < 3) agar tidak raise di `_split_products` (`:182-195`).

---

## 3. Belum diimplementasikan sama sekali

Berdasarkan grep menyeluruh (`hmmlearn`, `GaussianHMM`, `xgboost`, `XGBRegressor`,
`TfidfVectorizer`, `cosine_similarity`, `spearman`, `precision_at_k`, `lexicon`,
`leksikon` — nol match di seluruh repo) dan pembacaan struktur folder,
**seluruh bagian berikut dari arsitektur belum punya implementasi kode**:

1. **Jalur B lengkap (langkah 2B–3 di diagram):** TF-IDF vectorizer,
   leksikon genuine/counterfeit, cosine similarity, formula
   `suspicion_score = sim_counterfeit − sim_genuine`, agregasi ke level
   produk. Folder `src/features/` yang disebut di `docs/pipeline.md:129-132`
   (`tfidf.py`, `lexicon.py`) **tidak ada di disk** (`find ./src/features`
   kosong).
2. **Inference IndoBERT (langkah 2A) yang menghasilkan p_neg/p_neutral/p_pos
   per ulasan tak berlabel** — yang ada hanya trainer fine-tuning
   (`src/sentiment.py`), lihat 2.6.
3. **Sort per produk berdasarkan waktu** sebagai langkah eksplisit sebelum
   masuk ke mean/std dan HMM — tidak ada kode yang melakukan
   `sort_values(["product_id","timestamp"])` pada output sentimen di mana pun.
4. **Modul HMM** (`src/temporal/hmm.py` di desain) — tidak ada. Karena itu
   poin **C** (mekanisme `lengths` multi-sequence hmmlearn) dan **D**
   (representasi observasi + `covariance_type`) **tidak bisa dicek sama
   sekali** — tidak ada baris kode yang bisa diperiksa untuk itu. Ini harus
   ditandai sebagai gap kritis, bukan "lolos": risiko yang dijelaskan di
   poin C dan D brief kamu (leakage transisi antar-produk lewat sekuens yang
   disambung tanpa `lengths`; instabilitas Gaussian `covariance_type="full"`
   pada data simpleks) **akan langsung relevan begitu modul ini ditulis**,
   jadi harus jadi syarat desain sejak baris kode pertama modul ini dibuat,
   bukan diperbaiki belakangan.
   - `config.py:63` sudah menetapkan `hmm_covariance_type: str = "diag"` —
     ini pilihan yang tepat untuk data compositional dibanding `"full"`
     (mengurangi risiko singular, meski tidak menyelesaikan constraint
     sum-to-one). Tapi karena belum ada kode yang memakainya, belum bisa
     dipastikan constraint ini benar-benar diteruskan ke konstruktor
     `hmmlearn.GaussianHMM(...)`.
5. **Mean+std sentimen per produk** sebagai fitur statis eksplisit — belum
   ada fungsi agregasi ini di kode manapun.
6. **Feature union akhir** (mean+std ⊕ loglik+state proportions) — belum ada.
7. **Tiga model ablasi A/B/C + XGBoost regressor** (`src/models/model_a.py`,
   `model_b.py`, `model_c.py` di desain) — tidak ada file-nya, `xgboost`
   tidak muncul di grep manapun juga tidak ada di
   requirements/environment manapun yang saya temukan.
8. **Evaluator ranking** (Spearman, precision@k) — `src/evaluation/` tidak
   ada, tidak ada import `scipy.stats.spearmanr` di manapun.
9. **Kalibrasi (temperature scaling)** — sesuai instruksi brief poin E, ini
   cukup dicatat sebagai *open item*, bukan bug, karena belum ada baris kode
   sentimen-inference untuk diperiksa.
10. **`export.py` dan `pipeline/experiment.py`** (orkestrator end-to-end) —
    tidak ada; tidak ada satu skrip/notebook pun yang menjalankan pipeline
    dari raw CSV sampai prediksi skor. `notebooks/pipeline_llm_hmm_xgb.ipynb`
    namanya menjanjikan ini tapi isinya adalah eksperimen berbeda (mencoba
    beberapa LLM generatif — Qwen/Gemma/Phi — untuk menghasilkan vektor
    probabilitas 4-kelas dari next-token logits; ini bukan bagian dari
    arsitektur yang dideskripsikan di brief maupun `docs/pipeline.md`, dan
    tidak memakai IndoBERT/HMM/TF-IDF/XGBoost sama sekali).
11. **Error handling untuk produk dengan ulasan sangat sedikit (poin I)**
    — `config.py:72` punya `min_reviews_per_product: int = 3`, tapi field
    ini **tidak dipakai/direferensikan di kode manapun** (`grep
    min_reviews_per_product` hanya match di config.py dan `validate()`-nya
    sendiri; tidak dipakai untuk memfilter produk di preprocessing atau
    split). Jadi belum ada guard nyata terhadap sekuens HMM yang terlalu
    pendek.

---

## 4. Pertanyaan buat kamu

1. **Urutan tulis-ulang `notebooks/pipeline_llm_hmm_xgb.ipynb`:** notebook ini
   isinya eksperimen LLM generatif (Qwen2.5/Gemma/Phi-3) untuk skor 4-kelas
   dari next-token logits, sama sekali di luar arsitektur yang dideskripsikan
   (IndoBERT fine-tuned 3-kelas → HMM → XGBoost). Apakah ini eksperimen lama
   yang sudah ditinggalkan (boleh saya abaikan sepenuhnya dari audit
   berikutnya), atau masih ada rencana memakai salah satu LLM ini sebagai
   alternatif IndoBERT untuk Jalur A?
2. **File data yang tampaknya sudah punya label biner buatan-tangan**
   (`data/raw/model data.csv`, `scraping_data_balanced.csv` — dari
   `src/smoke_test/smoke_test_split.ipynb` & `Lagi` sumber lama, memberi
   `label = 1` untuk toko `['KKA', 'Maja']` berdasarkan nama toko, bukan dari
   TF-IDF+leksikon). Apakah file-file berlabel manual ini **hanya** untuk
   smoke-test/sanity-check splitter (bukan untuk suspicion_score final), atau
   ada risiko label heuristik toko ini nanti "bocor" jadi pengganti sementara
   `suspicion_score` asli sebelum modul TF-IDF+leksikon selesai ditulis? Ini
   penting untuk poin **B** (sirkularitas label vs fitur) begitu Jalur B
   mulai diimplementasikan — supaya label heuristik lama ini tidak
   tercampur/tertinggal di pipeline produksi.
3. **Level agregasi suspicion_score ke produk** (mean vs. max review paling
   mencurigakan) — `docs/pipeline.md:112` mencatat ini sebagai keputusan
   terbuka. Sudahkah ini diputuskan? Ini menentukan bentuk konkret fungsi
   agregasi Jalur B yang belum ada di kode, dan juga memengaruhi cara
   stratifikasi kuantil di split (poin 2.7) dikonstruksi.
4. **Field `min_reviews_per_product` (`config.py:72`) tidak dipakai** — apakah
   memang belum sempat diwireing, atau sengaja didefinisikan lebih dulu
   sebagai "kontrak" untuk modul filter yang akan datang? Saya perlu tahu ini
   supaya tidak salah menandainya sebagai bug vs. work-in-progress yang
   disengaja.
5. **hmmlearn vs. implementasi HMM custom** — brief menyebut `hmmlearn`
   secara eksplisit sebagai contoh pustaka. Apakah ini pustaka yang memang
   akan dipakai (sehingga saya bisa memvalidasi pemakaian parameter `lengths`
   begitu modul `src/temporal/hmm.py` ditulis), atau kamu mempertimbangkan
   implementasi HMM sendiri / pustaka lain (`pomegranate`, dll.) yang
   mekanisme multi-sequence-nya berbeda?
