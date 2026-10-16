"""Tes logika inferensi sentimen RM 1 TANPA torch dan TANPA model asli.

Yang diuji: validasi revision, pemetaan label dari id2label, pemeriksaan tokenizer, pemilihan device,
batching/urutan/NaN/pemotongan, ringkasan, dan CLI end-to-end dengan model palsu.
Yang TIDAK diuji di sini: unduhan Hugging Face, pemuatan bobot, forward torch/MPS, kualitas prediksi.
(Lihat tests/test_inference_smoke.py dan docs/inferensi_sentimen.md untuk uji dengan model nyata.)
"""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src import sentiment_inference as si
from src import run_sentiment_inference as run_mod


# ------------------------------------------------------------
# Model & tokenizer palsu (deterministik, tidak sensitif terhadap padding)
# ------------------------------------------------------------

class FakeTokenizer:
    def __init__(self, model_max_length=128):
        self.model_max_length = model_max_length
        self.seen_lengths = []
        self.batch_calls = []

    @staticmethod
    def _ids(text):
        return [1] + [3 + (sum(map(ord, w)) % 500) for w in text.split()] + [2]

    def __call__(self, texts, truncation=False, padding=False, max_length=None,
                 return_tensors=None, add_special_tokens=True, verbose=True):
        rows = []
        for text in texts:
            ids = self._ids(text)
            if truncation and max_length and len(ids) > max_length:
                ids = ids[: max_length - 1] + [2]
            rows.append(ids)
        if return_tensors != "np":
            return {"input_ids": rows}
        self.batch_calls.append((truncation, padding, max_length, [len(row) for row in rows]))
        width = max(len(r) for r in rows)
        input_ids = np.zeros((len(rows), width), dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = r
            mask[i, : len(r)] = 1
        self.seen_lengths.append(width)
        return {"input_ids": input_ids, "attention_mask": mask}


def make_bundle(indices=None, tokenizer=None, batch_size=4, bad=None):
    rng = np.random.default_rng(0)
    table = rng.normal(size=(600, 3))
    indices = indices or {"negative": 0, "neutral": 1, "positive": 2}
    tokenizer = tokenizer or FakeTokenizer()

    def forward(batch):
        ids, mask = batch["input_ids"], batch["attention_mask"]
        emb = table[ids] * mask[..., None]
        logits = emb.sum(axis=1) / mask.sum(axis=1, keepdims=True) * 4.0
        if bad == "nan":
            logits[0, 0] = np.nan
        if bad == "width":
            logits = logits[:, :2]
        return logits.astype(np.float32)

    return si.SentimentBundle(
        tokenizer=tokenizer, forward=forward, sentiment_indices=indices,
        device="cpu", max_length=si.MAX_LENGTH, batch_size=batch_size,
        meta={"id2label": {0: "negatif", 1: "netral", 2: "positif"}, "device": "cpu"},
    ), table


def make_semantic_bundle(swapped=False, neutral_fails=False, tokenizer=None):
    """Tiga urutan ID tetap untuk tes; tidak memakai bobot model nyata."""
    bundle, _ = make_bundle(tokenizer=tokenizer)
    base_forward = bundle.forward
    # Fixture independen dari konstanta kode agar perubahan kalimat terdeteksi.
    fixtures = (
        (2, "Barangnya bagus banget, sesuai deskripsi, pengiriman cepat"),
        (1, "Barang sudah sampai, belum saya coba"),
        (0, "Barang rusak, tidak sesuai foto, kecewa berat"),
    )
    expected = {tuple(FakeTokenizer._ids(text)): label for label, text in fixtures}

    def forward(batch):
        logits = base_forward(batch)
        for i, (ids, mask) in enumerate(zip(batch["input_ids"], batch["attention_mask"])):
            key = tuple(ids[mask.astype(bool)])
            if key in expected:
                target = expected[key]
                if neutral_fails and target == 1:
                    target = 2
                logits[i] = -2.0
                logits[i, target] = 2.0
        return logits[:, [2, 1, 0]] if swapped else logits

    bundle.forward = forward
    return bundle


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


TEXTS = [
    "barangnya bagus banget sesuai foto",
    "kecewa barang rusak tidak sesuai deskripsi",
    "barang sampai belum dicoba",
    "mantap pengiriman cepat",
    "layar tidak menyala sejak dibuka produk cacat",
    "paket diterima pagi ini",
]


class RevisionTests(unittest.TestCase):
    def test_full_sha_ok(self):
        sha = "a" * 40
        self.assertEqual(si.validate_revision(sha), sha)

    def test_rejects_missing_short_and_uppercase(self):
        for bad in (None, "", "5366aae", "A" * 40, "g" * 40, "a" * 41):
            with self.assertRaises(ValueError, msg=repr(bad)):
                si.validate_revision(bad)


class LabelMappingTests(unittest.TestCase):
    def test_expected_indonesian_and_english(self):
        for labels in ({0: "negatif", 1: "netral", 2: "positif"},
                       {0: "negative", 1: "neutral", 2: "positive"},
                       {"0": "NEGATIF", "1": "Netral", "2": "Positif"}):
            self.assertEqual(si.check_label_mapping(labels), {"negative": 0, "neutral": 1, "positive": 2})

    def test_permuted_order_rejected_by_default_but_mapped_when_relaxed(self):
        permuted = {0: "positive", 1: "negative", 2: "neutral"}
        with self.assertRaises(si.ModelCheckError):
            si.check_label_mapping(permuted)
        self.assertEqual(
            si.check_label_mapping(permuted, require_expected_order=False),
            {"positive": 0, "negative": 1, "neutral": 2},
        )

    def test_rejects_wrong_count_unknown_and_duplicates(self):
        for bad in ({0: "negatif", 1: "positif"},
                    {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
                    {0: "negatif", 1: "negatif", 2: "positif"},
                    {0: "negatif", 1: "netral", 2: "positif", 3: "lain"}):
            with self.assertRaises(si.ModelCheckError, msg=str(bad)):
                si.check_label_mapping(bad)


class TokenizerAndDeviceTests(unittest.TestCase):
    def test_default_length_matches_config_and_training(self):
        from src.config import ExperimentConfig

        self.assertEqual(ExperimentConfig().sentiment_max_length, 128)
        self.assertEqual(si.MAX_LENGTH, ExperimentConfig().sentiment_max_length)
        self.assertEqual(si.MAX_LENGTH, si.MAX_LENGTH_PELATIHAN)
        self.assertEqual(si.SentimentModelConfig().max_length, si.MAX_LENGTH)
        self.assertEqual(run_mod.build_parser().parse_args([]).max_length, si.MAX_LENGTH)

    def test_training_length_requires_override_and_warns(self):
        with self.assertLogs(si.LOGGER, level="WARNING") as logs:
            with self.assertRaisesRegex(si.ModelCheckError, "--allow-length-mismatch"):
                si.check_training_length(256)
        self.assertIn("PERINGATAN", logs.output[0])
        with self.assertLogs(si.LOGGER, level="WARNING"):
            meta = si.check_training_length(256, allow_length_mismatch=True)
        self.assertTrue(meta["length_mismatch"])
        self.assertTrue(meta["allow_length_mismatch"])
        self.assertEqual(meta["max_length_pelatihan"], 128)
        self.assertIn("256", meta["length_mismatch_warning"])
        self.assertFalse(si.check_training_length(128)["length_mismatch"])

    def test_load_rejects_mismatch_before_loading_weights(self):
        fake_transformers = SimpleNamespace(
            AutoConfig=Mock(), AutoTokenizer=Mock(), AutoModelForSequenceClassification=Mock())
        fake_transformers.AutoConfig.from_pretrained.return_value = SimpleNamespace(max_position_embeddings=512)
        with patch.dict("sys.modules", {"torch": Mock(), "transformers": fake_transformers}), \
                patch.object(si, "resolve_model_path", return_value=("fake", None)):
            with self.assertRaisesRegex(si.ModelCheckError, "--allow-length-mismatch"):
                si.load_sentiment_model(si.SentimentModelConfig(max_length=256))
        fake_transformers.AutoModelForSequenceClassification.from_pretrained.assert_not_called()

    def test_tokenizer_check(self):
        for length in (128, 512, int(1e30)):
            with self.assertLogs(si.LOGGER, level="INFO"):
                self.assertTrue(si.check_tokenizer(FakeTokenizer(length), 512))

    def test_capacity_limit_rejects_invalid_length(self):
        self.assertEqual(si.SentimentModelConfig().max_length, 128)
        si.check_max_length(512, 512)
        for length in (513, 0, -1, True):
            with self.assertRaises(si.ModelCheckError):
                si.check_max_length(length, 512)
        with self.assertRaisesRegex(si.ModelCheckError, "max_position_embeddings"):
            si.check_max_length(128, None)

    def test_load_rejects_capacity_before_loading_weights(self):
        fake_transformers = SimpleNamespace(
            AutoConfig=Mock(), AutoTokenizer=Mock(), AutoModelForSequenceClassification=Mock())
        fake_transformers.AutoConfig.from_pretrained.return_value = SimpleNamespace(max_position_embeddings=512)
        with patch.dict("sys.modules", {"torch": Mock(), "transformers": fake_transformers}), \
                patch.object(si, "resolve_model_path", return_value=("fake", None)):
            with self.assertRaisesRegex(si.ModelCheckError, "513.*max_position_embeddings=512"):
                si.load_sentiment_model(si.SentimentModelConfig(max_length=513))
        fake_transformers.AutoModelForSequenceClassification.from_pretrained.assert_not_called()

    def test_pick_device(self):
        self.assertEqual(si.pick_device("auto", True, True), "cuda")
        self.assertEqual(si.pick_device("auto", False, True), "mps")
        self.assertEqual(si.pick_device("cpu", False, False), "cpu")
        with self.assertRaises(RuntimeError):      # tidak boleh fallback diam-diam ke CPU
            si.pick_device("auto", False, False)
        with self.assertRaises(RuntimeError):
            si.pick_device("mps", True, False)
        with self.assertRaises(RuntimeError):
            si.pick_device("cuda", False, True)
        with self.assertRaises(ValueError):
            si.pick_device("tpu", True, True)


class PredictTests(unittest.TestCase):
    def test_manual_bundle_cannot_bypass_length_guard(self):
        bundle, _ = make_bundle()
        bundle.max_length = 256
        with patch.object(bundle, "forward", wraps=bundle.forward) as forward:
            with self.assertRaisesRegex(si.ModelCheckError, "--allow-length-mismatch"):
                si.predict_sentiment_probs(TEXTS, bundle=bundle)
            forward.assert_not_called()
            bundle.allow_length_mismatch = True
            si.predict_sentiment_probs(TEXTS, bundle=bundle)
            forward.assert_called()
        self.assertTrue(bundle.meta["length_mismatch"])
        self.assertTrue(bundle.meta["allow_length_mismatch"])

    def test_shape_sum_finite_and_matches_manual_softmax(self):
        bundle, table = make_bundle()
        out = si.predict_sentiment_probs(TEXTS, bundle=bundle)
        self.assertEqual(list(out.columns), list(si.PROB_COLUMNS))
        self.assertEqual(len(out), len(TEXTS))
        self.assertTrue(np.isfinite(out.to_numpy()).all())
        self.assertLess(np.abs(out.sum(axis=1) - 1).max(), 1e-9)
        ids = [FakeTokenizer._ids(t) for t in TEXTS]
        manual = np.stack([table[i].mean(axis=0) * 4.0 for i in ids])
        np.testing.assert_allclose(out.to_numpy(), softmax(manual.astype(np.float32).astype(float)), atol=1e-6)

    def test_column_order_follows_id2label_not_position(self):
        # indeks logit: positif=0, negatif=1, netral=2
        bundle, table = make_bundle(indices={"positive": 0, "negative": 1, "neutral": 2})
        out = si.predict_sentiment_probs(TEXTS, bundle=bundle)
        ids = [FakeTokenizer._ids(t) for t in TEXTS]
        raw = softmax(np.stack([table[i].mean(axis=0) * 4.0 for i in ids]).astype(np.float32).astype(float))
        np.testing.assert_allclose(out["p_positif"], raw[:, 0], atol=1e-6)
        np.testing.assert_allclose(out["p_negatif"], raw[:, 1], atol=1e-6)
        np.testing.assert_allclose(out["p_netral"], raw[:, 2], atol=1e-6)

    def test_batch_equals_single_and_order_independent(self):
        bundle, _ = make_bundle()
        full = si.predict_sentiment_probs(TEXTS, bundle=bundle, batch_size=len(TEXTS))
        single = pd.concat([si.predict_sentiment_probs([t], bundle=bundle, batch_size=1) for t in TEXTS],
                           ignore_index=True)
        np.testing.assert_allclose(full.to_numpy(), single.to_numpy(), atol=1e-6)
        shuffled = si.predict_sentiment_probs(TEXTS[::-1], bundle=bundle, batch_size=2)
        np.testing.assert_allclose(shuffled.to_numpy()[::-1], full.to_numpy(), atol=1e-6)

    def test_empty_and_nan_rows_kept_as_nan_and_index_preserved(self):
        bundle, _ = make_bundle()
        series = pd.Series(["bagus", None, "", "   ", np.nan, "rusak parah"], index=list("abcdef"))
        out = si.predict_sentiment_probs(series, bundle=bundle, return_diagnostics=True)
        self.assertEqual(list(out.index), list("abcdef"))
        self.assertEqual(out["kosong"].tolist(), [False, True, True, True, True, False])
        self.assertTrue(out.loc[["b", "c", "d", "e"], list(si.PROB_COLUMNS)].isna().all().all())
        self.assertTrue(out.loc[["a", "f"], list(si.PROB_COLUMNS)].notna().all().all())

    def test_all_empty_input(self):
        bundle, _ = make_bundle()
        out = si.predict_sentiment_probs(["", None], bundle=bundle, return_diagnostics=True)
        self.assertEqual(len(out), 2)
        self.assertTrue(out["kosong"].all())

    def test_truncation_flag_and_max_length_respected(self):
        tok = FakeTokenizer()
        bundle, _ = make_bundle(tokenizer=tok)
        texts = ["kata " * count for count in (600, 127, 510, 511, 126)]
        out = si.predict_sentiment_probs(texts, bundle=bundle, batch_size=2, return_diagnostics=True)
        self.assertEqual(out["n_token"].tolist(), [602, 129, 512, 513, 128])
        self.assertEqual(out["terpotong"].tolist(), [True, True, True, True, False])
        self.assertEqual(tok.seen_lengths, [128, 128, 128])  # sort panjang + padding dinamis
        self.assertTrue(all(call[:3] == (True, True, 128) for call in tok.batch_calls))
        summary = si.summarize_predictions(pd.Series(texts), out, max_length=bundle.max_length)
        self.assertEqual(summary["n_terpotong"], 4)
        self.assertEqual(summary["persen_terpotong"], 80.0)
        self.assertTrue(out[list(si.PROB_COLUMNS)].notna().all().all())

    def test_logits_returned_in_same_order_as_probs(self):
        bundle, _ = make_bundle(indices={"positive": 0, "negative": 1, "neutral": 2})
        out = si.predict_sentiment_probs(TEXTS, bundle=bundle, return_logits=True)
        probs = softmax(out[list(si.LOGIT_COLUMNS)].to_numpy())
        np.testing.assert_allclose(probs, out[list(si.PROB_COLUMNS)].to_numpy(), atol=1e-9)

    def test_non_finite_logits_and_wrong_width_raise(self):
        with self.assertRaises(FloatingPointError):
            si.predict_sentiment_probs(TEXTS, bundle=make_bundle(bad="nan")[0])
        with self.assertRaises(si.ModelCheckError):
            si.predict_sentiment_probs(TEXTS, bundle=make_bundle(bad="width")[0])

    def test_batch_size_must_be_positive(self):
        bundle, _ = make_bundle()
        with self.assertRaises(ValueError):
            si.predict_sentiment_probs(TEXTS, bundle=bundle, batch_size=0)

    def test_run_model_checks_tokenizer_limit_is_information(self):
        for length in (128, int(1e30)):
            checks = si.run_model_checks(make_semantic_bundle(tokenizer=FakeTokenizer(length)), TEXTS)
            self.assertTrue(all(c["lulus"] for c in checks), checks)


class LabelSemanticsTests(unittest.TestCase):
    def test_expected_mapping_passes(self):
        checks = si.check_label_semantics(make_semantic_bundle())
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(check["lulus"] for check in checks), checks)
        for check in checks:
            self.assertEqual(set(check["probabilitas"]), set(si.PROB_COLUMNS))
            for column in si.PROB_COLUMNS:
                self.assertRegex(check["detail"], column + r"=\d\.\d{4}(?:;|$)")

    def test_swapped_positive_negative_is_detected(self):
        checks = si.check_label_semantics(make_semantic_bundle(swapped=True))
        self.assertEqual([check["lulus"] for check in checks], [False, True, False])

    def test_neutral_failure_is_reported_without_relabeling(self):
        checks = si.check_label_semantics(make_semantic_bundle(neutral_fails=True))
        self.assertEqual([check["lulus"] for check in checks], [True, False, True])
        self.assertEqual(checks[1]["harapan"], "netral")
        self.assertEqual(checks[1]["argmax"], "positif")


class SummaryTests(unittest.TestCase):
    def test_report_uses_actual_lengths_and_truncation_counts(self):
        bundle, _ = make_bundle()
        texts = pd.Series(["bagus", "kata " * 200])
        probs = si.predict_sentiment_probs(texts, bundle=bundle, return_diagnostics=True)
        summary = si.summarize_predictions(texts, probs, max_length=bundle.max_length)
        meta = {**bundle.meta, "max_position_embeddings": 512}
        report = si.format_summary_md(summary, meta)
        self.assertIn("Fine-tuning dan inferensi memakai max_length 128 (kapasitas model 512).", report)
        self.assertIn("Ulasan > 128 token dipotong (1 ulasan, 50,00%).", report)
        self.assertNotIn("inferensi memakai 512 sesuai proposal", report)

        # Meta 256 harus menghasilkan narasi 256, bukan angka bawaan modul.
        summary = {**summary, "max_length": 256, "n_terpotong": 0, "persen_terpotong": 0.0}
        meta = {**meta, "max_length": 256}
        report = si.format_summary_md(summary, meta)
        self.assertIn("PERINGATAN: Fine-tuning memakai max_length 128; inferensi memakai max_length 256", report)
        self.assertIn("Ulasan > 256 token dipotong (0 ulasan, 0,00%).", report)

    def test_report_rejects_conflicting_metadata(self):
        with self.assertRaisesRegex(ValueError, "meta dan ringkasan berbeda"):
            si.format_summary_md({"max_length": 128}, {"max_length": 256})

    def test_summary_counts(self):
        bundle, _ = make_bundle()
        long_text = " ".join(f"k{i}" for i in range(600))
        texts = pd.Series(["bagus", "", None, "barang bagus sekali", long_text, "ok mantap"])
        probs = si.predict_sentiment_probs(texts, bundle=bundle, return_diagnostics=True)
        summary = si.summarize_predictions(texts, probs, max_length=bundle.max_length, short_words=2)
        self.assertEqual(summary["n_total"], 6)
        self.assertEqual(summary["n_kosong"], 2)
        self.assertEqual(summary["n_diinferensi"], 4)
        self.assertEqual(summary["n_sangat_pendek"], 2)        # "bagus", "ok mantap"
        self.assertEqual(summary["n_terpotong"], 1)
        self.assertEqual(sum(v["n"] for v in summary["distribusi_argmax"].values()), 4)
        self.assertEqual(summary["n_nan_pada_diinferensi"], 0)
        self.assertIn("Keterbatasan", si.format_summary_md(summary, {"device": "cpu"}))


class SafetyTests(unittest.TestCase):
    def test_import_does_not_construct_config_or_access_data(self):
        import subprocess
        import sys
        import textwrap

        # Dependensi dimuat sebelum guard; src.config dan modul inferensi tetap
        # diimpor baru. Pembacaan modul Python terpisah dari I/O data aplikasi.
        script = textwrap.dedent('''
            import json
            import dataclasses
            import numpy
            import pandas
            from contextlib import ExitStack
            from unittest.mock import patch

            guards = []
            real_dataclass = dataclasses.dataclass
            with ExitStack() as stack:
                def guarded_dataclass(cls):
                    cls = real_dataclass(cls)
                    if cls.__module__ == "src.config" and cls.__name__ == "ExperimentConfig":
                        for name in ("__init__", "setup", "validate", "create_directories"):
                            guards.append(stack.enter_context(patch.object(
                                cls, name, side_effect=AssertionError("import memanggil " + name))))
                    return cls
                stack.enter_context(patch("dataclasses.dataclass", side_effect=guarded_dataclass))
                for name in ("builtins.open", "io.open", "os.open", "os.mkdir", "os.makedirs",
                             "pathlib.Path.open", "pathlib.Path.read_text", "pathlib.Path.read_bytes",
                             "pathlib.Path.mkdir", "pandas.read_csv", "pandas.read_parquet", "numpy.load"):
                    guards.append(stack.enter_context(patch(
                        name, side_effect=AssertionError("I/O saat import: " + name))))
                import src.sentiment_inference as imported
                from src.config import ExperimentConfig
                assert imported.MAX_LENGTH == 128
                assert imported.MAX_LENGTH == ExperimentConfig.__dataclass_fields__["sentiment_max_length"].default
            print(json.dumps({"max_length": imported.MAX_LENGTH,
                              "guarded_calls": sum(mock.call_count for mock in guards),
                              "guards": len(guards)}))
        ''')
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[1], check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"max_length": 128, "guarded_calls": 0, "guards": 16})

    def test_training_args_never_downloaded(self):
        self.assertNotIn("training_args.bin", si.DOWNLOAD_PATTERNS)
        self.assertFalse(any(p.endswith((".bin", ".pt", ".pkl", ".pth")) for p in si.DOWNLOAD_PATTERNS))

    def test_no_token_literal_in_sources(self):
        root = Path(si.__file__).parent
        for name in ("sentiment_inference.py", "run_sentiment_inference.py"):
            text = (root / name).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", text), name)
            self.assertNotIn("os.environ[", text.replace("os.environ.get", ""), name)

    def test_model_dir_missing_files_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                si.resolve_model_path(si.SentimentModelConfig(model_dir=tmp))


class RunCliTests(unittest.TestCase):
    def _args(self, tmp, **over):
        argv = ["--input", str(Path(tmp) / "in.csv"), "--output", str(Path(tmp) / "out" / "hasil.csv"),
                "--batch-size", "3", "--sample-checks", "4"]
        for k, v in over.items():
            argv += [k] + ([v] if v is not True else [])
        return run_mod.build_parser().parse_args(argv)

    def _write_input(self, tmp):
        df = pd.DataFrame({
            "review_id": [f"p__{i}" for i in range(8)],
            "product_id": ["p"] * 8,
            "text_proposal": TEXTS + ["", None],
            "timestamp": pd.date_range("2026-01-01", periods=8).astype(str),
        })
        df.to_csv(Path(tmp) / "in.csv", index=False)
        return df

    def test_end_to_end_with_fake_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_df = self._write_input(tmp)
            bundle = make_semantic_bundle()
            code = run_mod.run(self._args(tmp, **{"--save-diagnostics": True}), bundle=bundle)
            self.assertEqual(code, run_mod.EXIT_OK)
            out = pd.read_csv(Path(tmp) / "out" / "hasil.csv", encoding="utf-8-sig")
            self.assertEqual(len(out), len(src_df))
            for col in src_df.columns:
                self.assertIn(col, out.columns)
            for col in si.PROB_COLUMNS + ("label_argmax", "n_token", "terpotong"):
                self.assertIn(col, out.columns)
            self.assertEqual(out["review_id"].tolist(), src_df["review_id"].tolist())   # urutan & id utuh
            ok = out[out["text_proposal"].notna() & (out["text_proposal"] != "")]
            self.assertLess((ok[list(si.PROB_COLUMNS)].sum(axis=1) - 1).abs().max(), 1e-6)
            self.assertTrue(out.loc[6:, list(si.PROB_COLUMNS)].isna().all().all())       # kosong -> NaN, tetap ada
            summary = json.loads((Path(tmp) / "out" / "hasil_ringkasan.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["ringkasan"]["n_total"], 8)
            self.assertEqual(summary["meta"]["max_length_pelatihan"], 128)
            self.assertEqual(summary["meta"]["tokenizer_model_max_length"], 128)
            self.assertEqual(summary["meta"]["max_length"], 128)
            semantics = [c for c in summary["pemeriksaan"] if "kalimat" in c]
            self.assertEqual(len(semantics), 3)
            self.assertTrue((Path(tmp) / "out" / "hasil_ringkasan.md").is_file())

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_input(tmp)
            (Path(tmp) / "out").mkdir()
            existing = Path(tmp) / "out" / "hasil.csv"
            existing.write_text("jangan ditimpa", encoding="utf-8")
            code = run_mod.run(self._args(tmp), bundle=make_bundle()[0])
            self.assertEqual(code, run_mod.EXIT_STOPPED)
            self.assertEqual(existing.read_text(encoding="utf-8"), "jangan ditimpa")

    def test_stops_when_checks_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_input(tmp)
            bad_bundle = make_semantic_bundle(neutral_fails=True)
            code = run_mod.run(self._args(tmp), bundle=bad_bundle)
            self.assertEqual(code, run_mod.EXIT_CHECK_FAILED)
            self.assertFalse((Path(tmp) / "out" / "hasil.csv").exists())
            summary = json.loads((Path(tmp) / "out" / "hasil_ringkasan.json").read_text())
            self.assertIsNone(summary["ringkasan"])
            self.assertEqual(summary["meta"]["status"], "berhenti_pemeriksaan_gagal")
            md = (Path(tmp) / "out" / "hasil_ringkasan.md").read_text()
            for column in si.PROB_COLUMNS:
                self.assertRegex(md, column + r"=\d\.\d{4}")

    def test_continue_preserves_failed_checks_and_custom_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_input(tmp)
            args = self._args(tmp, **{"--continue-on-failed-checks": True, "--max-length": "256",
                                     "--allow-length-mismatch": True})
            self.assertEqual(run_mod.run(args, bundle=make_semantic_bundle(swapped=True)), run_mod.EXIT_OK)
            summary = json.loads((Path(tmp) / "out" / "hasil_ringkasan.json").read_text())
            self.assertEqual(summary["meta"]["max_length"], 256)
            self.assertTrue(summary["meta"]["length_mismatch"])
            self.assertTrue(summary["meta"]["allow_length_mismatch"])
            self.assertIn("PERINGATAN", summary["meta"]["length_mismatch_warning"])
            self.assertFalse(all(c["lulus"] for c in summary["pemeriksaan"]))

    def test_length_mismatch_requires_flag_even_if_checks_are_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_input(tmp)
            bundle = make_semantic_bundle()
            bundle.allow_length_mismatch = True  # Persetujuan CLI tetap diperlukan.
            args = self._args(tmp, **{"--continue-on-failed-checks": True, "--max-length": "256"})
            with patch.object(bundle, "forward", wraps=bundle.forward) as forward, \
                    self.assertLogs(si.LOGGER, level="WARNING") as logs:
                self.assertEqual(run_mod.run(args, bundle=bundle), run_mod.EXIT_CHECK_FAILED)
                forward.assert_not_called()
            self.assertIn("PERINGATAN", logs.output[0])
            self.assertFalse((Path(tmp) / "out").exists())

    def test_capacity_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_input(tmp)
            args = self._args(tmp, **{"--continue-on-failed-checks": True, "--max-length": "513",
                                     "--allow-length-mismatch": True})
            self.assertEqual(run_mod.run(args, bundle=make_semantic_bundle()), run_mod.EXIT_CHECK_FAILED)
            self.assertFalse((Path(tmp) / "out" / "hasil.csv").exists())

    def test_defaults_and_missing_proposal_column(self):
        args = run_mod.build_parser().parse_args([])
        self.assertEqual(args.max_length, 128)
        self.assertEqual(args.text_col, "text_proposal")
        self.assertEqual(Path(args.input).name, "reviews_clean_v2.parquet")
        self.assertEqual(Path(args.output).name, "sentimen_ulasan_v2.csv")
        with tempfile.TemporaryDirectory() as tmp:
            self._write_input(tmp).rename(columns={"text_proposal": "text_light"}).to_csv(
                Path(tmp) / "in.csv", index=False)
            self.assertEqual(run_mod.run(self._args(tmp), bundle=make_semantic_bundle()), run_mod.EXIT_STOPPED)

    def test_stops_on_missing_text_column_and_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run_mod.run(self._args(tmp), bundle=make_bundle()[0]), run_mod.EXIT_STOPPED)
            self._write_input(tmp)
            self.assertEqual(run_mod.run(self._args(tmp, **{"--text-col": "tidak_ada"}), bundle=make_bundle()[0]),
                             run_mod.EXIT_STOPPED)


REAL_MODEL_SENTENCES = [
    "Barangnya bagus banget, sesuai foto, recommended!",
    "Kecewa, barang rusak, tidak sesuai deskripsi.",
    "Barang sampai, belum dicoba.",
    "Kualitasnya mantap, pengiriman cepat dan aman.",
    "Produk cacat, layar tidak menyala sejak dibuka.",
    "Paket diterima oleh saya pagi ini.",
    "Sangat puas, berfungsi dengan baik dan harganya murah.",
    "Buruk sekali, cepat panas dan baterainya boros.",
    "Warna hitam, kapasitas memori 128 GB.",
    "Barangnya palsu, KW, bukan original.",
]


@unittest.skipUnless(
    __import__("os").environ.get("INDOBERT_SENTIMENT_MODEL_DIR"),
    "Set INDOBERT_SENTIMENT_MODEL_DIR ke folder model (config.json, model.safetensors, tokenizer) untuk uji model nyata",
)
class RealModelSmokeTests(unittest.TestCase):
    """Uji model NYATA (butuh torch + transformers + folder model). Tidak dijalankan di CI/sandbox tanpa model."""

    def test_ten_sentences(self):
        import os

        model_dir = os.environ["INDOBERT_SENTIMENT_MODEL_DIR"]
        device = os.environ.get("INDOBERT_SENTIMENT_DEVICE", "cpu")
        bundle = si.load_sentiment_model(si.SentimentModelConfig(model_dir=model_dir, device=device, batch_size=4))
        checks = si.run_model_checks(bundle, REAL_MODEL_SENTENCES)
        self.assertTrue(all(c["lulus"] for c in checks), checks)
        out = si.predict_sentiment_probs(REAL_MODEL_SENTENCES, bundle=bundle, return_diagnostics=True)
        self.assertEqual(len(out), 10)
        self.assertLess((out[list(si.PROB_COLUMNS)].sum(axis=1) - 1).abs().max(), 1e-6)
        print(pd.concat([pd.Series(REAL_MODEL_SENTENCES, name="teks"), out], axis=1).to_string(index=False))


if __name__ == "__main__":
    unittest.main()
