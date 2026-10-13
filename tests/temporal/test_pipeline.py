import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal import HMMConfig, build_sequences, run_rm2, temporary_product_split
from src.temporal.fitting import hmmlearn_available
from src.temporal.pipeline import assert_no_leakage, load_hmm_model, transform_new_data

from .helpers import make_table


def split_by_index(df):
    ids = sorted(df.product_id.unique())
    test = ids[::5]
    return [i for i in ids if i not in test], test


def quick_cfg(**kw):
    base = dict(n_init=4, n_iter=100, backend="numpy", seed=7)
    base.update(kw)
    return HMMConfig(**base)


class TestSequences(unittest.TestCase):
    def test_order_is_deterministic_and_short_products_excluded(self):
        df = make_table(n_per_kind=3, length=10, seed=1)
        short = df[df.product_id == "good-000"].head(2)
        df = pd.concat([df[df.product_id != "good-000"], short])
        shuffled = df.sample(frac=1.0, random_state=3)
        cfg = quick_cfg()
        a, rep_a, _ = build_sequences(df, cfg)
        b, rep_b, _ = build_sequences(shuffled, cfg)
        np.testing.assert_array_equal(a.X, b.X)
        self.assertEqual(a.product_ids, b.product_ids)
        self.assertEqual(rep_a["n_products_excluded_short"], 1)
        self.assertNotIn("good-000", a.product_ids)
        self.assertEqual(a.lengths.sum(), len(a.X))

    def test_equal_timestamps_resolved_by_seq_in_product(self):
        df = make_table(n_per_kind=2, length=8, seed=2)
        df["timestamp"] = pd.Timestamp("2026-01-01")  # semua kembar (hanya tanggal)
        shuffled = df.sample(frac=1.0, random_state=5)
        a, rep, _ = build_sequences(df, quick_cfg())
        b, _, _ = build_sequences(shuffled, quick_cfg())
        np.testing.assert_array_equal(a.X, b.X)
        self.assertEqual(rep["n_unresolved_ties"], 0)
        self.assertGreater(rep["n_rows_tied_timestamp"], 0)

    def test_bad_probabilities_rejected_and_nan_dropped(self):
        df = make_table(n_per_kind=2, length=8, seed=3)
        bad = df.copy()
        bad.loc[bad.index[0], "p_positif"] = 0.99
        with self.assertRaises(ValueError):
            build_sequences(bad, quick_cfg())
        nan = df.copy()
        nan.loc[nan.index[0], ["p_positif", "p_netral", "p_negatif"]] = np.nan
        _, rep, _ = build_sequences(nan, quick_cfg())
        self.assertEqual(rep["n_rows_dropped_nan"], 1)

    def test_label_column_never_read(self):
        df = make_table(n_per_kind=2, length=8, seed=4)
        flipped = df.copy()
        flipped["label_counterfeit"] = 1 - flipped["label_counterfeit"]
        a, _, _ = build_sequences(df, quick_cfg())
        b, _, _ = build_sequences(flipped.drop(columns=["label_counterfeit"]), quick_cfg())
        np.testing.assert_array_equal(a.X, b.X)

    def test_representations_finite(self):
        df = make_table(n_per_kind=2, length=8, seed=5)
        for rep, dim in (("prob", 3), ("logit", 3), ("alr", 2), ("clr", 3)):
            seqs, _, _ = build_sequences(df, quick_cfg(representation=rep))
            self.assertEqual(seqs.X.shape[1], dim)
            self.assertTrue(np.isfinite(seqs.X).all())


class TestLeakageGuards(unittest.TestCase):
    def test_assert_no_leakage(self):
        assert_no_leakage({"a"}, {"a", "b"}, {"c"}, set())
        with self.assertRaises(AssertionError):
            assert_no_leakage({"a", "c"}, {"a", "b"}, {"c"}, set())  # fit di luar Train
        with self.assertRaises(AssertionError):
            assert_no_leakage({"a"}, {"a", "b"}, {"b"}, set())  # Train dan Test beririsan
        with self.assertRaises(AssertionError):
            assert_no_leakage({"a"}, {"a"}, {"c"}, {"c"})  # sintetis di Test

    def test_overlapping_split_rejected_by_pipeline(self):
        df = make_table(n_per_kind=4, length=10, seed=6)
        train, test = split_by_index(df)
        with self.assertRaises(AssertionError):
            run_rm2(df, quick_cfg(), train, test + [train[0]], save=False)

    def test_test_data_does_not_influence_fit(self):
        df = make_table(n_per_kind=6, length=14, seed=7)
        train, test = split_by_index(df)
        r1 = run_rm2(df, quick_cfg(), train, test, save=False)
        altered = df.copy()
        mask = altered.product_id.isin(test)
        altered.loc[mask, ["p_positif", "p_netral", "p_negatif"]] = np.array([0.05, 0.05, 0.90])
        r2 = run_rm2(altered, quick_cfg(), train, test, save=False)
        np.testing.assert_array_equal(r1.fit.model.means, r2.fit.model.means)
        np.testing.assert_array_equal(r1.fit.model.transmat, r2.fit.model.transmat)
        f1 = r1.features.query("split == 'train'").reset_index(drop=True)
        f2 = r2.features.query("split == 'train'").reset_index(drop=True)
        pd.testing.assert_frame_equal(f1, f2)
        self.assertNotEqual(
            r1.features.query("split == 'test'").prop_state1.sum(),
            r2.features.query("split == 'test'").prop_state1.sum(),
        )

    def test_synthetic_products_never_fit_and_flagged(self):
        df = make_table(n_per_kind=6, length=12, seed=8)
        syn = df[df.product_id.str.startswith("bad-00")].copy()
        syn["product_id"] = "syn-" + syn["product_id"]
        df["is_synthetic"] = False
        syn["is_synthetic"] = True
        full = pd.concat([df, syn], ignore_index=True)
        ids = sorted(df.product_id.unique())
        test = ids[::5]
        train = [i for i in ids if i not in test] + sorted(syn.product_id.unique())  # sintetis salah dimasukkan ke Train
        res = run_rm2(full, quick_cfg(), train, test, save=False)
        syn_rows = res.features[res.features.is_synthetic]
        self.assertTrue((syn_rows.split == "synthetic").all())
        self.assertEqual(set(res.features.query("split == 'train'").product_id) & set(syn.product_id), set())
        with self.assertRaises(AssertionError):
            run_rm2(full, quick_cfg(), train, test + sorted(syn.product_id.unique()), save=False)


class TestFitAndFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_table(n_per_kind=14, length=20, seed=11)
        cls.train, cls.test = split_by_index(cls.df)
        cls.res = run_rm2(cls.df, quick_cfg(), cls.train, cls.test, stability_seeds=[1, 2], save=False)

    def test_state_order_and_valid_parameters(self):
        m = self.res.fit.model
        m.check_valid()
        mass = self.res.fit.state_neg_mass
        self.assertLess(mass[0], mass[1])
        self.assertGreater(m.means[1, 2], m.means[0, 2])  # state 1 lebih negatif
        self.assertEqual(self.res.fit.state_names[-1], "state_negatif_tinggi")
        self.assertEqual(self.res.meta["restarts"][0]["init"], "proposal")

    def test_features_move_in_sensible_direction(self):
        f = self.res.features
        kind = f.product_id.str.split("-").str[0]
        mean = f.groupby(kind)[["prop_state1", "pos_transisi", "log_lik"]].mean()
        self.assertLess(mean.loc["good", "prop_state1"], 0.2)
        self.assertGreater(mean.loc["bad", "prop_state1"], 0.7)
        self.assertGreater(mean.loc["shift", "pos_transisi"], mean.loc["good", "pos_transisi"])
        self.assertTrue(((f.prop_state1 >= 0) & (f.prop_state1 <= 1)).all())
        self.assertTrue(((f.pos_transisi >= 0) & (f.pos_transisi <= 1)).all())
        self.assertTrue(np.isfinite(f[["log_lik", "prop_state1", "pos_transisi"]].to_numpy()).all())
        self.assertEqual(len(f), f.product_id.nunique())

    def test_shift_position_is_near_midpoint(self):
        f = self.res.features
        shift = f[f.product_id.str.startswith("shift") & (f.split != "synthetic")]
        rel = shift.idx_transisi / shift.n_reviews
        self.assertGreater((rel.between(0.25, 0.75)).mean(), 0.5)

    def test_loglik_is_per_timestep(self):
        # log_lik harus sebanding antar panjang sekuens, bukan tumbuh dengan T
        f = self.res.features
        self.assertLess(abs(np.corrcoef(f.n_reviews, f.log_lik)[0, 1]), 0.6)

    def test_seed_stability_reported(self):
        stab = self.res.meta["seed_stability"]["seeds"]
        self.assertEqual(len(stab), 2)
        for row in stab:
            self.assertTrue(row["state_order_consistent"])
            self.assertGreater(row["corr_prop_state1"], 0.95)

    def test_deterministic_given_seed(self):
        again = run_rm2(self.df, quick_cfg(), self.train, self.test, save=False)
        pd.testing.assert_frame_equal(self.res.features, again.features)

    def test_other_representations_run(self):
        for rep in ("logit", "alr", "clr"):
            res = run_rm2(self.df, quick_cfg(representation=rep, n_init=3), self.train, self.test, save=False)
            res.fit.model.check_valid()
            self.assertTrue(np.isfinite(res.features[["log_lik", "prop_state1", "pos_transisi"]].to_numpy()).all())
            self.assertLess(res.fit.state_neg_mass[0], res.fit.state_neg_mass[1])

    def test_temporary_split_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            temporary_product_split(sorted(self.df.product_id.unique()))
        self.assertTrue(any("TEMPORARY" in str(w.message) for w in caught))


class TestSaving(unittest.TestCase):
    def test_outputs_roundtrip_and_no_overwrite(self):
        df = make_table(n_per_kind=6, length=14, seed=13)
        train, test = split_by_index(df)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = quick_cfg(features_path=tmp / "f.csv", model_path=tmp / "hmm_model.pkl", meta_path=tmp / "hmm_meta.json")
            res = run_rm2(df, cfg, train, test, split_is_temporary=True)
            for p in res.paths.values():
                self.assertTrue(p.is_file())
            meta = __import__("json").loads((tmp / "hmm_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["split_is_temporary"])
            self.assertEqual(len(meta["train_ids_sha256"]), 64)
            self.assertIn("scipy", meta["versions"])
            model, _ = load_hmm_model(tmp / "hmm_model.pkl")
            np.testing.assert_allclose(model.means, res.fit.model.means)
            again = transform_new_data(df[df.product_id.isin(test)], tmp / "hmm_model.pkl")
            expected = res.features.query("split == 'test'")[again.columns.intersection(res.features.columns)]
            np.testing.assert_allclose(
                again.sort_values("product_id").prop_state1.to_numpy(),
                expected.sort_values("product_id").prop_state1.to_numpy(), atol=1e-9)
            with self.assertRaises(FileExistsError):
                run_rm2(df, cfg, train, test)
            run_rm2(df, replace(cfg, overwrite=True), train, test)


@unittest.skipUnless(hmmlearn_available(), "hmmlearn belum terpasang di lingkungan ini")
class TestHmmlearnParity(unittest.TestCase):
    def test_forward_matches_hmmlearn_score_and_posterior(self):
        from hmmlearn.hmm import GaussianHMM

        df = make_table(n_per_kind=6, length=14, seed=21)
        train, test = split_by_index(df)
        res = run_rm2(df, quick_cfg(backend="hmmlearn", n_init=3), train, test, save=False)
        self.assertEqual(res.fit.backend, "hmmlearn")
        m = res.fit.model
        ref = GaussianHMM(n_components=2, covariance_type="diag", init_params="", params="")
        ref.startprob_, ref.transmat_, ref.means_, ref.covars_ = m.startprob, m.transmat, m.means, m.variances
        seqs, _, _ = build_sequences(df, quick_cfg())
        for pid, X in list(seqs)[:8]:
            self.assertAlmostEqual(m.loglik(X), ref.score(X), places=8)
            np.testing.assert_allclose(m.posterior(X), ref.predict_proba(X), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
