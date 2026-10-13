import itertools
import unittest

import numpy as np

from src.temporal.gaussian_hmm import GaussianDiagHMM, baum_welch


def random_model(rng, k=2, d=3):
    pi = rng.dirichlet(np.ones(k))
    a = rng.dirichlet(np.ones(k), size=k)
    return GaussianDiagHMM(pi, a, rng.normal(size=(k, d)), rng.uniform(0.3, 1.5, size=(k, d)))


def brute_force(model, X):
    """Enumerasi semua jalur state: acuan independen untuk forward dan posterior."""
    k, t_len = model.n_states, len(X)
    log_b = model.log_emission(X)
    total, gamma = 0.0, np.zeros((t_len, k))
    for path in itertools.product(range(k), repeat=t_len):
        p = model.startprob[path[0]] * np.exp(log_b[0, path[0]])
        for t in range(1, t_len):
            p *= model.transmat[path[t - 1], path[t]] * np.exp(log_b[t, path[t]])
        total += p
        for t, s in enumerate(path):
            gamma[t, s] += p
    return np.log(total), gamma / total


class TestInference(unittest.TestCase):
    def test_forward_and_posterior_match_brute_force(self):
        rng = np.random.default_rng(1)
        for k in (2, 3):
            model = random_model(rng, k=k)
            X = rng.normal(size=(5, 3))
            ll_ref, gamma_ref = brute_force(model, X)
            self.assertAlmostEqual(model.loglik(X), ll_ref, places=10)
            np.testing.assert_allclose(model.posterior(X), gamma_ref, atol=1e-10)

    def test_batched_equals_single_with_unequal_lengths(self):
        rng = np.random.default_rng(2)
        model = random_model(rng)
        lengths = np.array([3, 7, 4, 12])
        X = rng.normal(size=(lengths.sum(), 3))
        batch = model.score_each(X, lengths)
        offs = np.concatenate([[0], np.cumsum(lengths)])
        for i in range(len(lengths)):
            seq = X[offs[i]:offs[i + 1]]
            self.assertAlmostEqual(batch[i], model.loglik(seq), places=10)
        gamma, ll, _ = model.posteriors(X, lengths)
        np.testing.assert_allclose(ll, batch, atol=1e-10)
        self.assertEqual(gamma.shape, (lengths.sum(), 2))
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-10)

    def test_sequences_are_not_concatenated(self):
        """Menyambung dua sekuens memberi log-likelihood berbeda dari menjumlahkan keduanya."""
        rng = np.random.default_rng(3)
        model = random_model(rng)
        a, b = rng.normal(size=(6, 3)), rng.normal(size=(5, 3))
        separate = model.loglik(a) + model.loglik(b)
        joined = model.loglik(np.vstack([a, b]))
        self.assertNotAlmostEqual(separate, joined, places=3)
        self.assertAlmostEqual(model.score_each(np.vstack([a, b]), np.array([6, 5])).sum(), separate, places=10)


class TestBaumWelch(unittest.TestCase):
    def test_loglik_monotone_and_params_valid(self):
        rng = np.random.default_rng(4)
        true = GaussianDiagHMM([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]], [[0, 0, 0], [3, 3, 3]], np.ones((2, 3)) * 0.5)
        lengths = rng.integers(8, 30, size=40)
        chunks = []
        for t_len in lengths:
            s = rng.choice(2, p=true.startprob)
            seq = []
            for _ in range(t_len):
                seq.append(rng.normal(true.means[s], np.sqrt(true.variances[s])))
                s = rng.choice(2, p=true.transmat[s])
            chunks.append(np.array(seq))
        X = np.vstack(chunks)
        init = GaussianDiagHMM([0.6, 0.4], [[0.8, 0.2], [0.3, 0.7]], [[0.5, 0.5, 0.5], [2, 2, 2]], np.ones((2, 3)))
        res = baum_welch(init, X, lengths, n_iter=100, tol=1e-6)
        diffs = np.diff(res.loglik_history)
        self.assertGreaterEqual(diffs.min(), -1e-6)
        self.assertEqual(res.n_decreases, 0)
        res.model.check_valid()
        np.testing.assert_allclose(res.model.means, true.means, atol=0.5)


if __name__ == "__main__":
    unittest.main()
