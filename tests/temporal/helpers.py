"""Data sintetis untuk menguji RM 2: pola yang diketahui, tanpa model IndoBERT."""

from __future__ import annotations

import numpy as np
import pandas as pd

MEAN_GOOD = np.array([0.75, 0.20, 0.05])  # [S+, S~, S-]
MEAN_BAD = np.array([0.10, 0.20, 0.70])


def _emit(rng: np.random.Generator, state: int, conc: float = 30.0) -> np.ndarray:
    mean = MEAN_GOOD if state == 0 else MEAN_BAD
    return rng.dirichlet(conc * mean)


def make_table(n_per_kind: int = 12, length: int = 20, seed: int = 0, with_label: bool = True) -> pd.DataFrame:
    """Tiga jenis produk: 'good' (konsisten positif), 'shift' (bergeser ke negatif), 'bad' (negatif)."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("good", "shift", "bad"):
        for i in range(n_per_kind):
            pid = f"{kind}-{i:03d}"
            t_len = length + int(rng.integers(-3, 4))
            if kind == "good":
                states = (rng.random(t_len) < 0.05).astype(int)
            elif kind == "bad":
                states = (rng.random(t_len) < 0.9).astype(int)
            else:
                cut = t_len // 2
                states = np.r_[(rng.random(cut) < 0.05), (rng.random(t_len - cut) < 0.95)].astype(int)
            for t, s in enumerate(states):
                p_pos, p_neu, p_neg = _emit(rng, int(s))
                rows.append({
                    "product_id": pid,
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=t),
                    "seq_in_product": t + 1,
                    "row_order": t,
                    "include_for_inference": True,
                    "p_positif": p_pos, "p_netral": p_neu, "p_negatif": p_neg,
                    "logit_positif": np.log(p_pos), "logit_netral": np.log(p_neu), "logit_negatif": np.log(p_neg),
                    "label_counterfeit": int(kind != "good") if with_label else None,
                })
    df = pd.DataFrame(rows)
    if not with_label:
        df = df.drop(columns=["label_counterfeit"])
    return df
