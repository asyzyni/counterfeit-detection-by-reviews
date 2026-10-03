from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..config import ExperimentConfig


@dataclass(frozen=True)
class LexiconEntry:
    word: str
    semantic_orientation: float
    ci_lower: float
    ci_upper: float
    label: str


class SemanticOrientationLexicon:

    def __init__(
        self,
        config: ExperimentConfig,
        genuine_seeds: Sequence[str],
        counterfeit_seeds: Sequence[str],
    ) -> None:

        self.config = config

        self.genuine_seeds = set(
            genuine_seeds
        )

        self.counterfeit_seeds = set(
            counterfeit_seeds
        )

    @staticmethod
    def tokenize(
        text: str,
    ) -> list[str]:

        return str(text).split()

    def _collect_statistics(
        self,
        documents: Sequence[list[str]],
    ):

        word_count = Counter()
        cooccurrence = defaultdict(Counter)

        window_size = (
            self.config.pmi_window_size
        )

        total_tokens = 0

        for tokens in documents:

            total_tokens += len(tokens)

            word_count.update(tokens)

            for index, word in enumerate(tokens):

                left = max(
                    0,
                    index - window_size,
                )

                right = min(
                    len(tokens),
                    index + window_size + 1,
                )

                context = tokens[
                    left:index
                ] + tokens[
                    index + 1:right
                ]

                for context_word in context:
                    cooccurrence[word][
                        context_word
                    ] += 1

        return (
            word_count,
            cooccurrence,
            total_tokens,
        )

    @staticmethod
    def _pmi(
        word: str,
        seed: str,
        word_count: Counter,
        cooccurrence,
        total_tokens: int,
    ) -> float:

        joint_count = (
            cooccurrence[word][seed]
        )

        if joint_count == 0:
            return 0.0

        word_frequency = word_count[word]
        seed_frequency = word_count[seed]

        if (
            word_frequency == 0
            or seed_frequency == 0
            or total_tokens == 0
        ):
            return 0.0

        numerator = (
            joint_count
            * total_tokens
        )

        denominator = (
            word_frequency
            * seed_frequency
        )

        return float(
            np.log(
                numerator / denominator
            )
        )

    def calculate_semantic_orientation(
        self,
        documents: Sequence[list[str]],
    ) -> dict[str, float]:

        (
            word_count,
            cooccurrence,
            total_tokens,
        ) = self._collect_statistics(
            documents
        )

        scores: dict[str, float] = {}

        minimum_count = (
            self.config.pmi_min_count
        )

        for word, count in word_count.items():

            if count < minimum_count:
                continue

            genuine_pmi = [
                self._pmi(
                    word,
                    seed,
                    word_count,
                    cooccurrence,
                    total_tokens,
                )
                for seed in self.genuine_seeds
            ]

            counterfeit_pmi = [
                self._pmi(
                    word,
                    seed,
                    word_count,
                    cooccurrence,
                    total_tokens,
                )
                for seed in self.counterfeit_seeds
            ]

            genuine_mean = (
                np.mean(genuine_pmi)
                if genuine_pmi
                else 0.0
            )

            counterfeit_mean = (
                np.mean(counterfeit_pmi)
                if counterfeit_pmi
                else 0.0
            )

            # Sesuai flowchart:
            # SO > 0 -> genuine
            # SO < 0 -> counterfeit

            scores[word] = float(
                genuine_mean
                - counterfeit_mean
            )

        return scores

    def bootstrap(
        self,
        texts: Sequence[str],
    ) -> pd.DataFrame:

        documents = [
            self.tokenize(text)
            for text in texts
        ]

        if not documents:
            raise ValueError(
                "Corpus kosong."
            )

        rng = np.random.default_rng(
            self.config.random_state
        )

        bootstrap_scores = (
            defaultdict(list)
        )

        n_documents = len(documents)

        for _ in range(
            self.config.bootstrap_iterations
        ):

            indices = rng.integers(
                0,
                n_documents,
                size=n_documents,
            )

            sample_documents = [
                documents[index]
                for index in indices
            ]

            scores = (
                self.calculate_semantic_orientation(
                    sample_documents
                )
            )

            for word, score in scores.items():
                bootstrap_scores[word].append(
                    score
                )

        alpha = (
            1.0
            - self.config.confidence_level
        )

        rows = []

        for word, values in (
            bootstrap_scores.items()
        ):

            if not values:
                continue

            values_array = np.asarray(
                values,
                dtype=float,
            )

            ci_lower = float(
                np.quantile(
                    values_array,
                    alpha / 2,
                )
            )

            ci_upper = float(
                np.quantile(
                    values_array,
                    1 - alpha / 2,
                )
            )

            mean_so = float(
                np.mean(values_array)
            )

            # CI melewati nol -> tidak signifikan
            significant = not (
                ci_lower <= 0 <= ci_upper
            )

            if not significant:
                label = "discard"

            elif mean_so > 0:
                label = "genuine"

            else:
                label = "counterfeit"

            rows.append(
                {
                    "word": word,
                    "semantic_orientation": mean_so,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "significant": significant,
                    "label": label,
                }
            )

        return pd.DataFrame(rows)

    def build_final_lexicons(
        self,
        texts: Sequence[str],
    ) -> tuple[
        pd.DataFrame,
        list[str],
        list[str],
    ]:

        lexicon = self.bootstrap(
            texts
        )

        genuine = (
            lexicon.loc[
                lexicon["label"] == "genuine",
                "word",
            ]
            .astype(str)
            .tolist()
        )

        counterfeit = (
            lexicon.loc[
                lexicon["label"] == "counterfeit",
                "word",
            ]
            .astype(str)
            .tolist()
        )

        return (
            lexicon,
            genuine,
            counterfeit,
        )