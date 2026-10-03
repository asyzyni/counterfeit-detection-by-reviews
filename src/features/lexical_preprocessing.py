from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ..config import ExperimentConfig


class LexicalTextPreprocessor:
    NEGATION_WORDS = {
        "tidak",
        "bukan",
        "belum",
    }

    def __init__(
        self,
        config: ExperimentConfig,
        genuine_seeds: Sequence[str],
        counterfeit_seeds: Sequence[str],
        domain_dictionary: Mapping[str, str] | None = None,
        public_dictionary: Mapping[str, str] | None = None,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        self.config = config

        self.domain_dictionary = {
            self._canonicalize(k): self._canonicalize(v)
            for k, v in (domain_dictionary or {}).items()
        }

        self.public_dictionary = {
            self._canonicalize(k): self._canonicalize(v)
            for k, v in (public_dictionary or {}).items()
        }

        self.stopwords = {
            self._canonicalize(word)
            for word in (stopwords or [])
        }

        self.genuine_seeds = {
            self.normalize_phrase(seed)
            for seed in genuine_seeds
        }

        self.counterfeit_seeds = {
            self.normalize_phrase(seed)
            for seed in counterfeit_seeds
        }

        self.seed_phrases = (
            self.genuine_seeds
            | self.counterfeit_seeds
        )

        self.seed_tokens = {
            token
            for phrase in self.seed_phrases
            for token in phrase.split()
        }

        self.protect_words = (
            self.seed_tokens
            | self.NEGATION_WORDS
        )

        self.sorted_seed_phrases = sorted(
            self.seed_phrases,
            key=lambda value: len(value.split()),
            reverse=True,
        )

    @staticmethod
    def _canonicalize(text: str) -> str:
        text = unicodedata.normalize(
            "NFKC",
            str(text),
        )

        text = text.lower().strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    def normalize_token(
        self,
        token: str,
    ) -> str:

        token = self._canonicalize(token)

        # Protected words tidak boleh diubah dictionary.
        if token in self.protect_words:
            return token

        if token in self.domain_dictionary:
            return self.domain_dictionary[token]

        if token in self.public_dictionary:
            return self.public_dictionary[token]

        return token

    def normalize_phrase(
        self,
        phrase: str,
    ) -> str:

        phrase = self._canonicalize(phrase)

        tokens = phrase.split()

        normalized = [
            self.normalize_token(token)
            for token in tokens
        ]

        return " ".join(normalized)

    def normalize_spelling(
        self,
        text: str,
    ) -> str:

        text = self._canonicalize(text)

        return " ".join(
            self.normalize_token(token)
            for token in text.split()
        )

    def join_seed_phrases(
        self,
        text: str,
    ) -> str:

        for phrase in self.sorted_seed_phrases:

            if " " not in phrase:
                continue

            replacement = phrase.replace(
                " ",
                "_",
            )

            pattern = (
                r"(?<!\w)"
                + re.escape(phrase)
                + r"(?!\w)"
            )

            text = re.sub(
                pattern,
                replacement,
                text,
            )

        return text

    @staticmethod
    def remove_punctuation_and_digits(
        text: str,
    ) -> str:

        # underscore sengaja dipertahankan
        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
        )

        text = re.sub(
            r"\d+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def mark_negation(
        self,
        tokens: list[str],
    ) -> list[str]:

        if not self.config.lexical_mark_negation:
            return tokens

        result: list[str] = []

        index = 0

        while index < len(tokens):

            token = tokens[index]

            if (
                token in self.NEGATION_WORDS
                and index + 1 < len(tokens)
            ):
                next_token = tokens[index + 1]

                result.append(
                    f"{token}_{next_token}"
                )

                index += 2
                continue

            result.append(token)
            index += 1

        return result

    def is_protected_token(
        self,
        token: str,
    ) -> bool:

        if token in self.seed_tokens:
            return True

        if token in self.protect_words:
            return True

        if any(
            token.startswith(f"{negation}_")
            for negation in self.NEGATION_WORDS
        ):
            return True

        return False

    def remove_stopwords(
        self,
        tokens: list[str],
    ) -> list[str]:

        return [
            token
            for token in tokens
            if (
                token not in self.stopwords
                or self.is_protected_token(token)
            )
        ]

    def transform_before_frequency_filter(
        self,
        text: str,
    ) -> list[str]:

        text = self.normalize_spelling(text)

        text = self.join_seed_phrases(text)

        text = self.remove_punctuation_and_digits(
            text
        )

        tokens = text.split()

        tokens = self.mark_negation(
            tokens
        )

        tokens = self.remove_stopwords(
            tokens
        )

        return tokens

    @staticmethod
    def build_global_token_frequency(
        tokenized_reviews: Iterable[list[str]],
    ) -> Counter[str]:

        frequency: Counter[str] = Counter()

        for tokens in tokenized_reviews:
            frequency.update(tokens)

        return frequency

    def apply_min_frequency_filter(
        self,
        tokens: list[str],
        global_frequency: Counter[str],
    ) -> list[str]:

        minimum = (
            self.config.lexical_min_token_frequency
        )

        return [
            token
            for token in tokens
            if (
                self.is_protected_token(token)
                or global_frequency[token] >= minimum
            )
        ]

    def transform_corpus(
        self,
        data: pd.DataFrame,
        text_column: str = "review",
    ) -> pd.DataFrame:

        if text_column not in data.columns:
            raise ValueError(
                f"Kolom '{text_column}' tidak ditemukan."
            )

        result = data.copy()

        tokenized = [
            self.transform_before_frequency_filter(
                text
            )
            for text in result[
                text_column
            ].fillna("").astype(str)
        ]

        global_frequency = (
            self.build_global_token_frequency(
                tokenized
            )
        )

        filtered = [
            self.apply_min_frequency_filter(
                tokens,
                global_frequency,
            )
            for tokens in tokenized
        ]

        result["text_lex"] = [
            " ".join(tokens)
            for tokens in filtered
        ]

        return result