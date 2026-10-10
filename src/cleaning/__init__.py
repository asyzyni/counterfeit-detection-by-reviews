"""Cleaning & preprocessing data ulasan (Data Beneran) -> tabel review siap inferensi RM 1.

Jalankan: python -m src.cleaning.run
"""

from .loader import load_clean_reviews

__all__ = ["load_clean_reviews"]
