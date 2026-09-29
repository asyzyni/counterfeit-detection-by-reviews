from __future__ import annotations

import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev


DATA_DIR = Path(__file__).resolve().parent
REPORT_PATH = DATA_DIR / "EDA_REPORT.md"
EXPECTED_COLUMNS = [
    "web_scraper_order",
    "web_scraper_start_url",
    "pagination",
    "review",
    "timestamp",
    "variation",
    "rating",
]
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "saya", "sudah", "dengan",
    "sesuai", "barang", "produk", "hp", "handphone", "ponsel", "nya", "kak", "ya", "terima",
    "kasih", "sangat", "lebih", "pada", "ada", "jadi", "juga", "semoga", "beli", "toko",
    "seller", "sampai", "aman", "baik", "dan", "di", "ke", "dalam", "dapat", "buat",
}


def split_file_name(file_path: Path) -> tuple[str, str]:
    """Memisahkan nama toko dan nama produk dari pola toko_produk.csv."""
    stem = file_path.stem
    if "_" not in stem:
        return "Tidak diketahui", stem
    return tuple(stem.rsplit("_", 1))  # type: ignore[return-value]


def brand_from_product(product: str) -> str:
    checks = [
        ("Samsung", "Samsung"), ("iPhone", "Apple"), ("Xiaomi", "Xiaomi"),
        ("Redmi", "Xiaomi"), ("POCO", "Xiaomi"), ("OPPO", "OPPO"),
        ("vivo", "vivo"), ("iQOO", "vivo/iQOO"), ("realme", "realme"),
        ("Infinix", "Infinix"), ("TECNO", "TECNO"),
    ]
    for prefix, brand in checks:
        if product.startswith(prefix):
            return brand
    return "Lainnya"


def safe_pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100) if denominator else 0.0


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def top_rows(counter: Counter, limit: int = 10) -> list[list[object]]:
    return [[label, f"{count:,}"] for label, count in counter.most_common(limit)]


def run_eda() -> dict[str, object]:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Tidak menemukan CSV di {DATA_DIR}")

    file_review_counts: list[tuple[str, str, str, int]] = []
    schema_counter: Counter[tuple[str, ...]] = Counter()
    shop_review_counts: Counter[str] = Counter()
    product_review_counts: Counter[str] = Counter()
    brand_review_counts: Counter[str] = Counter()
    brand_file_counts: Counter[str] = Counter()
    rating_counts: Counter[int] = Counter()
    month_counts: Counter[str] = Counter()
    capacity_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    word_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    review_lengths: list[int] = []
    review_word_counts: list[int] = []
    timestamps: list[datetime] = []
    invalid_timestamp_count = 0
    empty_review_count = 0
    invalid_rating_count = 0
    malformed_row_count = 0
    generated_id_count = 0
    total_rows = 0

    for csv_path in csv_files:
        shop, product = split_file_name(csv_path)
        brand = brand_from_product(product)
        brand_file_counts[brand] += 1
        file_rows = 0

        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or [])
            schema_counter[fields] += 1

            for row in reader:
                total_rows += 1
                file_rows += 1
                if set(row) != set(EXPECTED_COLUMNS) or any(value is None for value in row.values()):
                    malformed_row_count += 1

                review = (row.get("review") or "").strip()
                if not review:
                    empty_review_count += 1
                else:
                    review_counts[review] += 1
                    review_lengths.append(len(review))
                    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", review.lower())
                    review_word_counts.append(len(tokens))
                    word_counts.update(token for token in tokens if len(token) >= 3 and token not in STOPWORDS)

                if (row.get("web_scraper_order") or "").startswith("generated-"):
                    generated_id_count += 1

                try:
                    rating_counts[int(row.get("rating") or "")] += 1
                except ValueError:
                    invalid_rating_count += 1

                timestamp_text = (row.get("timestamp") or "").strip()
                try:
                    parsed_time = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M")
                    timestamps.append(parsed_time)
                    month_counts[parsed_time.strftime("%Y-%m")] += 1
                except ValueError:
                    invalid_timestamp_count += 1

                variation = row.get("variation") or ""
                capacity = re.search(r"(\d+/\d+GB)", variation)
                if capacity:
                    capacity_counts[capacity.group(1)] += 1
                color = re.search(r"Variasi:\s*([^,]+)", variation)
                if color:
                    color_counts[color.group(1).strip()] += 1

        file_review_counts.append((csv_path.name, shop, product, file_rows))
        shop_review_counts[shop] += file_rows
        product_review_counts[product] += file_rows
        brand_review_counts[brand] += file_rows

    unique_reviews = len(review_counts)
    duplicated_review_rows = sum(count for count in review_counts.values() if count > 1)
    duplicate_excess_rows = total_rows - unique_reviews
    rating_total = sum(rating_counts.values())
    average_rating = sum(rating * count for rating, count in rating_counts.items()) / rating_total

    return {
        "csv_files": csv_files,
        "file_review_counts": file_review_counts,
        "schema_counter": schema_counter,
        "shop_review_counts": shop_review_counts,
        "product_review_counts": product_review_counts,
        "brand_review_counts": brand_review_counts,
        "brand_file_counts": brand_file_counts,
        "rating_counts": rating_counts,
        "month_counts": month_counts,
        "capacity_counts": capacity_counts,
        "color_counts": color_counts,
        "word_counts": word_counts,
        "review_counts": review_counts,
        "review_lengths": review_lengths,
        "review_word_counts": review_word_counts,
        "timestamps": timestamps,
        "invalid_timestamp_count": invalid_timestamp_count,
        "empty_review_count": empty_review_count,
        "invalid_rating_count": invalid_rating_count,
        "malformed_row_count": malformed_row_count,
        "generated_id_count": generated_id_count,
        "total_rows": total_rows,
        "unique_reviews": unique_reviews,
        "duplicated_review_rows": duplicated_review_rows,
        "duplicate_excess_rows": duplicate_excess_rows,
        "average_rating": average_rating,
    }
