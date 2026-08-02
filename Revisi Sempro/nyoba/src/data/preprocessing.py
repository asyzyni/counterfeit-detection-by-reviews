import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob
import sys

# Add src to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config

# Basic slang mapping for Indonesian reviews
SLANG_MAP = {
    "yg": "yang",
    "dgn": "dengan",
    "gpp": "tidak apa-apa",
    "gk": "tidak",
    "ga": "tidak",
    "dpt": "dapat",
    "bgt": "banget",
    "bgtt": "banget",
    "tp": "tapi",
    "tpi": "tapi",
    "sy": "saya",
    "sya": "saya",
    "klo": "kalau",
    "klw": "kalau",
    "dgn": "dengan",
    "hp": "handphone",
    "hape": "handphone",
    "hpx": "handphonenya",
    "barangya": "barangnya",
    "barngnya": "barangnya",
    "mantaaaaab": "mantap",
    "mantpsssssssss": "mantap",
    "mantap": "mantap",
    "ori": "original",
    "kw": "tiruan",
    "palsu": "palsu",
    "lemot": "lambat",
    "seler": "penjual",
    "seller": "penjual",
    "trm": "terima",
    "kasih": "kasih",
    "tks": "terima kasih",
    "mksih": "terima kasih",
    "terimakasih": "terima kasih",
    "co": "checkout",
    "dtg": "datang",
    "bocill": "bocah",
    "bocil": "bocah"
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Case folding
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\s*\S+|www\.\S+', '', text)
    
    # 3. Handle specific unicode characters (e.g. mathematical bold/italic script like "𝑖𝑃ℎ𝑜𝑛𝑒 𝑝𝑎𝑙𝑠𝑢")
    # Mapping some common script letters to standard letters if necessary
    # A simple regex to keep standard Indonesian/English characters, numbers, and space
    # Remove emoji and strange characters, but keep standard alphanumeric
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize unicode script letters to regular letters if they are common (like math letters in TAOB1688)
    # The letters like 𝑖 𝑃 ℎ 𝑜 𝑛 𝑒 𝑝 𝑎 𝑙 𝑠 𝑢 are mathematical italic small letters.
    # We can normalize them using a transliteration mapping or simple replace.
    # Let's map italic script characters:
    script_chars = {
        '𝑖': 'i', '𝑃': 'p', 'ℎ': 'h', '𝑜': 'o', '𝑛': 'n', '𝑒': 'e',
        '𝑝': 'p', '𝑎': 'a', '𝑙': 'l', '𝑠': 's', '𝑢': 'u', '𝑑': 'd',
        '𝑘': 'k', '𝑡': 't', '𝑟': 'r', '𝑏': 'b', '𝑔': 'g', '𝑗': 'j',
        '𝑚': 'm', 'ℎ': 'h'
    }
    for script_c, regular_c in script_chars.items():
        text = text.replace(script_c, regular_c)
        
    # 4. Tokenize and replace slang
    tokens = text.split()
    normalized_tokens = [SLANG_MAP.get(token, token) for token in tokens]
    
    # 5. Rejoin and remove excess whitespaces
    cleaned = " ".join(normalized_tokens)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def parse_indonesian_timestamp(ts_str):
    """
    Parses dynamic Indonesian timestamps.
    E.g.: 
      - "2026-04-14"
      - "2025-09-08 11:53 | Variasi: Silver"
      - "3 minggu lalu" (3 weeks ago)
      - "2 minggu lalu" (2 weeks ago)
    """
    if not isinstance(ts_str, str) or pd.isna(ts_str):
        return None
    
    # Remove Shopee variations separator
    ts_str = ts_str.split("|")[0].strip()
    
    # Check standard datetime format: YYYY-MM-DD HH:MM
    try:
        return pd.to_datetime(ts_str)
    except:
        pass
        
    # Check standard date format: YYYY-MM-DD
    try:
        return datetime.strptime(ts_str, "%Y-%m-%d")
    except:
        pass
        
    # Handle relative dates (e.g. "X minggu lalu", "X hari lalu")
    # We use a base date of 2026-06-01 for calculations
    base_date = datetime(2026, 6, 1)
    
    match_weeks = re.search(r'(\d+)\s+minggu\s+lalu', ts_str, re.IGNORECASE)
    if match_weeks:
        weeks = int(match_weeks.group(1))
        return base_date - timedelta(weeks=weeks)
        
    match_days = re.search(r'(\d+)\s+hari\s+lalu', ts_str, re.IGNORECASE)
    if match_days:
        days = int(match_days.group(1))
        return base_date - timedelta(days=days)
        
    match_months = re.search(r'(\d+)\s+bulan\s+lalu', ts_str, re.IGNORECASE)
    if match_months:
        months = int(match_months.group(1))
        return base_date - timedelta(days=months * 30)
        
    if "hari ini" in ts_str.lower():
        return base_date
        
    if "kemarin" in ts_str.lower():
        return base_date - timedelta(days=1)
        
    return None

def process_all_raw_data():
    raw_files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
    print(f"Found {len(raw_files)} raw CSV files for preprocessing.")
    
    all_data = []
    
    for file_path in raw_files:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if file_name not in config.STORE_LABELS:
            print(f"Warning: File {file_name} has no label in STORE_LABELS, skipping.")
            continue
            
        store_label = config.STORE_LABELS[file_name]
        print(f"Processing: {file_name} (Label: {store_label})")
        
        try:
            df = pd.read_csv(file_path)
            
            # Identify columns
            # Text review column is usually 'review'
            text_col = 'review' if 'review' in df.columns else None
            # Time column is either 'timestamp' or 'Timestamp'
            time_col = 'timestamp' if 'timestamp' in df.columns else ('Timestamp' if 'Timestamp' in df.columns else None)
            
            if not text_col or not time_col:
                print(f"Error: Missing review or timestamp column in {file_name}. Columns: {list(df.columns)}")
                continue
                
            # Copy relevant columns and drop rows with empty reviews
            processed_df = pd.DataFrame()
            processed_df['review_id'] = df['web_scraper_order'] if 'web_scraper_order' in df.columns else [f"{file_name}-{i}" for i in range(len(df))]
            processed_df['store_id'] = file_name
            processed_df['review_raw'] = df[text_col]
            processed_df['timestamp_raw'] = df[time_col]
            processed_df['store_label'] = store_label
            
            # Clean text
            processed_df['review_text'] = processed_df['review_raw'].apply(clean_text)
            
            # Parse timestamp
            processed_df['review_date'] = processed_df['timestamp_raw'].apply(parse_indonesian_timestamp)
            
            # Drop empty reviews
            processed_df = processed_df.dropna(subset=['review_text', 'review_date'])
            processed_df = processed_df[processed_df['review_text'].str.strip() != ""]
            
            # Drop duplicates based on cleaned review_text to prevent fake spam
            initial_count = len(processed_df)
            processed_df = processed_df.drop_duplicates(subset=['review_text'])
            print(f"  Rows after dropping empty/duplicates: {len(processed_df)} (dropped {initial_count - len(processed_df)})")
            
            all_data.append(processed_df)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
    if all_data:
        consolidated_df = pd.concat(all_data, ignore_index=True)
        # Sort by store_id and review_date (chronological order)
        consolidated_df = consolidated_df.sort_values(by=['store_id', 'review_date']).reset_index(drop=True)
        
        output_path = os.path.join(config.INTERIM_DATA_DIR, "cleaned_reviews.csv")
        consolidated_df.to_csv(output_path, index=False)
        print(f"\nSaved consolidated preprocessed reviews to {output_path} with {len(consolidated_df)} rows.")
        print(consolidated_df['store_id'].value_counts())
    else:
        print("Error: No data processed!")

if __name__ == "__main__":
    process_all_raw_data()
