import os
import sys
import pandas as pd
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config

def build_sequences(window_size=10):
    input_path = os.path.join(config.INTERIM_DATA_DIR, "cleaned_reviews_with_sentiment.csv")
    split_map_path = os.path.join(config.PROCESSED_DATA_DIR, "split_map.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run sentiment inference first.")
        return
        
    df = pd.read_csv(input_path)
    split_df = pd.read_csv(split_map_path)
    
    # Merge split mapping
    df = df.merge(split_df, on=['review_id', 'store_id'])
    
    # Sort reviews chronologically
    df['review_date'] = pd.to_datetime(df['review_date'])
    df = df.sort_values(by=['store_id', 'review_date']).reset_index(drop=True)
    
    sequences_by_split = {'train': [], 'val': [], 'test': []}
    
    for (store_id, split_name), group in df.groupby(['store_id', 'split']):
        group = group.sort_values(by='review_date').reset_index(drop=True)
        n = len(group)
        
        # Divide into non-overlapping windows of window_size
        for i in range(0, n, window_size):
            window = group.iloc[i : i + window_size]
            if len(window) > 0:
                obs_seq = window['sentiment_label'].tolist()
                neg_probs = window['prob_neg'].tolist()
                neu_probs = window['prob_neu'].tolist()
                pos_probs = window['prob_pos'].tolist()
                
                sequences_by_split[split_name].append({
                    'sequence_id': f"{store_id}_{split_name}_seq{i//window_size}",
                    'store_id': store_id,
                    'store_label': int(window.iloc[0]['store_label']),
                    'observations': obs_seq,
                    'prob_matrix': list(zip(neg_probs, neu_probs, pos_probs)), # For potential continuous HMM or soft decoding
                    'length': len(window),
                    'start_date': str(window.iloc[0]['review_date'].date()),
                    'end_date': str(window.iloc[-1]['review_date'].date())
                })
                
    # Save as Pickle (preserves nested structure)
    pkl_output_path = os.path.join(config.PROCESSED_DATA_DIR, "sequences.pkl")
    with open(pkl_output_path, 'wb') as f:
        pickle.dump(sequences_by_split, f)
    print(f"Saved aggregated sequences pickle to {pkl_output_path}")
    
    # Export split sequences summary as CSV for readability
    for split_name in ['train', 'val', 'test']:
        seqs = sequences_by_split[split_name]
        summary_rows = []
        for s in seqs:
            summary_rows.append({
                'sequence_id': s['sequence_id'],
                'store_id': s['store_id'],
                'store_label': s['store_label'],
                'sequence_length': s['length'],
                'observations_str': ",".join(map(str, s['observations'])),
                'start_date': s['start_date'],
                'end_date': s['end_date']
            })
        summary_df = pd.DataFrame(summary_rows)
        csv_output_path = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_sequences.csv")
        summary_df.to_csv(csv_output_path, index=False)
        print(f"  Saved {split_name} sequences summary ({len(summary_df)} sequences) to {csv_output_path}")
        print(summary_df['store_label'].value_counts())

if __name__ == "__main__":
    build_sequences()
