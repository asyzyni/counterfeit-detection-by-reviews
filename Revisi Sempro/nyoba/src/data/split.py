import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config

def split_data():
    input_path = os.path.join(config.INTERIM_DATA_DIR, "cleaned_reviews.csv")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run preprocessing first.")
        return
        
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} cleaned reviews.")
    
    # We will split reviews chronologically for each store:
    # 70% Train, 15% Val, 15% Test
    # This prevents future-to-past data leakage and allows us to have Asli (Panda) 
    # and Counterfeit data in all train, val, and test splits.
    
    splits = []
    
    for store_id, group in df.groupby('store_id'):
        group = group.sort_values(by='review_date').reset_index(drop=True)
        n = len(group)
        
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        
        # Assign splits
        for idx in range(n):
            if idx < n_train:
                split_name = 'train'
            elif idx < n_train + n_val:
                split_name = 'val'
            else:
                split_name = 'test'
                
            splits.append({
                'review_id': group.loc[idx, 'review_id'],
                'store_id': store_id,
                'split': split_name
            })
            
    split_df = pd.DataFrame(splits)
    
    # Save the mapping of review_id -> split
    output_map_path = os.path.join(config.PROCESSED_DATA_DIR, "split_map.csv")
    split_df.to_csv(output_map_path, index=False)
    print(f"Saved split map to {output_map_path}")
    
    # Merge back to save train/val/test CSVs
    final_df = df.merge(split_df, on=['review_id', 'store_id'])
    
    for split_name in ['train', 'val', 'test']:
        split_data = final_df[final_df['split'] == split_name].copy()
        split_out_path = os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}.csv")
        split_data.to_csv(split_out_path, index=False)
        print(f"  Saved {split_name} split to {split_out_path} ({len(split_data)} rows)")
        print(split_data.groupby('store_id')['store_label'].value_counts())

if __name__ == "__main__":
    split_data()
