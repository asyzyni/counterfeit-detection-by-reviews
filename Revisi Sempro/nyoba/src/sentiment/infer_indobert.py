import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config
from src.sentiment.dataset import ReviewDataset

def infer():
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Check if we have a fine-tuned model checkpoint
    if os.path.exists(os.path.join(config.INDOBERT_CHECKPOINT_DIR, "config.json")):
        model_path = config.INDOBERT_CHECKPOINT_DIR
        print(f"Loading custom fine-tuned model from checkpoint: {model_path}")
    else:
        model_path = config.INDOBERT_MODEL_NAME
        print(f"Checkpoint not found. Defaulting to pre-trained model: {model_path}")
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load consolidated cleaned reviews
    input_path = os.path.join(config.INTERIM_DATA_DIR, "cleaned_reviews.csv")
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist. Please run preprocessing first.")
        return
        
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} cleaned reviews for inference.")
    
    # Create dataset & dataloader
    dataset = ReviewDataset(
        texts=df['review_text'].tolist(),
        tokenizer=tokenizer,
        max_length=config.MAX_SEQ_LENGTH
    )
    
    # We use a batch size of 32 for fast inference
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Sentiment Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply softmax to get probability scores
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Add sentiment predictions and probabilities to dataframe
    df['sentiment_label'] = all_preds
    
    all_probs = np.array(all_probs)
    df['prob_neg'] = all_probs[:, 0]
    df['prob_neu'] = all_probs[:, 1]
    df['prob_pos'] = all_probs[:, 2]
    
    # Convert label to human readable name
    df['sentiment_name'] = df['sentiment_label'].map(config.SENTIMENT_LABELS)
    
    output_path = os.path.join(config.INTERIM_DATA_DIR, "cleaned_reviews_with_sentiment.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved sentiment inference results to {output_path}")
    
    # Display some stats
    print("\nSentiment Label Distribution:")
    print(df['sentiment_name'].value_counts())
    
    # Check how labels distribute by store
    print("\nSentiment distribution per store:")
    print(df.groupby('store_id')['sentiment_name'].value_counts().unstack().fillna(0))

if __name__ == "__main__":
    infer()
