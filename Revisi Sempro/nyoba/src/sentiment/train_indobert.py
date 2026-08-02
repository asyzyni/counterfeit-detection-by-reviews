import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils import config
from src.sentiment.dataset import ReviewDataset

def load_smsa_dataset():
    """
    Downloads and loads indonlp/smsa dataset from HuggingFace datasets.
    Aligns labels to: 0 = negative, 1 = neutral, 2 = positive.
    Original SMSA mapping: 0=positive, 1=neutral, 2=negative.
    """
    print("Loading indonlp/smsa dataset from HuggingFace Hub...")
    from datasets import load_dataset
    dataset = load_dataset("indonlp/smsa")
    
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    
    # Map labels:
    # Original: 0=pos, 1=neu, 2=neg
    # Target: 0=neg, 1=neu, 2=pos
    mapping = {0: 2, 1: 1, 2: 0}
    
    train_df['label'] = train_df['label'].map(mapping)
    val_df['label'] = val_df['label'].map(mapping)
    
    return train_df, val_df

def train():
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load dataset
    train_df, val_df = load_smsa_dataset()
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Load model and tokenizer
    # We use indobenchmark/indobert-base-p1 as standard base model
    model_name = "indobenchmark/indobert-base-p1"
    print(f"Initializing tokenizer and model from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    
    # Create dataset & dataloader
    train_dataset = ReviewDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=config.MAX_SEQ_LENGTH
    )
    val_dataset = ReviewDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=config.MAX_SEQ_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, correct_bias=False)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # For class imbalance handling (compute class weights)
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = sum(class_counts)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights computed: {class_weights}")
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    best_f1 = 0.0
    
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        # Training loop
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                preds = torch.argmax(logits, dim=-1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Validation Accuracy: {val_acc:.4f} | Validation F1 (Macro): {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            print(f"Validation F1 improved from {best_f1:.4f} to {val_f1:.4f}. Saving best model...")
            os.makedirs(config.INDOBERT_CHECKPOINT_DIR, exist_ok=True)
            model.save_pretrained(config.INDOBERT_CHECKPOINT_DIR)
            tokenizer.save_pretrained(config.INDOBERT_CHECKPOINT_DIR)
            
    print(f"\nTraining completed! Best Validation F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()
