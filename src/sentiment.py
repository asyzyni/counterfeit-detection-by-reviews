from __future__ import annotations
from os import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from datasets import Dataset
import numpy as np
import pandas as pd
import torch
import accelerate

from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer, 
    TrainingArguments,
)

from .config import ExperimentConfig 

## Mapping Label 

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

class IndoBERTSentimentTrainer:

    def __init__(
        self,
        config: ExperimentConfig,
    ):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.sentiment_base_model
        )
        
        model_config = AutoConfig.from_pretrained(
            self.config.sentiment_base_model
        )

        model_config.num_labels = 3 
        model_config.label2id = LABEL2ID
        model_config.id2label = ID2LABEL

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                self.config.sentiment_base_model,
                config=model_config,
            )
        )

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

        self.training_args = TrainingArguments(
            output_dir=str(
                self.config.sentiment_model_dir
        ),

        num_train_epochs=(
            self.config.sentiment_num_epochs
        ),

        learning_rate=(
            self.config.sentiment_learning_rate
        ),

        weight_decay=(
            self.config.sentiment_weight_decay
        ),

        warmup_steps=0.1,

        per_device_train_batch_size=(
            self.config.sentiment_train_batch_size
        ),

        per_device_eval_batch_size=(
            self.config.sentiment_eval_batch_size
        ),

        eval_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1_macro",
        greater_is_better=True,

        save_total_limit=2,

        report_to="none",

        seed=self.config.random_seed,
    )

        self.trainer = None

    def _tokenize(
    self, batch, 
    ): 
        return self.tokenizer(
            batch['text'], truncation=True, max_length=self.config.sentiment_max_length,
        )



    def _create_dataset(
        self, 
        df: pd.DataFrame,
    ): 

        dataset = Dataset.from_pandas(
            df[
                ['text', 'label_id']
            ], preserve_index=False,
        )

        dataset = dataset.rename_column(
            "label_id", "labels", 
        )

        dataset = dataset.map(
            self._tokenize, batched=True, remove_columns=['text'], 
        )

        return dataset

    
    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        predictions = np.argmax(
            logits, axis=-1,
        )

        accuracy = accuracy_score(
            labels, predictions,
        )

        macro_f1 = f1_score(
            labels, predictions, average="macro",
        )

        weighted_f1 = f1_score(
            labels, predictions, average="weighted",
        )

        return {
            "accuracy": accuracy,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1,
        }

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ):

        train_dataset = self._create_dataset(
            train_df
        )

        valid_dataset = self._create_dataset(
            valid_df
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

        return self.trainer

    def evaluate(
        self,
        test_df: pd.DataFrame,
    ):
        if self.trainer is None:
            raise RuntimeError(
                "Model belum di-training. Jalankan fit() terlebih dahulu."
            )

        test_dataset = self._create_dataset(
            test_df
        )

        metrics = self.trainer.evaluate(
            eval_dataset=test_dataset
        )

        return metrics

    def save(self):
        if self.trainer is None:
            raise RuntimeError(
                "Tidak ada model hasil training. "
            "Jalankan fit() terlebih dahulu."
        )

        save_dir = self.config.sentiment_model_dir

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.trainer.save_model(
        str(save_dir)
        )

        self.tokenizer.save_pretrained(
            str(save_dir)
        )

        print(
            f"Model sentiment disimpan di: {save_dir}"
        )
    

    