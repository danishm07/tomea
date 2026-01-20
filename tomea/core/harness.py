"""
Golden Harness - Standard training loop for all papers.

This is written ONCE and never changes. It provides:
- Consistent training setup
- Fair comparison (same hyperparameters)
- Structured metric collection
"""

# This file contains the TEMPLATE for harness.py
# The actual harness.py will be uploaded to Modal

HARNESS_TEMPLATE = '''"""
Golden Training Harness for Tomea
Generated for fair comparison across methods

DO NOT MODIFY - This ensures consistent evaluation
"""

import adapter

import sys
import json
import time
import torch
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score





def load_custom_dataset(dataset_url: str):
    """Load custom dataset from URL."""
    import os
    
    # Download dataset
    df = pd.read_csv(dataset_url)
    
    # Auto-detect columns
    # Assume last column is label, second-to-last is text (or first if only 2 cols)
    if len(df.columns) == 2:
        text_col = df.columns[0]
        label_col = df.columns[1]
    else:
        text_col = df.columns[-2]
        label_col = df.columns[-1]
    
    # Convert to HuggingFace dataset format
    from datasets import Dataset
    
    dataset_dict = {
        'text': df[text_col].tolist(),
        'label': df[label_col].tolist()
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Encode string labels if needed
    if dataset['label'][0].__class__.__name__ == 'str':
        unique_labels = sorted(set(dataset['label']))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        dataset = dataset.map(lambda x: {'label': label2id[x['label']]})
    
    # Split train/test (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    return dataset


def run_experiment(
    dataset_url: str,
    method_name: str = "paper_method",
    log_callback=None
    
):
    """
    Main training loop.
    
    Args:
        dataset_url: URL or name of dataset
        base_model: Base model name
        method_name: Name of method being tested
        output_dir: Where to save results
    """
    base_model = "bert-base-uncased"
    output_dir: str = "/root/experiment/results"

    print("="*80)
    print("TOMEA GOLDEN HARNESS")
    print("="*80)
    print(f"Method: {method_name}")
    print(f"Base Model: {base_model}")
    print(f"Dataset: {dataset_url}")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Load dataset
    print("\\n[1/6] Loading dataset...")
    dataset = load_custom_dataset(dataset_url)
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Test samples: {len(dataset['test']):,}")
    
    # 2. Setup tokenizer
    print("\\n[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    def tokenize(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512
        )
    
    dataset = dataset.map(tokenize, batched=True)
    
    # 3. Get number of labels
    num_labels = len(set(dataset['train']['label']))
    print(f"  Number of classes: {num_labels}")
    
    # 4. Load and modify model using adapter
    print(f"\\n[3/6] Loading model with {method_name}...")
    model = adapter.get_model(base_model, num_labels)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 5. Training arguments (STANDARDIZED - never change)
    print("\\n[4/6] Setting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=0.5,
        
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard
        fp16=False #torch.cuda.is_available(),  # Use mixed precision if GPU
    )
    
    # 6. Define metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
        }
    
    # 7. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # 8. Train
    print("\\n[5/6] Training...")
    print("-"*80)
    
    train_result = trainer.train()
    
    print("-"*80)
    print("Training complete!")
    
    # 9. Final evaluation
    print("\\n[6/6] Final evaluation...")
    eval_result = trainer.evaluate()
    
    # 10. Collect metrics
    end_time = time.time()
    training_time_hours = (end_time - start_time) / 3600
    
    # Get peak memory
    if torch.cuda.is_available():
        memory_peak_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        memory_peak_gb = 0.0
    
    metrics = {
        'method': method_name,
        'base_model': base_model,
        'accuracy': eval_result['eval_accuracy'],
        'f1': eval_result['eval_f1'],
        'precision': eval_result['eval_precision'],
        'recall': eval_result['eval_recall'],
        'training_time_hours': training_time_hours,
        'memory_peak_gb': memory_peak_gb,
        'trainable_parameters_millions': trainable_params / 1e6,
        'total_parameters_millions': total_params / 1e6,
        'status': 'success'
    }
    
    # 11. Save results
    print("\\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Training Time: {metrics['training_time_hours']:.2f} hours")
    print(f"Memory Peak: {metrics['memory_peak_gb']:.2f} GB")
    print(f"Trainable Params: {metrics['trainable_parameters_millions']:.2f}M")
    print("="*80)
    
    # Save as JSON for easy parsing
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\\nResults saved to {output_dir}/metrics.json")
    
    return metrics



'''


def get_harness_code() -> str:
    """Get the harness code to upload to Modal."""
    return HARNESS_TEMPLATE