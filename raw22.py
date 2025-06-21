"""
Kenya Clinical Challenge - Optimized Solution
"""

import os
import re
import warnings
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from rouge_score import rouge_scorer
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
import wandb
import onnx
import onnxruntime as ort
from pathlib import Path
from datetime import datetime

# Set up logging
os.makedirs("logs", exist_ok=True)

# Initialize W&B for tracking
os.makedirs("wandb", exist_ok=True)
wandb.init(project="kenya-clinical-challenge")

# Configuration
CONFIG = {
    "model_name": "google/flan-t5-base",
    "max_input_length": 1024,
    "max_target_length": 512,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "num_train_epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
    "seed": 42,
    "n_splits": 5,
    "use_fp16": True,
    "output_dir": "./results",
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "rouge-combined",
    "do_lower": True,
    "early_stopping_patience": 3,
    "use_prompt_tuning": True,
    "prefix_length": 20,
    "num_beams": 4,
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95
}

# Set random seed for reproducibility
set_seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# Data paths (adjust as needed)
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test.csv'
SUBMISSION_CSV = 'submission_competitive.csv'

# Initialize log file
log_file = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
def log_message(message):
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().isoformat()}: {message}\n")
    print(message)

# Normalize text
def normalize_text(text):
    """Normalize text for consistent evaluation"""
    if pd.isna(text):
        return ""
    text = str(text)
    if CONFIG["do_lower"]:
        text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Dataset Class
class ClinicalDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=CONFIG["max_input_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if self.labels is not None:
            label_encoding = self.tokenizer(
                self.labels[idx],
                max_length=CONFIG["max_target_length"],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            item["labels"] = label_encoding["input_ids"].squeeze(0)
            
        return item

# Prompt Tuning Module
class PromptTuning(nn.Module):
    def __init__(self, base_model, prefix_length=20):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.prefix_encoder = nn.Embedding(prefix_length, base_model.get_input_embeddings().embedding_dim)
        self.init_prefix_weights()
        
    def init_prefix_weights(self):
        """Initialize prefix embeddings with random values"""
        nn.init.xavier_uniform_(self.prefix_encoder.weight)
        
    def forward(self, input_ids, attention_mask=None, labels=None, num_items_in_batch=None, **kwargs):
        # Get input embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Concatenate prefix embeddings
        batch_size = input_ids.size(0)
        prefix_embeds = self.prefix_encoder.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine prefix and input embeddings
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.prefix_length).to(attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Forward pass through base model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs

# Compute Metrics
def compute_metrics(pred, tokenizer):
    """Compute ROUGE metrics for evaluation"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    predictions = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    references = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        scores.append({
            "rouge1": score["rouge1"].fmeasure,
            "rouge2": score["rouge2"].fmeasure,
            "rougeL": score["rougeL"].fmeasure,
        })
    
    result = {k: np.mean([s[k] for s in scores]) for k in scores[0]}
    result["rouge-combined"] = result["rouge1"] + result["rouge2"] + result["rougeL"]
    
    # Log metrics to W&B
    wandb.log({
        "eval/rouge1": result["rouge1"],
        "eval/rouge2": result["rouge2"],
        "eval/rougeL": result["rougeL"],
        "eval/rouge-combined": result["rouge-combined"]
    })
    
    return result

# Data Augmentation
def augment_prompt(text):
    """Augment clinical prompts for data augmentation"""
    # Simple synonym replacement
    replacements = {
        "child": "pediatric patient",
        "kidney": "renal",
        "heart": "cardiac",
        "lungs": "pulmonary",
        "liver": "hepatic",
        "intestine": "gastrointestinal",
        "stomach": "gastric",
        "bladder": "urinary"
    }
    
    # Replace words with synonyms
    for old, new in replacements.items():
        if random.random() < 0.3:  # 30% chance to replace each word
            text = text.replace(old, new)
    
    return text

# Load and Preprocess Data
def load_data():
    """Load and preprocess the clinical data"""
    log_message("Loading and preprocessing data...")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Check if data files exist
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Data files not found: {TRAIN_CSV} and {TEST_CSV}")
    
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    log_message(f"Original train shape: {df_train.shape}")
    log_message(f"Original test shape: {df_test.shape}")
    
    # Validate required columns
    required_columns = ['Prompt', 'Clinician']
    for col in required_columns:
        if col not in df_train.columns:
            raise ValueError(f"Missing required column in train data: {col}")
    
    # Clean and normalize text
    df_train["text"] = df_train["Prompt"].apply(normalize_text)
    df_train["summary"] = df_train["Clinician"].apply(normalize_text)
    
    # Apply data augmentation
    augmented_texts = df_train["text"].apply(augment_prompt).tolist()
    augmented_summaries = df_train["summary"].tolist()
    
    df_augmented = pd.DataFrame({
        "text": augmented_texts,
        "summary": augmented_summaries
    })
    
    # Combine original and augmented data
    df_full = pd.concat([df_train[["text", "summary"]], df_augmented], axis=0, ignore_index=True)
    df_full = df_full.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)
    
    log_message(f"Cleaned train shape: {df_full.shape}")
    
    return df_full[["text", "summary"]], df_test

# ONNX Exporter
def export_to_onnx(model, tokenizer, output_path="model.onnx"):
    """Export model to ONNX format for deployment"""
    log_message("Exporting model to ONNX format...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = "A 4-year-old child presents with second-degree burns."
    inputs = tokenizer(dummy_input, return_tensors="pt", padding=True, truncation=True)
    
    # Export model to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=13,
            export_params=True
        )
    
    log_message(f"Model exported to {output_path}")
    return output_path

# ONNX Inference
class ONNXInference:
    def __init__(self, onnx_path):
        self.ort_session = ort.InferenceSession(onnx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        
    def predict(self, text):
        """Generate prediction from ONNX model"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Run inference
        outputs = self.ort_session.run(
            None,
            {
                "input_ids": inputs["input_ids"].numpy(),
                "attention_mask": inputs["attention_mask"].numpy()
            }
        )
        
        # Decode output
        predictions = np.argmax(outputs[0], axis=-1)
        return self.tokenizer.decode(predictions[0], skip_special_tokens=True)

# Main Function
def main():
    """Main training and prediction pipeline"""
    log_message("Starting Kenya Clinical Challenge solution")
    
    # Load and preprocess data
    df_train, df_test = load_data()
    train_texts = df_train["text"].tolist()
    train_labels = df_train["summary"].tolist()
    test_texts = df_test["Prompt"].apply(normalize_text).tolist()
    
    # Load tokenizer and model
    log_message(f"Loading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    log_message(f"Loading model: {CONFIG['model_name']}")
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
    
    # Add prompt tuning if enabled
    if CONFIG["use_prompt_tuning"]:
        log_message("Applying prompt tuning...")
        model = PromptTuning(model, CONFIG["prefix_length"])
    
    # Resize token embeddings (if prompt tuning is applied)
    model.base_model.resize_token_embeddings(len(tokenizer))
    
    # Create dataset
    dataset = ClinicalDataset(train_texts, train_labels, tokenizer)
    
    # K-Fold Cross Validation
    log_message(f"Starting {CONFIG['n_splits']}-fold cross-validation")
    kf = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])
    
    # Store all predictions for ensemble
    all_predictions = []
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        log_message(f"Training fold {fold+1}/{CONFIG['n_splits']}")
        
        # Split dataset
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=f"{CONFIG['output_dir']}_fold{fold+1}",
            overwrite_output_dir=True,
            num_train_epochs=CONFIG["num_train_epochs"],
            per_device_train_batch_size=CONFIG["batch_size"],
            per_device_eval_batch_size=CONFIG["batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            learning_rate=CONFIG["learning_rate"],
            warmup_steps=CONFIG["warmup_steps"],
            weight_decay=CONFIG["weight_decay"],
            logging_dir=f"./logs/fold_{fold+1}",
            logging_steps=CONFIG["logging_steps"],
            save_steps=CONFIG["save_steps"],
            eval_steps=CONFIG["eval_steps"],
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=CONFIG["save_total_limit"],
            load_best_model_at_end=CONFIG["load_best_model_at_end"],
            metric_for_best_model=CONFIG["metric_for_best_model"],
            fp16=CONFIG["use_fp16"],
            report_to="wandb",
            disable_tqdm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
            compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
            tokenizer=tokenizer
        )
        
        # Train model
        log_message("Starting training...")
        trainer.train()
        
        # Evaluate model
        log_message("Evaluating model...")
        metrics = trainer.evaluate()
        log_message(f"Fold {fold+1} Evaluation: {metrics}")
        
        # Generate predictions for this fold
        log_message(f"Generating predictions for fold {fold+1}...")
        predictions = trainer.predict(val_subset)
        pred_texts = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
        
        # Store predictions for ensemble
        all_predictions.append(pred_texts)
        
        # Save model checkpoint
        model_path = f"./models/fold_{fold+1}"
        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        # Export to ONNX
        onnx_path = f"./models/fold_{fold+1}.onnx"
        export_to_onnx(model, tokenizer, onnx_path)
    
    # Ensemble predictions from all folds
    log_message("Generating final ensemble predictions...")
    final_predictions = []
    
    # Average predictions from all folds
    test_dataset = ClinicalDataset(test_texts, tokenizer=tokenizer)
    
    # Generate predictions for test set using each fold
    fold_predictions = []
    for fold in range(CONFIG["n_splits"]):
        log_message(f"Generating predictions for fold {fold+1}...")
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="./results"),
        )
        predictions = trainer.predict(test_dataset)
        pred_texts = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
        fold_predictions.append(pred_texts)
        
    # Ensemble predictions by averaging
    for i in range(len(test_texts)):
        # Combine predictions from all folds
        combined_pred = " ".join([pred[i] for pred in fold_predictions])
        
        # Post-process
        if not combined_pred.startswith("summary "):
            combined_pred = "summary " + combined_pred
            
        final_predictions.append(combined_pred)
    
    # Create submission file
    log_message("Creating submission file...")
    submission = pd.DataFrame({
        "Master_Index": df_test["Master_Index"],
        "Clinician": final_predictions
    })
    
    # Save submission
    submission.to_csv(SUBMISSION_CSV, index=False)
    log_message(f"Submission file created: {SUBMISSION_CSV}")
    
    # Export final model to ONNX
    log_message("Exporting final model to ONNX format...")
    final_onnx_path = "final_model.onnx"
    export_to_onnx(model, tokenizer, final_onnx_path)
    
    log_message("Training and prediction pipeline completed successfully!")

if __name__ == "__main__":
    main()