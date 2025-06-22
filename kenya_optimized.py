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
    TrainingArguments,  # Ensure this is imported
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from rouge_score import rouge_scorer
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
import onnx
import onnxruntime as ort
from pathlib import Path
from datetime import datetime
import gc

# Force HuggingFace to save checkpoints in PyTorch format instead of safetensors
os.environ["HF_SAVE_FORMAT"] = "pt"

# Set up logging
os.makedirs("logs", exist_ok=True)

# Configuration
CONFIG = {
    "model_name": "google/flan-t5-base",
    "max_input_length": 286,
    "max_target_length": 154,
    "batch_size": 2,  # Fast debug
    "learning_rate": 3e-4,
    "num_train_epochs": 1,  # Fast debug
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 4,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 50,
    "seed": 42,
    "n_splits": 2,  # Fast debug
    "use_fp16": False,
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
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided to ClinicalDataset")
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
            labels = label_encoding["input_ids"].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100  # Set padding to -100 for loss
            item["labels"] = labels
        
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
        
    def save_pretrained(self, save_directory):
        """Saves prompt-tuning prefix-encoder."""
        log_message(f"Saving prompt-tuning prefix encoder to {save_directory}")
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.prefix_encoder.state_dict(), os.path.join(save_directory, "prefix_encoder.pth"))
        self.base_model.save_pretrained(save_directory)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    def __getattr__(self, name):
        """Forward missing attributes to the base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

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
        
        return outputs  # Ensure outputs is a ModelOutput with .loss

# Compute Metrics
def compute_metrics(pred, tokenizer):
    """Compute ROUGE metrics for evaluation"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Ensure predictions and labels are numpy arrays
    preds = pred.predictions
    labels = pred.label_ids
    # If preds is a tuple (e.g., (logits,)), take the first element
    if isinstance(preds, tuple):
        preds = preds[0]
    # If predictions are a list of lists, convert to numpy array
    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(labels, list):
        labels = np.array(labels)
    # If predictions are logits, take argmax
    if hasattr(preds, 'ndim') and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    # Replace -100 in labels with pad_token_id for decoding
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    labels = np.where(labels == -100, pad_token_id, labels)
    predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)
    scores = []
    for pred_str, ref_str in zip(predictions, references):
        score = scorer.score(pred_str, ref_str)
        scores.append({
            "rouge1": score["rouge1"].fmeasure,
            "rouge2": score["rouge2"].fmeasure,
            "rougeL": score["rougeL"].fmeasure,
        })
    result = {k: np.mean([s[k] for s in scores]) for k in scores[0]}
    result["rouge-combined"] = result["rouge1"] + result["rouge2"] + result["rougeL"]
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
    # FAST DEBUG: Use only first 20 rows
    df_train = df_train.iloc[:20].copy()
    df_test = df_test.iloc[:20].copy()
    
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
        logits = outputs[0]
        if not isinstance(logits, np.ndarray):
            logits = np.array(logits)
        predictions = np.argmax(logits, axis=-1)
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
    
    # Create dataset
    dataset = ClinicalDataset(train_texts, train_labels, tokenizer)
    
    # K-Fold Cross Validation
    log_message(f"Starting {CONFIG['n_splits']}-fold cross-validation")
    kf = KFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])
    all_predictions = []
    # Use indices for KFold
    indices = np.arange(len(dataset))
    try:
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            log_message(f"Training fold {fold+1}/{CONFIG['n_splits']}")
            
            # Safer per-fold model
            log_message(f"Loading model for fold {fold+1}: {CONFIG['model_name']}")
            model_fold = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
            if CONFIG["use_prompt_tuning"]:
                log_message("Applying prompt tuning...")
                model_fold = PromptTuning(model_fold, CONFIG["prefix_length"])
            model_fold.base_model.resize_token_embeddings(len(tokenizer))
            
            # Split dataset
            train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
            val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())
            
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
                eval_strategy="steps", #commented out to avoid frequent evals
                save_strategy="steps",
                save_total_limit=CONFIG["save_total_limit"],
                load_best_model_at_end=CONFIG["load_best_model_at_end"],
                metric_for_best_model=CONFIG["metric_for_best_model"],
                fp16=CONFIG["use_fp16"],
                report_to="none",
                disable_tqdm=False,
                gradient_checkpointing=True,
                save_safetensors=False
            )
            
            # Debug: Print first batch of data and check for empty/all-padding labels
            train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
            first_batch = next(iter(train_loader))
            print("\n[DEBUG] First batch input_ids:", first_batch["input_ids"][:2])
            print("[DEBUG] First batch labels:", first_batch["labels"][:2] if "labels" in first_batch else "No labels")
            if "labels" in first_batch:
                num_all_pad = (first_batch["labels"] == 0).all(dim=1).sum().item()
                print(f"[DEBUG] Number of all-padding label rows in first batch: {num_all_pad}/{first_batch['labels'].shape[0]}")
                num_empty = (first_batch["labels"] == 0).sum().item()
                print(f"[DEBUG] Total number of padding tokens in labels: {num_empty}")

            # Create trainer
            trainer = Trainer(
                model=model_fold,
                args=training_args,
                train_dataset=train_subset,
                eval_dataset=val_subset,
                data_collator=DataCollatorForSeq2Seq(tokenizer, model=model_fold),
                compute_metrics=lambda pred: compute_metrics(pred, tokenizer)
            )
            
            # Train model
            log_message("Starting training...")
            trainer.train()
            log_message("Training complete for this fold.")
            
            # Evaluate model
            log_message("Evaluating model...")
            metrics = trainer.evaluate()
            log_message(f"Fold {fold+1} Evaluation: {metrics}")
            
            # Generate predictions for this fold
            log_message(f"Generating predictions for fold {fold+1}...")
            preds = predictions.predictions
            if isinstance(preds, tuple):
                preds = preds[0]
            if isinstance(preds, list):
                preds = np.array(preds)
            pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
            
            # Store predictions for ensemble
            all_predictions.append(pred_texts)
            
            # Save model checkpoint
            model_path = f"./models/fold_{fold+1}"
            model_fold.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # Export to ONNX
            onnx_path = f"./models/fold_{fold+1}.onnx"
            export_to_onnx(model_fold, tokenizer, onnx_path)

            # Clean up memory
            del trainer
            del model_fold
            torch.cuda.empty_cache()
            gc.collect()
            log_message(f"Fold {fold+1} finished successfully.")
    except Exception as e:
        import traceback
        log_message(f"Exception occurred during cross-validation: {e}\n{traceback.format_exc()}")
        raise
    
    # Ensemble predictions from all folds
    log_message("Generating final ensemble predictions...")
    final_predictions = []
    
    # Average predictions from all folds
    test_dataset = ClinicalDataset(test_texts, tokenizer=tokenizer)
    
    # Generate predictions for test set using each fold
    fold_predictions = []
    for fold in range(CONFIG["n_splits"]):
        log_message(f"Generating predictions for fold {fold+1}...")
        
        model_path = f"./models/fold_{fold+1}"
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        if CONFIG["use_prompt_tuning"]:
            model_fold = PromptTuning(base_model, CONFIG["prefix_length"])
            prefix_encoder_path = os.path.join(model_path, "prefix_encoder.pth")
            if os.path.exists(prefix_encoder_path):
                model_fold.prefix_encoder.load_state_dict(torch.load(prefix_encoder_path))
        else:
            model_fold = base_model

        model_fold.base_model.resize_token_embeddings(len(tokenizer))
        
        # Use a simplified TrainingArguments for prediction
        predict_args = TrainingArguments(
            output_dir=f"./results_pred_fold{fold+1}",
            per_device_eval_batch_size=CONFIG["batch_size"],
            fp16=CONFIG["use_fp16"],
            disable_tqdm=True,
            report_to="none",
            save_safetensors=False
        )

        trainer = Trainer(
            model=model_fold,
            args=predict_args,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model_fold)
        )
        predictions = trainer.predict(test_dataset)
        pred_texts = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
        fold_predictions.append(pred_texts)

        # Clean up memory
        del trainer
        del model_fold
        del base_model
        torch.cuda.empty_cache()
        gc.collect()

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
    
    # Load the best model (from fold 1 as an example) for final export
    log_message("Loading model from fold 1 for final ONNX export...")
    model_path = "./models/fold_1"
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    if CONFIG["use_prompt_tuning"]:
        model_fold = PromptTuning(base_model, CONFIG["prefix_length"])
        prefix_encoder_path = os.path.join(model_path, "prefix_encoder.pth")
        if os.path.exists(prefix_encoder_path):
            model_fold.prefix_encoder.load_state_dict(torch.load(prefix_encoder_path))
    else:
        model_fold = base_model
        
    export_to_onnx(model_fold, tokenizer, final_onnx_path)
    
    log_message("Training and prediction pipeline completed successfully!")

if __name__ == "__main__":
    main()