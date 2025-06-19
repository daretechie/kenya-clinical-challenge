"""
Kenya Clinical Challenge - Enhanced Optimized Solution
====================================================

Competition Constraints:
- <2GB RAM usage
- <100ms inference per vignette  
- <1B parameters
- Jetson Nano deployable
- ROUGE evaluation metric

Major Improvements:
- Fixed batch processing issues
- Enhanced model architecture selection
- Improved preprocessing pipeline
- Better memory management
- Advanced ensemble techniques
- Optimized hyperparameters
"""

import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import psutil
import re
import json
from collections import Counter, defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

# Memory monitoring
def get_memory_usage():
    """Monitor memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EnhancedKenyaClinical:
    """
    Enhanced solution for Kenya Clinical Challenge with comprehensive optimizations
    """
    
    def __init__(self, 
                 model_name: str = 'google/flan-t5-small',
                 max_input_length: int = 512,
                 max_target_length: int = 128,  # Reduced for efficiency
                 batch_size: int = 16,  # Increased for better GPU utilization
                 num_epochs: int = 6,
                 learning_rate: float = 3e-4,  # Higher LR for faster convergence
                 weight_decay: float = 0.01,
                 warmup_ratio: float = 0.1,
                 num_beams: int = 4,
                 early_stopping_patience: int = 2,
                 n_splits: int = 5,  # More splits for better validation
                 gradient_accumulation_steps: int = 2):
        
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_beams = num_beams
        self.early_stopping_patience = early_stopping_patience
        self.n_splits = n_splits
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize tokenizer with proper handling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.memory_usage = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def enhanced_preprocessing(self, text: str) -> str:
        """Enhanced text preprocessing for clinical text"""
        if pd.isna(text):
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize medical abbreviations (add more as needed)
        medical_abbrevs = {
            r'\bpt\b': 'patient',
            r'\bpts\b': 'patients', 
            r'\bhx\b': 'history',
            r'\btx\b': 'treatment',
            r'\bdx\b': 'diagnosis',
            r'\brx\b': 'prescription',
            r'\bsy\b': 'symptoms',
            r'\bc/o\b': 'complains of',
            r'\bp/e\b': 'physical examination'
        }
        
        for abbrev, full_form in medical_abbrevs.items():
            text = re.sub(abbrev, full_form, text, flags=re.IGNORECASE)
        
        return text

    def create_stratified_splits(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified splits based on text length for better validation"""
        # Create length-based strata
        lengths = df['Prompt'].str.len()
        quartiles = pd.qcut(lengths, q=4, labels=['short', 'medium', 'long', 'very_long'])
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return list(skf.split(df, quartiles))

    def preprocess_data(self, df: pd.DataFrame, is_train=True) -> Dataset:
        """Enhanced preprocessing with better prompt engineering"""
        
        # Enhanced prompt engineering
        input_texts = []
        for _, row in df.iterrows():
            prompt = self.enhanced_preprocessing(row['Prompt'])
            
            # Create more informative prompts
            enhanced_prompt = f"Summarize the following clinical case in a concise manner: {prompt}"
            input_texts.append(enhanced_prompt)
        
        data = {"input_ids": [], "attention_mask": []}
        
        if is_train:
            target_texts = [self.enhanced_preprocessing(text) for text in df['Clinician'].tolist()]
            data["labels"] = []
        
        # Tokenize all at once for efficiency
        tokenized_inputs = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,  # Don't pad here, will be done by collator
            return_tensors=None
        )
        
        data["input_ids"] = tokenized_inputs["input_ids"]
        data["attention_mask"] = tokenized_inputs["attention_mask"]
        
        if is_train:
            tokenized_targets = self.tokenizer(
                target_texts,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            data["labels"] = tokenized_targets["input_ids"]
        
        return Dataset.from_dict(data)

    def compute_metrics(self, eval_preds):
        """Enhanced metrics computation with multiple ROUGE variants"""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Handle padding properly
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute multiple ROUGE metrics
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in zip(decoded_labels, decoded_preds):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores),
            "combined_rouge": np.mean(rouge1_scores) + np.mean(rougeL_scores)
        }

    def train_fold(self, train_dataset, val_dataset, fold):
        """Enhanced training with better optimization"""
        print(f"Training fold {fold + 1}/{self.n_splits}")
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)
        
        # Calculate training steps for scheduler
        num_training_steps = len(train_dataset) // (self.batch_size * self.gradient_accumulation_steps) * self.num_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        # Enhanced training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./results_fold{fold}",
            eval_strategy="steps",
            eval_steps=50,  # More frequent evaluation
            save_strategy="steps",
            save_steps=50,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=num_warmup_steps,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="combined_rouge",
            greater_is_better=True,
            save_total_limit=2,
            seed=42 + fold,
            report_to="none",
            generation_max_length=self.max_target_length,
            generation_num_beams=self.num_beams,
            lr_scheduler_type="cosine",  # Better learning rate schedule
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            padding=True,
            max_length=self.max_input_length,
            pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        )

        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
        )

        # Train model
        trainer.train()
        
        # Generate predictions for validation
        predictions = self.generate_predictions_safe(trainer.model, val_dataset, "validation")
        
        # Clean up
        best_model = trainer.model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return best_model, predictions

    def generate_predictions_safe(self, model, dataset, split_name) -> List[str]:
        """Safe prediction generation with proper batch handling"""
        print(f"Generating predictions for {split_name}...")
        model.eval()
        predictions = []
        
        # Create data loader with proper collation
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            padding=True,
            max_length=self.max_input_length,
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=data_collator,
            shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating {split_name}"):
                # Move batch to device safely
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Generate with proper parameters
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_target_length,
                    min_length=10,  # Minimum length for meaningful summaries
                    num_beams=self.num_beams,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False,  # Deterministic for consistency
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode predictions
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(decoded)
        
        return predictions

    def weighted_ensemble_predictions(self, predictions_list: List[List[str]], 
                                    weights: Optional[List[float]] = None) -> List[str]:
        """Enhanced ensemble with weighted voting based on fold performance"""
        if weights is None:
            weights = [1.0] * len(predictions_list)
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        ensemble_predictions = []
        for i in range(len(predictions_list[0])):
            # Get predictions for this sample from all folds
            sample_preds = [pred_list[i] for pred_list in predictions_list]
            
            # For text generation, we'll use the prediction from the best performing fold
            # or implement a more sophisticated text combination strategy
            best_fold_idx = np.argmax(weights)
            ensemble_predictions.append(sample_preds[best_fold_idx])
        
        return ensemble_predictions

    def evaluate_performance(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Comprehensive performance evaluation"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores),
            "rouge1_std": np.std(rouge1_scores),
            "rouge2_std": np.std(rouge2_scores),
            "rougeL_std": np.std(rougeL_scores)
        }

    def train_and_predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Enhanced training and prediction pipeline"""
        # Preprocess test data once
        test_dataset = self.preprocess_data(test_df, is_train=False)
        
        # Create stratified splits
        splits = self.create_stratified_splits(train_df)
        
        oof_predictions = np.empty(len(train_df), dtype=object)
        test_predictions_folds = []
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            set_seed(42 + fold)
            
            train_data = train_df.iloc[train_idx].reset_index(drop=True)
            val_data = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Preprocess data for current fold
            train_dataset = self.preprocess_data(train_data)
            val_dataset = self.preprocess_data(val_data)

            # Train fold
            model, val_preds = self.train_fold(train_dataset, val_dataset, fold)
            oof_predictions[val_idx] = val_preds

            # Evaluate fold performance
            val_references = val_data["Clinician"].tolist()
            fold_score = self.evaluate_performance(val_preds, val_references)
            fold_scores.append(fold_score["rougeL"])
            print(f"Fold {fold + 1} ROUGE-L Score: {fold_score['rougeL']:.4f}")

            # Generate test predictions
            test_preds = self.generate_predictions_safe(model, test_dataset, f"test_fold_{fold}")
            test_predictions_folds.append(test_preds)

            # Memory cleanup
            del model, train_dataset, val_dataset
            gc.collect()
            torch.cuda.empty_cache()

        # Weighted ensemble based on fold performance
        weights = np.array(fold_scores)
        final_test_predictions = self.weighted_ensemble_predictions(test_predictions_folds, weights)
        
        return oof_predictions.tolist(), final_test_predictions

    def benchmark_inference(self, num_samples: int = 10) -> Dict[str, float]:
        """Benchmark inference performance with a lightweight model"""
        print("Benchmarking inference performance...")
        
        # Load a small model for benchmarking
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        
        # Create dummy data
        dummy_texts = ["Patient presents with chest pain and shortness of breath."] * num_samples
        
        # Tokenize
        inputs = self.tokenizer(
            dummy_texts,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Benchmark
        start_time = time.time()
        start_memory = get_memory_usage()
        
        with torch.no_grad():
            for i in range(num_samples):
                sample_start = time.time()
                
                outputs = model.generate(
                    input_ids=inputs["input_ids"][i:i+1],
                    attention_mask=inputs["attention_mask"][i:i+1],
                    max_length=self.max_target_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
                
                sample_time = time.time() - sample_start
                print(f"Sample {i+1}: {sample_time*1000:.2f}ms")
        
        total_time = time.time() - start_time
        end_memory = get_memory_usage()
        
        # Cleanup
        del model, inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "avg_inference_time_ms": (total_time / num_samples) * 1000,
            "memory_usage_mb": end_memory - start_memory,
            "total_time_s": total_time
        }

def main():
    """Enhanced main execution function"""
    print("=== Kenya Clinical Challenge - Enhanced Optimized Solution ===")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check memory constraints
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Data paths
    TRAIN_CSV = 'data/train.csv'
    TEST_CSV = 'data/test.csv'
    SUBMISSION_CSV = 'submission_enhanced.csv'
    
    # Verify data exists
    if not os.path.exists('data'):
        print("Error: 'data' directory not found. Please place train.csv and test.csv inside a 'data' folder.")
        return

    # Load and validate data
    print("Loading and validating data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Data validation and cleaning
    print(f"Original train shape: {train_df.shape}")
    train_df.dropna(subset=['Prompt', 'Clinician'], inplace=True)
    print(f"Cleaned train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Initialize enhanced model
    model_handler = EnhancedKenyaClinical(
        model_name='google/flan-t5-small',  # Best compromise for constraints
        max_input_length=512,
        max_target_length=128,
        batch_size=16,
        num_epochs=6,
        learning_rate=3e-4,
        n_splits=5
    )
    
    # Benchmark inference first
    benchmark_results = model_handler.benchmark_inference(num_samples=5)
    print(f"Benchmark Results:")
    print(f"  Average inference time: {benchmark_results['avg_inference_time_ms']:.2f}ms")
    print(f"  Memory usage: {benchmark_results['memory_usage_mb']:.2f}MB")
    
    if benchmark_results['avg_inference_time_ms'] > 100:
        print("⚠️  WARNING: Inference time may exceed 100ms constraint!")
    
    # Train and predict
    print("\nStarting training and prediction...")
    oof_predictions, test_predictions = model_handler.train_and_predict(train_df, test_df)
    
    # Comprehensive evaluation
    oof_references = train_df["Clinician"].tolist()
    oof_scores = model_handler.evaluate_performance(oof_predictions, oof_references)
    
    print(f"\n=== Out-of-Fold Performance ===")
    print(f"ROUGE-1: {oof_scores['rouge1']:.4f} ± {oof_scores['rouge1_std']:.4f}")
    print(f"ROUGE-2: {oof_scores['rouge2']:.4f} ± {oof_scores['rouge2_std']:.4f}")
    print(f"ROUGE-L: {oof_scores['rougeL']:.4f} ± {oof_scores['rougeL_std']:.4f}")
    
    # Create submission
    submission = pd.DataFrame({
        "Master_Index": test_df["Master_Index"],
        "Clinician": test_predictions
    })
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"\nSubmission saved: {SUBMISSION_CSV}")
    
    # Final checks
    final_memory = get_memory_usage()
    print(f"\nFinal memory usage: {final_memory:.2f} MB")
    
    if final_memory > 2000:
        print("⚠️  WARNING: Memory usage exceeds 2GB constraint!")
    else:
        print("✅ Memory usage within constraints")
    
    print("\n=== Enhanced Optimization Complete ===")

if __name__ == "__main__":
    main()