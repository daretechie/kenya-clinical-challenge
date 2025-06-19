"""
Kenya Clinical Challenge - Fixed Competitive Solution
==================================================

Key Fixes:
- Fixed model training issues (proper labels, loss computation)
- Reduced memory usage to <2GB
- Optimized for <100ms inference
- Better preprocessing and prompt engineering
- Improved model architecture choices
- Fixed gradient computation issues
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
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

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

class CompetitiveKenyaClinical:
    """
    Fixed competitive solution for Kenya Clinical Challenge
    """
    
    def __init__(self, 
                 model_name: str = 'google/flan-t5-small',
                 max_input_length: int = 384,  # Reduced for memory
                 max_target_length: int = 64,   # Reduced for speed
                 batch_size: int = 8,           # Reduced for memory
                 num_epochs: int = 4,           # Reduced for efficiency
                 learning_rate: float = 5e-4,   # Better learning rate
                 weight_decay: float = 0.01,
                 warmup_ratio: float = 0.1,
                 num_beams: int = 2,            # Reduced for speed
                 early_stopping_patience: int = 2,
                 n_splits: int = 3):            # Reduced for efficiency
        
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
        
        # Initialize tokenizer with proper handling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def enhanced_preprocessing(self, text: str) -> str:
        """Enhanced text preprocessing for clinical text"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.,;:()-]', '', text)
        
        # Standardize medical abbreviations
        medical_abbrevs = {
            r'\bpt\b': 'patient',
            r'\bpts\b': 'patients', 
            r'\bhx\b': 'history',
            r'\btx\b': 'treatment',
            r'\bdx\b': 'diagnosis',
            r'\brx\b': 'prescription',
            r'\bsy\b': 'symptoms',
            r'\bc/o\b': 'complains of',
            r'\bp/e\b': 'physical examination',
            r'\by/o\b': 'year old',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without'
        }
        
        for abbrev, full_form in medical_abbrevs.items():
            text = re.sub(abbrev, full_form, text, flags=re.IGNORECASE)
        
        return text[:500]  # Truncate for memory efficiency

    def create_stratified_splits(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified splits based on text length"""
        lengths = df['Prompt'].str.len()
        quartiles = pd.qcut(lengths, q=3, labels=['short', 'medium', 'long'])  # 3 quartiles for 3 splits
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return list(skf.split(df, quartiles))

    def preprocess_data(self, df: pd.DataFrame, is_train=True) -> Dataset:
        """Fixed preprocessing with proper label handling"""
        
        # Create better prompts
        input_texts = []
        for _, row in df.iterrows():
            prompt = self.enhanced_preprocessing(row['Prompt'])
            
            # More concise prompt for better performance
            enhanced_prompt = f"Summarize this clinical case: {prompt}"
            input_texts.append(enhanced_prompt)
        
        # Tokenize inputs
        tokenized_inputs = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        data = {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"]
        }
        
        if is_train:
            # CRITICAL FIX: Proper target preprocessing
            target_texts = []
            for target in df['Clinician'].tolist():
                cleaned_target = self.enhanced_preprocessing(target)
                if not cleaned_target.strip():  # Handle empty targets
                    cleaned_target = "No summary available."
                target_texts.append(cleaned_target)
            
            # Tokenize targets properly
            tokenized_targets = self.tokenizer(
                target_texts,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
                return_tensors=None
            )
            
            # CRITICAL FIX: Proper label assignment
            data["labels"] = tokenized_targets["input_ids"]
        
        return Dataset.from_dict(data)

    def compute_metrics(self, eval_preds):
        """Fixed metrics computation"""
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Decode predictions - handle -100 labels properly
        decoded_preds = []
        for pred in preds:
            # Remove padding tokens
            pred_clean = [token for token in pred if token != self.tokenizer.pad_token_id]
            decoded_pred = self.tokenizer.decode(pred_clean, skip_special_tokens=True)
            decoded_preds.append(decoded_pred)
        
        # Decode labels - handle -100 properly
        decoded_labels = []
        for label in labels:
            # Replace -100 with pad token for decoding
            label_clean = [token if token != -100 else self.tokenizer.pad_token_id for token in label]
            label_clean = [token for token in label_clean if token != self.tokenizer.pad_token_id]
            decoded_label = self.tokenizer.decode(label_clean, skip_special_tokens=True)
            decoded_labels.append(decoded_label)
        
        # Compute ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in zip(decoded_labels, decoded_preds):
            if not ref.strip():  # Handle empty references
                ref = "No summary"
            if not pred.strip():  # Handle empty predictions
                pred = "No summary"
                
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
        """Fixed training with proper model initialization"""
        print(f"Training fold {fold + 1}/{self.n_splits}")
        
        # Load model fresh for each fold
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)
        
        # CRITICAL FIX: Ensure model is in training mode
        model.train()
        
        # Verify model parameters are trainable
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Fixed training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./results_fold{fold}",
            eval_strategy="epoch",  # Evaluate each epoch
            save_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=True,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",  # Use rouge1 as primary metric
            greater_is_better=True,
            save_total_limit=1,  # Save space
            seed=42 + fold,
            report_to="none",
            generation_max_length=self.max_target_length,
            generation_num_beams=self.num_beams,
            remove_unused_columns=False,
            # CRITICAL FIX: Proper gradient settings
            gradient_checkpointing=True,  # Save memory
            dataloader_num_workers=0,     # Avoid multiprocessing issues
        )
        
        # Data collator with proper padding
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,  # CRITICAL: Use -100 for ignored tokens
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
        predictions = self.generate_predictions_optimized(trainer.model, val_dataset)
        
        # Clean up
        best_model = trainer.model
        del trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        return best_model, predictions

    def generate_predictions_optimized(self, model, dataset) -> List[str]:
        """Optimized prediction generation for speed"""
        model.eval()
        predictions = []
        
        # Process in smaller batches for memory efficiency
        effective_batch_size = 4  # Smaller for memory
        
        for i in tqdm(range(0, len(dataset), effective_batch_size), desc="Generating predictions"):
            batch_data = dataset[i:i+effective_batch_size]
            
            # Handle single sample vs batch
            if not isinstance(batch_data['input_ids'][0], list):
                batch_data = {k: [v] for k, v in batch_data.items()}
            
            # Convert to tensors
            input_ids = torch.tensor(batch_data['input_ids']).to(self.device)
            attention_mask = torch.tensor(batch_data['attention_mask']).to(self.device)
            
            with torch.no_grad():
                # Optimized generation for speed
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_target_length,
                    min_length=5,
                    num_beams=self.num_beams,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2  # Prevent repetition
                )
                
                # Decode predictions
                batch_predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(batch_predictions)
        
        return predictions

    def ensemble_predictions(self, predictions_list: List[List[str]], 
                           weights: Optional[List[float]] = None) -> List[str]:
        """Simple ensemble - use best performing fold"""
        if weights is None:
            weights = [1.0] * len(predictions_list)
        
        # Use predictions from best fold
        best_fold_idx = np.argmax(weights)
        return predictions_list[best_fold_idx]

    def evaluate_performance(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Performance evaluation"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            if not ref.strip():
                ref = "No summary"
            if not pred.strip():
                pred = "No summary"
                
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
        """Main training and prediction pipeline"""
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
            fold_scores.append(fold_score["rouge1"])  # Use rouge1 for weighting
            print(f"Fold {fold + 1} ROUGE-1 Score: {fold_score['rouge1']:.4f}")

            # Generate test predictions
            test_preds = self.generate_predictions_optimized(model, test_dataset)
            test_predictions_folds.append(test_preds)

            # Memory cleanup
            del model, train_dataset, val_dataset
            gc.collect()
            torch.cuda.empty_cache()

        # Ensemble predictions
        final_test_predictions = self.ensemble_predictions(test_predictions_folds, fold_scores)
        
        return oof_predictions.tolist(), final_test_predictions

    def benchmark_inference(self, num_samples: int = 10) -> Dict[str, float]:
        """Benchmark inference performance"""
        print("Benchmarking inference performance...")
        
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()
        
        # Single sample for realistic benchmarking
        dummy_text = "Patient presents with chest pain and shortness of breath."
        
        # Tokenize
        inputs = self.tokenizer(
            dummy_text,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(3):
                _ = model.generate(**inputs, max_length=self.max_target_length, num_beams=self.num_beams)
        
        # Benchmark
        times = []
        start_memory = get_memory_usage()
        
        with torch.no_grad():
            for i in range(num_samples):
                start_time = time.time()
                
                outputs = model.generate(
                    **inputs,
                    max_length=self.max_target_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
                
                sample_time = (time.time() - start_time) * 1000
                times.append(sample_time)
                print(f"Sample {i+1}: {sample_time:.2f}ms")
        
        end_memory = get_memory_usage()
        
        # Cleanup
        del model, inputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "avg_inference_time_ms": np.mean(times),
            "memory_usage_mb": end_memory - start_memory,
            "max_time_ms": np.max(times),
            "min_time_ms": np.min(times)
        }

def main():
    """Main execution function"""
    print("=== Kenya Clinical Challenge - Fixed Competitive Solution ===")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Check memory constraints
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")

    # Data paths (adjust as needed)
    TRAIN_CSV = 'data/train.csv'
    TEST_CSV = 'data/test.csv'
    SUBMISSION_CSV = 'submission_competitive.csv'
    
    # Create dummy data for testing if files don't exist
    if not os.path.exists('data'):
        print("Creating dummy data for testing...")
        os.makedirs('data', exist_ok=True)
        
        # Create dummy training data
        dummy_train = pd.DataFrame({
            'Prompt': [f"Patient {i} presents with various symptoms including pain and discomfort. Medical history shows previous treatments." for i in range(100)],
            'Clinician': [f"Patient {i} summary: symptoms and treatment plan." for i in range(100)]
        })
        dummy_train.to_csv(TRAIN_CSV, index=False)
        
        # Create dummy test data
        dummy_test = pd.DataFrame({
            'Master_Index': range(25),
            'Prompt': [f"Test patient {i} case description with medical details." for i in range(25)]
        })
        dummy_test.to_csv(TEST_CSV, index=False)
        print("Dummy data created for testing")

    # Load and validate data
    print("Loading and validating data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Data validation and cleaning
    print(f"Original train shape: {train_df.shape}")
    train_df.dropna(subset=['Prompt', 'Clinician'], inplace=True)
    print(f"Cleaned train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Initialize model with optimized parameters
    model_handler = CompetitiveKenyaClinical(
        model_name='google/flan-t5-small',  # Good balance for constraints
        max_input_length=384,   # Reduced for memory
        max_target_length=64,   # Reduced for speed
        batch_size=8,          # Reduced for memory
        num_epochs=4,          # Reduced for efficiency
        learning_rate=5e-4,    # Better learning rate
        n_splits=3             # Reduced for efficiency
    )
    
    # Benchmark inference
    benchmark_results = model_handler.benchmark_inference(num_samples=5)
    print(f"\nBenchmark Results:")
    print(f"  Average inference time: {benchmark_results['avg_inference_time_ms']:.2f}ms")
    print(f"  Max inference time: {benchmark_results['max_time_ms']:.2f}ms")
    print(f"  Memory usage: {benchmark_results['memory_usage_mb']:.2f}MB")
    
    if benchmark_results['avg_inference_time_ms'] > 100:
        print("⚠️  WARNING: Average inference time may exceed 100ms constraint!")
    if benchmark_results['max_time_ms'] > 100:
        print("⚠️  WARNING: Max inference time exceeds 100ms constraint!")
    
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
    
    print("\n=== Fixed Competitive Solution Complete ===")

if __name__ == "__main__":
    main()