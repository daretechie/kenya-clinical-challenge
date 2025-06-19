"""
Kenya Clinical Challenge - Optimized Solution (Fixed)
=====================================================

Competition Constraints:
- <2GB RAM usage
- <100ms inference per vignette  
- <1B parameters
- Jetson Nano deployable
- ROUGE evaluation metric

Optimizations:
- Model quantization (INT8)
- Efficient preprocessing pipeline
- Memory-optimized inference
- Reduced model complexity
- Optimized batching strategy
"""

import os
import gc
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import psutil
import re
from collections import Counter
from tqdm import tqdm

# Core ML libraries
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback, default_data_collator
)
from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.model_selection import KFold

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

class OptimizedKenyaClinical:
    """
    Optimized solution for Kenya Clinical Challenge
    """
    
    def __init__(self, 
                 model_name: str = 'google/flan-t5-small',
                 # === FIX 1: Increased max lengths for less truncation ===
                 max_input_length: int = 512,
                 max_target_length: int = 256,
                 batch_size: int = 8, # Reduced batch size to accommodate longer sequences
                 # === FIX 2: More stable hyperparameters ===
                 num_epochs: int = 5,    
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.01,
                 num_beams: int = 3, # Increased beams for potentially better generation
                 early_stopping_patience: int = 2,
                 n_splits: int = 3):
        self.model_name = model_name
        # Store separate lengths for input and target
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_beams = num_beams
        self.early_stopping_patience = early_stopping_patience
        self.n_splits = n_splits
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.memory_usage = []

    # === FIX 3: Corrected preprocessing pipeline ===
    def preprocess_data(self, df: pd.DataFrame, is_train=True) -> Dataset:
        """
        Tokenizes data without padding. Padding will be handled dynamically 
        by the DataCollatorForSeq2Seq.
        """
        input_texts = df['Prompt'].tolist()
        data = {"input_text": input_texts}
        if is_train:
            target_texts = df['Clinician'].tolist()
            data["target_text"] = target_texts

        dataset = Dataset.from_dict(data)

        def tokenize_function(examples):
            # Tokenize inputs, only truncating
            model_inputs = self.tokenizer(
                examples["input_text"], 
                max_length=self.max_input_length, 
                truncation=True
            )
            
            if is_train:
                # Tokenize targets, only truncating. Labels are created automatically by the trainer.
                labels = self.tokenizer(
                    text_target=examples["target_text"], 
                    max_length=self.max_target_length, 
                    truncation=True
                )
                model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        tokenized = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=dataset.column_names
        )
        return tokenized
    
    # === FIX 4: Added compute_metrics function for ROUGE evaluation during training ===
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Replace -100 (token used for padding) to the actual pad token ID
        preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Simple ROUGE-L scoring
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(decoded_labels, decoded_preds)]
        
        result = {"rougeL": np.mean(scores)}
        return result
    
    def quantize_model(self, model):
        """
        Apply INT8 quantization to reduce memory usage (CPU only)
        """
        if torch.cuda.is_available():
            print("Skipping quantization: only supported on CPU.")
            return model
        print("Applying INT8 quantization...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def train_fold(self, train_dataset, val_dataset, fold):
        print(f"Training fold {fold + 1}/{self.n_splits}")
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            model.cuda()

        # === FIX 5: Updated Training Arguments for stability and better evaluation ===
        training_args = Seq2SeqTrainingArguments(
            output_dir=f"./results_fold{fold}",
            eval_strategy="epoch",      # Evaluate at the end of each epoch
            save_strategy="epoch",      # Save at the end of each epoch
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            num_train_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_steps=25,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL", # Use ROUGE score to find the best model
            greater_is_better=True,
            save_total_limit=1,
            seed=42 + fold,
            report_to="none",
            generation_max_length=self.max_target_length, # Ensure generation length is consistent
            generation_num_beams=self.num_beams,
        )
        
        # === FIX 6: Simplified Data Collator ===
        # Let the collator handle padding dynamically. It will also create decoder_input_ids
        # and replace padding in labels with -100 automatically.
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, 
            model=model
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics, # Add the metrics function
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
        )

        trainer.train()
        
        # Note: Quantization is CPU-only. This line won't affect the GPU model used for prediction below.
        # It's here for a potential final CPU deployment step.
        # quantized_model = self.quantize_model(trainer.model)

        # Use the best model saved by the trainer for predictions
        best_model = trainer.model 
        predictions = self.generate_predictions(best_model, val_dataset, "validation")

        del trainer, model; gc.collect(); torch.cuda.empty_cache()
        return best_model, predictions
    
    def generate_predictions(self, model, dataset, split_name) -> List[str]:
        print(f"Generating predictions for {split_name}...")
        model.eval()
        predictions = []
        batch_size = self.batch_size * 2 # Use eval batch size
        
        # The dataset is unpadded, so we need a collator for the dataloader
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating {split_name}"):
                # The collator already prepares the batch for the correct device if a GPU is available
                # so we can simplify this part.
                if torch.cuda.is_available():
                    batch = {k: v.to('cuda') for k, v in batch.items()}
                
                outputs = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=self.max_target_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(decoded)
        return predictions

    # ... (Rest of the class methods like ensemble_predictions, evaluate_performance, etc. can remain the same) ...
    def ensemble_predictions(self, predictions_list: List[List[str]]) -> List[str]:
        return [Counter(preds).most_common(1)[0][0] for preds in zip(*predictions_list)]

    def evaluate_performance(self, predictions: List[str], references: List[str]) -> float:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        return np.mean([scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(predictions, references)])

    def train_and_predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        test_dataset = self.preprocess_data(test_df, is_train=False)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        oof_predictions = np.empty(len(train_df), dtype=object)
        test_predictions_folds = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            set_seed(42 + fold)
            train_data, val_data = train_df.iloc[train_idx], train_df.iloc[val_idx]
            
            # Preprocess data for the current fold
            train_dataset = self.preprocess_data(train_data)
            val_dataset = self.preprocess_data(val_data)

            model, val_preds = self.train_fold(train_dataset, val_dataset, fold)
            oof_predictions[val_idx] = val_preds

            test_preds = self.generate_predictions(model, test_dataset, f"test_fold_{fold}")
            test_predictions_folds.append(test_preds)

            del model, train_dataset, val_dataset
            gc.collect(); torch.cuda.empty_cache()

        final_test_predictions = self.ensemble_predictions(test_predictions_folds)
        return oof_predictions.tolist(), final_test_predictions
    
    # ... (benchmark_inference can remain the same) ...
    def benchmark_inference(self, model, test_dataset, num_samples: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance
        """
        print("Benchmarking inference performance...")
        
        model.eval()
        
        # Sample test cases
        sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        start_time = time.time()
        start_memory = get_memory_usage()
        
        predictions = []
        
        with torch.no_grad():
            for idx in sample_indices:
                sample_start = time.time()
                
                # Get single sample
                sample = test_dataset[idx]
                input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0) # create a batch of 1
                attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0)
                
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                
                # Generate
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_target_length,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
                
                # Decode
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.append(decoded[0])
                
                sample_time = time.time() - sample_start
                print(f"Sample {idx}: {sample_time*1000:.2f}ms")
        
        total_time = time.time() - start_time
        end_memory = get_memory_usage()
        
        avg_time_per_sample = total_time / num_samples
        memory_used = end_memory - start_memory
        
        return {
            "avg_inference_time_ms": avg_time_per_sample * 1000,
            "memory_usage_mb": memory_used,
            "total_time_s": total_time
        }

def main():
    """
    Main execution function
    """
    print("=== Kenya Clinical Challenge - Optimized Solution (Fixed) ===")
    
    # Set seed for reproducibility
    set_seed(42)

    # Data paths - assuming data is in a 'data' subdirectory
    if not os.path.exists('data'):
        print("Error: 'data' directory not found. Please place train.csv and test.csv inside a 'data' folder.")
        return

    TRAIN_CSV = 'data/train.csv'
    TEST_CSV = 'data/test.csv'
    SUBMISSION_CSV = 'submission_optimized_fixed.csv'
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Simple cleanup: remove rows with missing prompts or summaries
    train_df.dropna(subset=['Prompt', 'Clinician'], inplace=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Initialize optimized model
    model_handler = OptimizedKenyaClinical()
    
    # Train and predict
    oof_predictions, test_predictions = model_handler.train_and_predict(train_df, test_df)
    
    # Evaluate OOF performance
    oof_references = train_df["Clinician"].tolist()
    oof_score = model_handler.evaluate_performance(oof_predictions, oof_references)
    print(f"OOF ROUGE-L Score: {oof_score:.4f}")
    
    # Create submission
    submission = pd.DataFrame({
        "Master_Index": test_df["Master_Index"],
        "Clinician": test_predictions
    })
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Submission saved: {SUBMISSION_CSV}")
    
    # Final memory check
    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory:.2f} MB")
    
    if final_memory > 2000:
        print("⚠️  WARNING: Memory usage exceeds 2GB constraint!")
    else:
        print("✅ Memory usage within constraints")
    
    print("=== Optimization Complete ===")

if __name__ == "__main__":
    main()