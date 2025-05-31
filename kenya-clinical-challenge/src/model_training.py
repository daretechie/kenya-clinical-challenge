import random
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
from datasets import Dataset
from rouge_score import rouge_scorer

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_datasets(train_df, val_df):
    train_dataset = Dataset.from_pandas(train_df[["Prompt", "Clinician"]])
    val_dataset = Dataset.from_pandas(val_df[["Prompt", "Clinician"]])
    return train_dataset, val_dataset

def load_model(model_name="t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def preprocess_data(tokenizer, examples):
    inputs = [ex for ex in examples["Prompt"]]
    targets = [ex for ex in examples["Clinician"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = np.mean([scorer.score(label, pred)["rougeL"].fmeasure for label, pred in zip(decoded_labels, decoded_preds)])
    
    return {"rougeL": rouge_l}

def train_model(train_dataset, val_dataset, model_name="t5-small"):
    set_random_seed()
    tokenizer, model = load_model(model_name)
    
    tokenized_train = train_dataset.map(preprocess_data, batched=True)
    tokenized_val = val_dataset.map(preprocess_data, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        predict_with_generate=True,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_dir="./logs",
        logging_steps=50,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    return model, tokenizer

def main(train_df, val_df):
    train_dataset, val_dataset = prepare_datasets(train_df, val_df)
    model, tokenizer = train_model(train_dataset, val_dataset)