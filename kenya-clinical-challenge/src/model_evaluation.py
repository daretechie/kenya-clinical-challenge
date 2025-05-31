def evaluate_model(model, tokenizer, val_dataset):
    device = 'cpu'
    model.to(device)

    val_prompts = val_dataset['Prompt'].tolist()
    val_references = val_dataset['Clinician'].tolist()

    predictions = []
    batch_size = 8

    for i in range(0, len(val_prompts), batch_size):
        batch_prompts = val_prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=2.5,
                min_length=50
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded_preds)

    return predictions, val_references


def compute_rouge_scores(references, predictions):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []

    for reference, prediction in zip(references, predictions):
        score = scorer.score(reference, prediction)
        rouge_l_scores.append(score["rougeL"].fmeasure)

    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    return average_rouge_l


def main_evaluation(model, tokenizer, val_dataset):
    predictions, references = evaluate_model(model, tokenizer, val_dataset)
    average_rouge_l = compute_rouge_scores(references, predictions)
    print(f"Average ROUGE-L F1 Score on Validation Set: {average_rouge_l:.4f}")