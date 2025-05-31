def prepare_submission_data(test_data, predictions):
    submission_df = pd.DataFrame({
        'Master_Index': test_data['Master_Index'],
        'Clinician': predictions
    })
    return submission_df

def clean_submission_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def save_submission_file(submission_df, filename='submission.csv'):
    submission_df['Clinician'] = submission_df['Clinician'].apply(clean_submission_text)
    submission_df.to_csv(filename, index=False)
    print(f"Submission file '{filename}' created.")

def generate_predictions(model, tokenizer, test_prompts, batch_size=8):
    predictions = []
    for i in range(0, len(test_prompts), batch_size):
        batch_prompts = test_prompts[i:i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=256)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded_preds)
    return predictions

def main(model, tokenizer, test_data):
    test_prompts = test_data['Prompt'].tolist()
    test_predictions = generate_predictions(model, tokenizer, test_prompts)
    submission_df = prepare_submission_data(test_data, test_predictions)
    save_submission_file(submission_df)