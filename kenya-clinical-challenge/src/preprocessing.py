def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^\\w\\s]', '', text)
        text = re.sub(r'\\s+', ' ', text)
        return text
    return ""

def tokenize_text(text):
    if isinstance(text, str):
        return word_tokenize(text)
    return []

def preprocess_dataframe(df):
    df['Prompt'] = df['Prompt'].apply(preprocess_text)
    df['Clinician'] = df['Clinician'].apply(preprocess_text)
    df['Prompt_tokens'] = df['Prompt'].apply(tokenize_text)
    df['Clinician_tokens'] = df['Clinician'].apply(tokenize_text)
    return df

def preprocess_data(train_df, test_df):
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)
    return train_df, test_df