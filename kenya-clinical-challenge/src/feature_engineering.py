def generate_embeddings(model, prompts):
    try:
        embeddings = model.encode(prompts, convert_to_tensor=True)
        return embeddings.tolist()
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def process_feature_engineering(df_train, df_test):
    from sentence_transformers import SentenceTransformer

    model_name = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)

    df_train['Prompt_embeddings_train'] = generate_embeddings(model, df_train['Prompt'].tolist())
    df_test['Prompt_embeddings_test'] = generate_embeddings(model, df_test['Prompt'].tolist())

    return df_train, df_test

def main():
    import pandas as pd

    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')

    df_train, df_test = process_feature_engineering(df_train, df_test)

    print("Feature engineering completed.")
    return df_train, df_test

if __name__ == "__main__":
    main()