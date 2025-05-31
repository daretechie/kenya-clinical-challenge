def identify_values_to_remove(df, column):
    value_counts = df[column].value_counts()
    return value_counts[value_counts == 1].index

def filter_dataframe(df, column, values_to_remove):
    return df[~df[column].isin(values_to_remove)]

def split_data(df, target_column, test_size=0.2, stratify_column=None, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(
        df,
        df[target_column],
        test_size=test_size,
        stratify=df[stratify_column] if stratify_column else None,
        random_state=random_state
    )

def print_shapes(df_train_split, df_val_split, X_train, X_val):
    print(f"Shape of df_train_split: {df_train_split.shape}")
    print(f"Shape of df_val_split: {df_val_split.shape}")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_val: {X_val.shape}")

def main_split_data(df, target_column='Years of Experience', embeddings_column='Prompt_embeddings_train'):
    values_to_remove = identify_values_to_remove(df, target_column)
    df_filtered = filter_dataframe(df, target_column, values_to_remove)
    df_train_split, df_val_split, X_train, X_val = split_data(df_filtered, embeddings_column, stratify_column=target_column)
    print_shapes(df_train_split, df_val_split, X_train, X_val)
    return df_train_split, df_val_split, X_train, X_val