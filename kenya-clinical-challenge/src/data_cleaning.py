def impute_missing_values(df_train, df_test, column):
    df_train[column] = df_train[column].fillna(df_train[column].median())
    df_test[column] = df_test[column].fillna(df_test[column].median())

def drop_rows_with_missing_values(df, column):
    df.dropna(subset=[column], inplace=True)

def ensure_consistency_in_categorical_features(df_train, df_test, categorical_cols):
    for col in categorical_cols:
        if col in df_train.columns and col in df_test.columns:
            unique_train = set(df_train[col].unique())
            unique_test = set(df_test[col].unique())
            diff = unique_train.symmetric_difference(unique_test)
            print(f"Inconsistencies in column '{col}': {diff}")
            for value in diff:
                if value.lower() in unique_train and value.lower() in unique_test:
                    df_train[col] = df_train[col].replace(value, value.lower())
                    df_test[col] = df_test[col].replace(value, value.lower())

def clean_data(df_train, df_test):
    impute_missing_values(df_train, df_test, 'Years of Experience')
    drop_rows_with_missing_values(df_train, 'DDX SNOMED')
    ensure_consistency_in_categorical_features(df_train, df_test, ['County', 'Health level', 'Nursing Competency', 'Clinical Panel'])
    return df_train, df_test