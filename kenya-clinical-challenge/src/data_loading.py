import pandas as pd

def load_data(train_file_path, test_file_path):
    try:
        df_train = pd.read_csv(train_file_path)
        df_test = pd.read_csv(test_file_path)
        
        print(f"Shape of df_train: {df_train.shape}")
        print(f"Shape of df_test: {df_test.shape}")
        
        print("\nColumns of df_train:")
        print(df_train.columns)
        print("\nColumns of df_test:")
        print(df_test.columns)
        
        print("\nFirst 5 rows of df_train:")
        print(df_train.head())
        print("\nFirst 5 rows of df_test:")
        print(df_test.head())
        
        return df_train, df_test
    
    except FileNotFoundError:
        print("Error: One or both of the CSV files were not found.")
        return None, None
    except pd.errors.ParserError:
        print("Error: There was a problem parsing the CSV file(s). Check the file format.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

def inspect_data(df):
    if df is not None:
        print("\nData Types:")
        print(df.dtypes)
        print("\nMissing Values:")
        print(df.isnull().sum())