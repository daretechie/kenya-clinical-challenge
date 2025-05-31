def explore_data(df_train, df_test):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Examine data types
    print("Data Types in df_train:")
    print(df_train.dtypes)
    print("\nData Types in df_test:")
    print(df_test.dtypes)

    # Investigate unique values and distributions for categorical features
    categorical_cols = ['County', 'Health level', 'Years of Experience']
    for col in categorical_cols:
        print(f"\nUnique values in {col} (df_train):")
        print(df_train[col].unique())
        print(f"\nUnique values in {col} (df_test):")
        print(df_test[col].unique())

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.countplot(x=col, data=df_train)
        plt.title(f'Distribution of {col} (Train)')
        plt.xticks(rotation=45, ha='right')
        plt.subplot(1, 2, 2)
        sns.countplot(x=col, data=df_test)
        plt.title(f'Distribution of {col} (Test)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Analyze text data length
    def text_length(text):
        if isinstance(text, str):
            return len(text.split())
        return 0

    df_train['Prompt_Length'] = df_train['Prompt'].apply(text_length)
    df_train['Clinician_Length'] = df_train['Clinician'].apply(text_length)

    print("\nDescriptive statistics for Prompt Length (df_train):")
    print(df_train['Prompt_Length'].describe())
    plt.figure(figsize=(10, 5))
    plt.hist(df_train['Prompt_Length'], bins=20)
    plt.xlabel("Prompt Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prompt Lengths (df_train)")
    plt.show()

    print("\nDescriptive statistics for Clinician Length (df_train):")
    print(df_train['Clinician_Length'].describe())
    plt.figure(figsize=(10, 5))
    plt.hist(df_train['Clinician_Length'], bins=20)
    plt.xlabel("Clinician Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Clinician Lengths (df_train)")
    plt.show()

    # Check for missing values
    print("\nMissing Values in df_train:")
    print(df_train.isnull().sum())
    print("\nMissing Values in df_test:")
    print(df_test.isnull().sum())

    # Analyze 'Clinician' column
    print("\nAverage Clinician Response Length:")
    print(df_train['Clinician_Length'].mean())

    # Compare Prompt Length distributions (train vs. test)
    df_test['Prompt_Length'] = df_test['Prompt'].apply(text_length)

    plt.figure(figsize=(10, 5))
    plt.hist(df_train['Prompt_Length'], bins=20, alpha=0.5, label='Train')
    plt.hist(df_test['Prompt_Length'], bins=20, alpha=0.5, label='Test')
    plt.xlabel("Prompt Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prompt Lengths (Train vs. Test)")
    plt.legend(loc='upper right')
    plt.show()