# Kenya Clinical Challenge

## Overview
The Kenya Clinical Challenge project aims to develop a machine learning model for analyzing clinical prompts and generating clinician responses. This project encompasses various stages, including data loading, exploration, cleaning, preprocessing, feature engineering, model training, evaluation, and submission.

## Project Structure
The project is organized into the following directories and files:

```
kenya-clinical-challenge
├── src
│   ├── data_loading.py        # Functions for loading datasets
│   ├── data_exploration.py     # Functions for exploring datasets
│   ├── data_cleaning.py        # Functions for cleaning datasets
│   ├── preprocessing.py         # Functions for preprocessing text data
│   ├── feature_engineering.py   # Functions for generating text embeddings
│   ├── data_splitting.py        # Functions for splitting data into train/validation sets
│   ├── model_training.py        # Functions for training the model
│   ├── model_quantization.py    # Functions for optimizing and quantizing the model
│   ├── model_evaluation.py      # Functions for evaluating the model
│   └── submission.py            # Functions for preparing submission files
├── kenya_clinical.ipynb        # Main Jupyter notebook for workflow orchestration
├── data
│   ├── train.csv                # Training dataset
│   └── test.csv                 # Test dataset
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd kenya-clinical-challenge
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Open the `kenya_clinical.ipynb` notebook to run the entire workflow.
2. The notebook will guide you through the steps of loading data, exploring it, cleaning, preprocessing, training the model, evaluating it, and preparing submissions.

## Components
- **Data Loading**: Load training and test datasets from CSV files while handling exceptions.
- **Data Exploration**: Analyze data types, unique values, and distributions for categorical features.
- **Data Cleaning**: Impute missing values and ensure consistency in categorical features.
- **Preprocessing**: Lowercase text, remove punctuation, and tokenize the text data.
- **Feature Engineering**: Generate text embeddings for clinical prompts using a pre-trained model.
- **Data Splitting**: Split the training data into training and validation sets with stratification.
- **Model Training**: Train the model using the prepared datasets with specified configurations.
- **Model Quantization**: Optimize and quantize the trained model for deployment.
- **Model Evaluation**: Evaluate the model on validation data and compute metrics like ROUGE scores.
- **Submission**: Prepare final predictions for submission and save them to a CSV file.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.