
# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from huggingface_hub import HfApi

# ==========================================
# CONFIG
# ==========================================

DATASET_REPO = "Kaddy4613/Predictive-Maintenance"

TRAIN_REPO = "Kaddy4613/Predictive-Maintenance-Train"

TEST_REPO = "Kaddy4613/Predictive-Maintenance-Test"

LOCAL_SAVE_PATH = "HardeepKadian_PredictiveMaintenance/data"

RANDOM_STATE = 42

TEST_SIZE = 0.2

# ==========================================
# LOAD DATASET FROM HUGGING FACE
# ==========================================

print("Loading dataset from Hugging Face...")

dataset = load_dataset(DATASET_REPO)

df = dataset["train"].to_pandas()

print("Dataset loaded successfully.")
print(f"Dataset Shape: {df.shape}")

# ==========================================
# DATA CLEANING
# ==========================================

print("\nStarting data cleaning...")

# Remove duplicate rows
duplicates = df.duplicated().sum()

print(f"Duplicate rows found: {duplicates}")

df = df.drop_duplicates()

# Handle missing values
missing_values = df.isnull().sum().sum()

print(f"Total missing values: {missing_values}")

# Fill numerical missing values with median
df = df.fillna(df.median(numeric_only=True))

print("Missing values handled.")

# ==========================================
# REMOVE UNNECESSARY COLUMNS
# ==========================================

# Example:
# df = df.drop(columns=["Unnecessary_Column"])

print("No unnecessary columns found.")

# ==========================================
# FEATURE-TARGET SPLIT
# ==========================================

X = df.drop("Engine Condition", axis=1)

y = df["Engine Condition"]

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

print("\nSplitting dataset into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ==========================================
# CREATE TRAIN & TEST DATAFRAMES
# ==========================================

train_df = X_train.copy()

train_df["Engine Condition"] = y_train

test_df = X_test.copy()

test_df["Engine Condition"] = y_test

print(f"Train Shape: {train_df.shape}")

print(f"Test Shape: {test_df.shape}")

# ==========================================
# SAVE LOCALLY
# ==========================================

print("\nSaving datasets locally...")

train_path = f"{LOCAL_SAVE_PATH}/train_data.csv"

test_path = f"{LOCAL_SAVE_PATH}/test_data.csv"

train_df.to_csv(train_path, index=False)

test_df.to_csv(test_path, index=False)

print("Datasets saved locally.")

# ==========================================
# CONVERT TO HF DATASETS
# ==========================================

train_dataset = Dataset.from_pandas(train_df)

test_dataset = Dataset.from_pandas(test_df)

# ==========================================
# UPLOAD TRAIN DATASET
# ==========================================

print("\nUploading train dataset to Hugging Face...")

train_dataset.push_to_hub(TRAIN_REPO)

print("Train dataset uploaded successfully.")

# ==========================================
# UPLOAD TEST DATASET
# ==========================================

print("\nUploading test dataset to Hugging Face...")

test_dataset.push_to_hub(TEST_REPO)

print("Test dataset uploaded successfully.")

# ==========================================
# FINAL MESSAGE
# ==========================================

print("\nData preparation completed successfully.")
