"""
data_loader.py — Functions to load the ApacheJIT dataset

This module provides functions to load the training and test datasets.
It's designed to be both:
1. Importable: other scripts can call load_train_data(), load_test_data()
2. Runnable: `python src/data_loader.py` shows dataset info for verification

WHY SEPARATE THIS INTO A MODULE:
- DRY (Don't Repeat Yourself): loading logic is in one place
- If the file format changes, we only fix it here
- Makes the training scripts cleaner — they just call load_train_data()
"""

import pandas as pd
from pathlib import Path


def get_data_dir():
    """Returns the path to the data/ directory."""
    # This file is in src/, so data/ is at ../data/
    return Path(__file__).parent.parent / "data"


def load_train_data():
    """
    Load the balanced training dataset (apachejit_train.csv).
    
    This file contains commits from 2003-2016, balanced so that roughly
    half are buggy and half are clean. Balancing the training set helps
    the model learn to recognize buggy commits without being biased toward
    the majority class.
    
    Returns:
        pandas.DataFrame with the training data
    """
    path = get_data_dir() / "apachejit_train.csv"
    
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            "Run `python scripts/download_data.py` first."
        )
    
    return pd.read_csv(path)


def load_test_data():
    """
    Load the test dataset (held-out data from the last 3 years: 2017-2019).

    WHY WE USE TIME-BASED SPLITTING (not a random split):
    The ApacheJIT dataset is split by TIME — training data is from 2003-2016,
    test data is from 2017-2019. This is intentional and critical:

    1. REALISM: In production, you train on past commits and predict future ones.
       You can't use future data to predict the past!

    2. NO DATA LEAKAGE: A random shuffle would mix future commits into training,
       artificially inflating model performance. Time-based splitting prevents this.

    3. PROPER EVALUATION: This is the standard methodology in JIT defect prediction
       research (Kamei et al., 2013). Using it makes your results comparable to
       published papers.

    WHY WE EXTRACT FROM TOTAL.CSV:
    The Zenodo record doesn't have separate test files. We create the test set
    by filtering apachejit_total.csv for years >= 2017, which gives us ~30k
    commits with real-world class imbalance (~19% buggy).

    Returns:
        pandas.DataFrame with the test data (years 2017-2019)
    """
    path = get_data_dir() / "apachejit_total.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Total dataset not found at {path}. "
            "Run `python scripts/download_data.py` first."
        )

    # Load full dataset and filter for test years
    total_df = pd.read_csv(path)

    # Training data covers 2003-2016, test data is 2017-2019
    # This matches the JIT defect prediction research methodology
    test_df = total_df[total_df["year"] >= 2017].copy()

    return test_df


def print_dataset_info(df, name="Dataset"):
    """
    Print detailed information about a DataFrame.
    
    This is useful for verifying what columns actually exist before
    writing any feature-selection code. The CLAUDE.md specifically says
    to NOT assume column names — check first!
    """
    print("=" * 70)
    print(f"{name}")
    print("=" * 70)
    
    # Shape
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Columns
    print(f"\n--- COLUMNS ({len(df.columns)}) ---")
    print(list(df.columns))
    
    # Data types and non-null counts
    print(f"\n--- INFO ---")
    print(df.info())
    
    # First few rows
    print(f"\n--- HEAD (first 5 rows) ---")
    print(df.head().to_string())
    
    # Class balance for the target variable
    if "buggy" in df.columns:
        print(f"\n--- CLASS BALANCE (buggy column) ---")
        counts = df["buggy"].value_counts()
        total = len(df)

        # Handle both boolean (True/False) and integer (0/1) formats
        # The dataset uses boolean, but we want to display it clearly
        clean_count = counts.get(False, counts.get(0, 0))
        buggy_count = counts.get(True, counts.get(1, 0))

        print(f"  Clean (False/0): {clean_count:,} ({clean_count/total*100:.1f}%)")
        print(f"  Buggy (True/1):  {buggy_count:,} ({buggy_count/total*100:.1f}%)")

        # Check if balanced (within 5% of 50/50)
        ratio = buggy_count / total
        if 0.45 <= ratio <= 0.55:
            print(f"  -> Dataset is BALANCED (roughly 50/50)")
        else:
            print(f"  -> Dataset is IMBALANCED ({ratio*100:.1f}% buggy)")
    else:
        print("\nWARNING: No 'buggy' column found!")


# This block runs only when you execute this file directly
# It won't run when you import the module
if __name__ == "__main__":
    print("Loading datasets to verify columns and structure...\n")

    try:
        # Load and display training data
        train_df = load_train_data()
        print_dataset_info(train_df, "TRAINING DATA (apachejit_train.csv, 2003-2016)")

        print("\n" + "=" * 70 + "\n")

        # Load and display test data
        test_df = load_test_data()
        print_dataset_info(test_df, "TEST DATA (from apachejit_total.csv, 2017-2019)")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run: python scripts/download_data.py")
