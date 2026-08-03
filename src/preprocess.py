"""
preprocess.py - Feature selection and preprocessing for bug prediction

This module handles:
1. Selecting which columns to use as features (and which to exclude)
2. Scaling features for models that need it (Logistic Regression, MLP)
3. Converting the target variable to the right format

WHY PREPROCESSING MATTERS:
- Different models have different assumptions about input data
- Logistic Regression and neural networks work better with scaled features
- We must be careful not to leak information from test data into training

CRITICAL: The scaler is FIT on training data only, then APPLIED to both train and test.
If we fit on test data too, we would be using future information to transform past data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# These are the JIT (Just-In-Time) defect prediction features from Kamei et al.
# We verified these exist in the actual dataset during Phase 1.
#
# WHY THESE FEATURES:
# JIT prediction uses commit-level metadata - things you know at commit time -
# NOT the actual code diff. The idea is: can we predict bug risk from how a
# commit "looks" (size, spread, author experience) without reading the code?

FEATURE_COLUMNS = [
    # ----- Size metrics -----
    # Larger changes are harder to review and more likely to contain bugs
    "la",      # Lines Added
    "ld",      # Lines Deleted

    # ----- Spread/diffusion metrics -----
    # Changes spread across many files/directories are harder to understand
    "nf",      # Number of Files modified
    "nd",      # Number of Directories modified
    "ns",      # Number of Subsystems modified
    "ent",     # Entropy - measures how spread out the changes are
               # High entropy = changes scattered across many files
               # Low entropy = changes concentrated in few files

    # ----- History metrics -----
    # Files with complex history may be more bug-prone
    "ndev",    # Number of Developers who previously modified these files
    "age",     # Average age of the modified files (in some time unit)
    "nuc",     # Number of Unique Changes to these files historically

    # ----- Experience metrics -----
    # More experienced developers may write fewer bugs
    "aexp",    # Author total EXPerience (total commits)
    "arexp",   # Author Recent EXPerience (recent commits)
    "asexp",   # Author Subsystem EXPerience (commits to this subsystem)

    # ----- Commit type -----
    # Bug-fix commits might have different characteristics
    "fix",     # Whether this commit is itself a bug fix (boolean)
]

# Columns we intentionally EXCLUDE from features:
#
# - commit_id: Just an identifier, no predictive value
# - project: Could be useful but would require encoding; keeping it simple for now
# - buggy: This is our TARGET variable, not a feature!
# - year: Used for train/test split - using it as a feature would be temporal leakage
# - author_date: Same issue - temporal information that could cause leakage

TARGET_COLUMN = "buggy"


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def extract_features_and_target(df):
    """
    Extract feature matrix X and target vector y from a DataFrame.

    Args:
        df: pandas DataFrame with the raw data

    Returns:
        X: pandas DataFrame with only the feature columns
        y: pandas Series with the target variable (as integers 0/1)
    """
    # Select only the feature columns
    X = df[FEATURE_COLUMNS].copy()

    # Convert target to integer (0/1) since it is stored as boolean (True/False)
    # Most ML libraries expect numeric targets
    y = df[TARGET_COLUMN].astype(int)

    return X, y


def check_for_missing_values(X, name="Dataset"):
    """
    Check for and report any missing values in the feature matrix.

    WHY: Missing values can cause models to fail or produce incorrect results.
    We verified in Phase 1 that this dataset has no missing values, but it is
    good practice to check defensively.

    Args:
        X: Feature matrix (DataFrame)
        name: Name to use in print statements

    Returns:
        True if there are missing values, False otherwise
    """
    missing_counts = X.isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing > 0:
        print(f"WARNING: {name} has {total_missing} missing values!")
        print(missing_counts[missing_counts > 0])
        return True
    else:
        print(f"OK: {name} has no missing values.")
        return False


def create_scaler(X_train):
    """
    Create and fit a StandardScaler on the training data.

    WHY STANDARD SCALING:
    - Logistic Regression uses gradient descent, which converges faster when
      features are on similar scales
    - Neural networks (MLP) also benefit from normalized inputs
    - Features like "la" (lines added) can range from 0 to thousands, while
      "ent" (entropy) is typically 0-5. Without scaling, large-valued features
      would dominate the learning.

    WHY FIT ON TRAINING DATA ONLY:
    - The scaler learns the mean and std of each feature
    - If we included test data, we would be using future information
    - This is a form of data leakage that inflates performance metrics
    - In production, you will not have access to future data when making predictions

    Args:
        X_train: Training feature matrix

    Returns:
        Fitted StandardScaler object
    """
    scaler = StandardScaler()

    # fit() computes mean and std from training data
    # These values are stored in scaler.mean_ and scaler.scale_
    scaler.fit(X_train)

    print(f"Scaler fitted on {X_train.shape[0]:,} training samples.")
    print(f"Feature means (rounded): {dict(zip(FEATURE_COLUMNS, scaler.mean_.round(2)))}")

    return scaler


def apply_scaler(scaler, X):
    """
    Apply a fitted scaler to transform features.

    The transformation is: X_scaled = (X - mean) / std

    After scaling:
    - Each feature has mean ~ 0 and std ~ 1 (exactly 0 and 1 for training data)
    - Test data might have slightly different mean/std since we use training stats

    Args:
        scaler: Fitted StandardScaler
        X: Feature matrix to transform

    Returns:
        Scaled feature matrix as a DataFrame (preserves column names)
    """
    X_scaled = scaler.transform(X)

    # Convert back to DataFrame to preserve column names
    # This makes debugging and interpretation easier
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def preprocess_data(train_df, test_df, scale=True):
    """
    Full preprocessing pipeline for training and test data.

    This is the main function you will call from training scripts.

    Args:
        train_df: Raw training DataFrame
        test_df: Raw test DataFrame
        scale: Whether to apply StandardScaler (True for LogReg/MLP, False for RF)

    Returns:
        X_train: Processed training features
        y_train: Training labels (0/1)
        X_test: Processed test features
        y_test: Test labels (0/1)
        scaler: The fitted scaler (or None if scale=False)
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Extract features and target
    print("\n--- Step 1: Extracting features and target ---")
    print(f"Using {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}")

    X_train, y_train = extract_features_and_target(train_df)
    X_test, y_test = extract_features_and_target(test_df)

    print(f"Training set: {X_train.shape[0]:,} samples")
    print(f"Test set: {X_test.shape[0]:,} samples")

    # Step 2: Check for missing values
    print("\n--- Step 2: Checking for missing values ---")
    check_for_missing_values(X_train, "Training data")
    check_for_missing_values(X_test, "Test data")

    # Step 3: Scale features (if requested)
    scaler = None
    if scale:
        print("\n--- Step 3: Scaling features (StandardScaler) ---")
        scaler = create_scaler(X_train)
        X_train = apply_scaler(scaler, X_train)
        X_test = apply_scaler(scaler, X_test)
        print("Scaling applied to both train and test.")
    else:
        print("\n--- Step 3: Skipping scaling ---")
        print("(Scaling not needed for tree-based models like Random Forest)")

    # Summary
    print("\n--- Preprocessing complete ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train distribution: {y_train.value_counts().to_dict()}")
    print(f"y_test distribution: {y_test.value_counts().to_dict()}")

    return X_train, y_train, X_test, y_test, scaler


# =============================================================================
# MAIN - Run this file directly to test preprocessing
# =============================================================================

if __name__ == "__main__":
    # Import data loader (relative import will not work when running directly)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_train_data, load_test_data

    print("Testing preprocessing pipeline...\n")

    # Load data
    train_df = load_train_data()
    test_df = load_test_data()

    # Run preprocessing with scaling
    X_train, y_train, X_test, y_test, scaler = preprocess_data(
        train_df, test_df, scale=True
    )

    # Show sample of scaled data
    print("\n--- Sample of scaled training data ---")
    print(X_train.head().round(3).to_string())

    # Verify scaling worked (training data should have mean~0, std~1)
    print("\n--- Verifying scaling (training data stats) ---")
    print(f"Means (should be ~0): {X_train.mean().round(6).to_dict()}")
    print(f"Stds (should be ~1): {X_train.std().round(3).to_dict()}")
