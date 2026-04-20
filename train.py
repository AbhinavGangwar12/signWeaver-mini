# =============================================================================
# SignWeaver | Script 2: Model Training
# =============================================================================
# PURPOSE:
#   Reads the landmark geometry captured by 1_collect.py, trains a Random
#   Forest classifier, evaluates it, and serialises the trained model to
#   signweaver_model.pkl for use by the live demo.
#
# USAGE:
#   Ensure gesture_data.csv contains at least 2 distinct gestures, then run:
#       python 2_train.py
#
# OUTPUTS:
#   signweaver_model.pkl  — Serialised RandomForestClassifier
# =============================================================================

import os
import pickle
import pandas as pd
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# =============================================================================
# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# =============================================================================

INPUT_CSV   = "gesture_data.csv"
OUTPUT_PKL  = "signweaver_model.pkl"

# Random Forest hyperparameters
N_ESTIMATORS  = 100   # Number of trees — good balance of speed vs accuracy
RANDOM_STATE  = 42    # Seed for reproducibility

# Train / test split ratio  (0.2 = 20 % held out for evaluation)
TEST_SIZE = 0.2

# =============================================================================
# ── STEP 1 · LOAD DATA ────────────────────────────────────────────────────────
# =============================================================================

def load_data(filepath: str):
    """
    Reads gesture_data.csv and returns feature matrix X and label vector y.

    Expected CSV layout (written by 1_collect.py):
        label   | feat_0 | feat_1 | ... | feat_62
        --------|--------|--------|-----|--------
        Hello   | 0.512  | 0.321  | ... | -0.003
        ThankYou| 0.211  | 0.654  | ... |  0.001
        ...

    Returns
    -------
    X : pd.DataFrame  shape (n_samples, 63)
    y : pd.Series     shape (n_samples,)
    class_names : list[str]  sorted unique label names
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] '{filepath}' not found.\n"
            "Run 1_collect.py first to generate training data."
        )

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError("[ERROR] The CSV file is empty. Record some gestures first.")

    label_counts = df["label"].value_counts()
    print("\n[DATA] Gesture class distribution:")
    for gesture, count in label_counts.items():
        print(f"       {gesture:<20} {count:>5} frames")

    if len(label_counts) < 2:
        raise ValueError(
            "[ERROR] Only one gesture class found. "
            "Record at least 2 different gestures before training."
        )

    # Split features and labels
    y = df["label"]
    X = df.drop(columns=["label"])

    print(f"\n[DATA] Total samples : {len(df)}")
    print(f"[DATA] Feature count : {X.shape[1]}   (21 landmarks × 3 axes)")
    print(f"[DATA] Classes found : {sorted(y.unique().tolist())}")

    return X, y, sorted(y.unique().tolist())


# =============================================================================
# ── STEP 2 · SPLIT ────────────────────────────────────────────────────────────
# =============================================================================

def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Performs a stratified 80/20 train-test split.
    Stratification ensures every class is proportionally represented in both
    splits, which matters when gesture classes have unequal sample counts.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,       # Preserve class ratios in each split
    )

    print(f"\n[SPLIT] Training samples : {len(X_train)}")
    print(f"[SPLIT] Testing  samples : {len(X_test)}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# ── STEP 3 · TRAIN ────────────────────────────────────────────────────────────
# =============================================================================

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier on the training split.

    Why Random Forest for this task?
    ─────────────────────────────────
    • Handles the 63-dimensional feature space well without scaling.
    • Inherently multi-class; no one-vs-rest wrapper needed.
    • Robust to slightly noisy landmark readings.
    • Trains and predicts in milliseconds — perfect for real-time inference.
    • Requires zero GPU; runs entirely on CPU.
    """
    print(f"\n[TRAIN] Fitting RandomForestClassifier "
          f"(n_estimators={N_ESTIMATORS}, random_state={RANDOM_STATE}) ...")

    clf = RandomForestClassifier(
        n_estimators  = N_ESTIMATORS,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,   # Use all available CPU cores
        class_weight  = "balanced",  # Compensates for unequal class sizes
    )
    clf.fit(X_train, y_train)
    print("[TRAIN] Training complete ✓")
    return clf


# =============================================================================
# ── STEP 4 · EVALUATE ─────────────────────────────────────────────────────────
# =============================================================================

def evaluate_model(
    clf: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list,
) -> None:
    """
    Prints accuracy, per-class precision / recall / F1, and a confusion matrix.
    """
    y_pred    = clf.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Test Accuracy : {accuracy * 100:.2f}%")
    print("=" * 60)

    print("\n[REPORT] Per-class metrics:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # ── Confusion matrix (text-based, no matplotlib dependency) ─────────────
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    print("[REPORT] Confusion matrix (rows = actual, cols = predicted):")

    # Header row
    col_width = max(len(name) for name in class_names) + 2
    header    = " " * col_width + "".join(f"{n:>{col_width}}" for n in class_names)
    print(header)

    for i, row_name in enumerate(class_names):
        row = f"{row_name:>{col_width}}" + "".join(
            f"{val:>{col_width}}" for val in cm[i]
        )
        print(row)

    print()

    # Hackathon tip
    if accuracy < 0.85:
        print("[TIP] Accuracy below 85%. Try recording more frames per gesture,")
        print("      or ensure your gestures look distinct from each other.")
    else:
        print("[✓] Model looks solid. Ready for the live demo!")


# =============================================================================
# ── STEP 5 · SAVE MODEL ───────────────────────────────────────────────────────
# =============================================================================

def save_model(clf: RandomForestClassifier, filepath: str) -> None:
    """
    Serialises the trained classifier to disk using pickle so that
    3_demo.py can load it without retraining on every run.
    """
    with open(filepath, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n[SAVE] Model saved to: {os.path.abspath(filepath)}")


# =============================================================================
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# =============================================================================

def main():
    print("=" * 60)
    print("  SignWeaver — Model Trainer")
    print("=" * 60)

    # 1. Load
    X, y, class_names = load_data(INPUT_CSV)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Train
    clf = train_model(X_train, y_train)

    # 4. Evaluate
    evaluate_model(clf, X_test, y_test, class_names)

    # 5. Save
    save_model(clf, OUTPUT_PKL)

    print("\n[DONE] Run  python 3_demo.py  to start the live translator.\n")


if __name__ == "__main__":
    main()