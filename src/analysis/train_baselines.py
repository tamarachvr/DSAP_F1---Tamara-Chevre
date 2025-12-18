from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_numeric_clean(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the exact crash you're seeing:
    - empty strings / whitespace -> NaN
    - force numeric (strings become NaN)
    - NaN -> 0
    """
    X = X.replace(r"^\s*$", np.nan, regex=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.fillna(0.0)


# ---------------------------------------------------------------------
# Load data (baseline-safe, minimal, NO LEAKAGE)
# ---------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    # Target + years
    if "label" not in df.columns:
        raise ValueError("Missing 'label' column in dataset.")
    if "year" not in df.columns:
        raise ValueError("Missing 'year' column in dataset.")

    y = df["label"].astype(int)
    years = df["year"].astype(int).values

    return df, y, years


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------
def evaluate(name, model, X_train, X_test, y_train, y_test):
    # IMPORTANT: clean numeric types to avoid sklearn crash on '' strings
    X_train = _to_numeric_clean(X_train)
    X_test = _to_numeric_clean(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n================ {name} ================")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy           : {acc:.3f}")
    print(f"Balanced Accuracy  : {bal_acc:.3f}")
    print(f"Macro F1           : {macro_f1:.3f}")

    return acc, bal_acc, macro_f1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading data for baselines...")
    df, y, years = load_data()

    # Temporal split
    train_mask = years <= 2022
    test_mask = years == 2023

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    # ===============================================================
    # BASELINE A — GRID (+ year)
    # ===============================================================
    print("\n=== BASELINE A: grid (+ year) ===")

    X_A = df[["grid", "year"]].copy()
    X_A = _to_numeric_clean(X_A)

    X_train_A = X_A.loc[train_mask]
    X_test_A = X_A.loc[test_mask]

    model_A = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    evaluate(
        "Baseline A (grid only)",
        model_A,
        X_train_A,
        X_test_A,
        y_train,
        y_test,
    )

    # ===============================================================
    # BASELINE B — GRID + CONSTRUCTOR STANDINGS
    # ===============================================================
    print("\n=== BASELINE B: grid + constructor standings ===")

    needed = [
        "grid",
        "constructor_standings_position_pre_race",
        "constructor_standings_points_pre_race",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Baseline B: {missing}")

    X_B = df[needed].copy()
    X_B = _to_numeric_clean(X_B)

    X_train_B = X_B.loc[train_mask]
    X_test_B = X_B.loc[test_mask]

    model_B = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    evaluate(
        "Baseline B (grid + constructor)",
        model_B,
        X_train_B,
        X_test_B,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()