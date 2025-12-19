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
RESULTS_DIR = Path("results")
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_numeric_clean(X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace(r"^\s*$", np.nan, regex=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.fillna(0.0)


def _flatten_report_dict(report: dict) -> dict:
    out = {}
    for cls in ["0", "1", "2"]:
        d = report.get(cls, {})
        out[f"precision_{cls}"] = d.get("precision", np.nan)
        out[f"recall_{cls}"] = d.get("recall", np.nan)
        out[f"f1_{cls}"] = d.get("f1-score", np.nan)
        out[f"support_{cls}"] = d.get("support", np.nan)

    out["report_accuracy"] = report.get("accuracy", np.nan)

    for avg_key, prefix in [("macro avg", "macro"), ("weighted avg", "weighted")]:
        d = report.get(avg_key, {})
        out[f"{prefix}_precision"] = d.get("precision", np.nan)
        out[f"{prefix}_recall"] = d.get("recall", np.nan)
        out[f"{prefix}_f1"] = d.get("f1-score", np.nan)
        out[f"{prefix}_support"] = d.get("support", np.nan)
    return out


def _flatten_cm(cm: np.ndarray) -> dict:
    out = {}
    for i in range(3):
        for j in range(3):
            out[f"cm_{i}{j}"] = int(cm[i, j])
    return out


# ---------------------------------------------------------------------
# Load data (baseline-safe, minimal, NO LEAKAGE)
# ---------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

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
def evaluate(name, tag, model, X_train, X_test, y_train, y_test):
    X_train = _to_numeric_clean(X_train)
    X_test = _to_numeric_clean(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    labels = [0, 1, 2]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True, labels=labels)

    print(f"\n================ {name} ================")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy           : {acc:.3f}")
    print(f"Balanced Accuracy  : {bal_acc:.3f}")
    print(f"Macro F1           : {macro_f1:.3f}")

    row = {
        "baseline": name,
        "tag": tag,
        "test_year": 2023,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "random_state": RANDOM_STATE,
    }
    row.update(_flatten_cm(cm))
    row.update(_flatten_report_dict(report_dict))

    return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading data for baselines...")
    df, y, years = load_data()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_mask = years <= 2022
    test_mask = years == 2023

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    rows = []

    # ===============================================================
    # BASELINE A — GRID (+ year)
    # ===============================================================
    print("\n=== BASELINE A: grid (+ year) ===")

    X_A = df[["grid", "year"]].copy()
    X_train_A = X_A.loc[train_mask]
    X_test_A = X_A.loc[test_mask]

    model_A = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    rows.append(
        evaluate(
            "Baseline A (grid + year)",
            "A_grid_year",
            model_A,
            X_train_A,
            X_test_A,
            y_train,
            y_test,
        )
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
    X_train_B = X_B.loc[train_mask]
    X_test_B = X_B.loc[test_mask]

    model_B = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    rows.append(
        evaluate(
            "Baseline B (grid + constructor)",
            "B_grid_constructor",
            model_B,
            X_train_B,
            X_test_B,
            y_train,
            y_test,
        )
    )

    # ---------------------------------------------------------------
    # Save ONE CSV
    # ---------------------------------------------------------------
    out_csv = RESULTS_DIR / "train_baselines_results_2023.csv"
    df_out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()