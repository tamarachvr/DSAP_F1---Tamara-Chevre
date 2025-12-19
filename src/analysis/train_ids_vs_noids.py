from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42


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
# Load data (safe, minimal, NO LEAKAGE)
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
        "variant": name,
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
    print("Loading data for IDs vs no-IDs test...")
    df, y, years = load_data()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_mask = years <= 2022
    test_mask = years == 2023

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    rows = []

    # ===============================================================
    # VERSION A — WITH IDs
    # ===============================================================
    print("\n=== RANDOM FOREST — WITH IDs ===")

    FEATURES_WITH_IDS = [
        "year",
        "round",
        "grid",
        "driverId",
        "constructorId",
        "raceId",
        "team_name",
        "driver_standings_position_pre_race",
        "driver_standings_points_pre_race",
        "constructor_standings_position_pre_race",
        "constructor_standings_points_pre_race",
        "airtemp_mean",
        "tracktemp_mean",
        "humidity_mean",
        "pressure_mean",
        "windspeed_mean",
        "rainfall_any",
    ]
    FEATURES_WITH_IDS = [c for c in FEATURES_WITH_IDS if c in df.columns]

    X_ids = df[FEATURES_WITH_IDS].copy()
    X_ids = pd.get_dummies(
        X_ids,
        columns=[c for c in ["team_name", "driverId", "constructorId", "raceId"] if c in X_ids.columns],
        drop_first=True,
    )
    X_ids = X_ids.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train_ids = X_ids.loc[train_mask]
    X_test_ids = X_ids.loc[test_mask]

    rf_ids = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    rows.append(
        evaluate(
            "Random Forest (WITH IDs)",
            "with_ids",
            rf_ids,
            X_train_ids,
            X_test_ids,
            y_train,
            y_test,
        )
    )

    # ===============================================================
    # VERSION B — NO IDs
    # ===============================================================
    print("\n=== RANDOM FOREST — NO IDs ===")

    FEATURES_NO_IDS = [
        "year",
        "round",
        "grid",
        "driver_standings_position_pre_race",
        "driver_standings_points_pre_race",
        "constructor_standings_position_pre_race",
        "constructor_standings_points_pre_race",
        "airtemp_mean",
        "tracktemp_mean",
        "humidity_mean",
        "pressure_mean",
        "windspeed_mean",
        "rainfall_any",
    ]
    FEATURES_NO_IDS = [c for c in FEATURES_NO_IDS if c in df.columns]

    X_noids = df[FEATURES_NO_IDS].copy()
    X_noids = X_noids.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train_noids = X_noids.loc[train_mask]
    X_test_noids = X_noids.loc[test_mask]

    rf_noids = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    rows.append(
        evaluate(
            "Random Forest (NO IDs)",
            "no_ids",
            rf_noids,
            X_train_noids,
            X_test_noids,
            y_train,
            y_test,
        )
    )

    # ---------------------------------------------------------------
    # Save ONE CSV (+ delta)
    # ---------------------------------------------------------------
    df_out = pd.DataFrame(rows)

    f1_with = df_out.loc[df_out["tag"] == "with_ids", "macro_f1"].values[0]
    df_out["delta_macro_f1_vs_with_ids"] = df_out["macro_f1"] - f1_with

    out_csv = RESULTS_DIR / "train_ids_vs_noids_results_2023.csv"
    df_out.to_csv(out_csv, index=False)

    print("\n================ FINAL COMPARISON =================")
    f1_no = df_out.loc[df_out["tag"] == "no_ids", "macro_f1"].values[0]
    print(f"Macro F1 WITH IDs : {f1_with:.3f}")
    print(f"Macro F1 NO IDs   : {f1_no:.3f}")
    print(f"Delta             : {f1_with - f1_no:.3f}")
    print(f"\n✅ Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()