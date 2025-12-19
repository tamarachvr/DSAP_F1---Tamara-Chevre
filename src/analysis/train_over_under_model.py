import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RANDOM_STATE = 42  # for full reproducibility (prof requirement)


# ---------------------------------------------------------------------
# Load data and build X / y (NO LEAKAGE)
# ---------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    # ---------------------------------------------------------------
    # Clean rainfall_any
    # ---------------------------------------------------------------
    if "rainfall_any" in df.columns:
        df["rainfall_any"] = (
            df["rainfall_any"]
            .astype(str)
            .str.strip()
            .replace(
                {
                    "True": 1,
                    "False": 0,
                    "1": 1,
                    "0": 0,
                    "": 0,
                    "nan": 0,
                    "NaN": 0,
                    None: 0,
                }
            )
            .astype(float)
            .fillna(0)
            .astype(int)
        )

    # ---------------------------------------------------------------
    # Feature engineering (PRE-RACE ONLY)
    # ---------------------------------------------------------------
    # 1) grid_norm
    if "grid" in df.columns:
        max_grid = df["grid"].replace(0, pd.NA).max()
        df["grid_norm"] = df["grid"] / max_grid if pd.notna(max_grid) else 0.0

    # 2) team_avg_grid (by year + constructor)
    if {"year", "constructorId", "grid"}.issubset(df.columns):
        df["team_avg_grid"] = (
            df.groupby(["year", "constructorId"])["grid"].transform("mean")
        )

    # 3) teammate_grid
    if {"raceId", "constructorId", "grid"}.issubset(df.columns):

        def teammate_grid(series):
            if len(series) <= 1:
                return pd.Series([series.iloc[0]] * len(series), index=series.index)
            return (series.sum() - series) / (len(series) - 1)

        df["teammate_grid"] = (
            df.groupby(["raceId", "constructorId"])["grid"]
            .transform(teammate_grid)
            .fillna(df["grid"])
        )

    # ---------------------------------------------------------------
    # Target
    # ---------------------------------------------------------------
    if "label" not in df.columns:
        raise ValueError("Missing target column 'label'")

    y = df["label"].astype(int)

    # ---------------------------------------------------------------
    # SAFE FEATURE SELECTION (WHITELIST)
    # ---------------------------------------------------------------
    FEATURE_COLS = [
        "year",
        "round",
        "raceId",
        "driverId",
        "constructorId",
        "team_name",
        "grid",
        "grid_norm",
        "team_avg_grid",
        "teammate_grid",
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

    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
    X = df[FEATURE_COLS].copy()

    # ---------------------------------------------------------------
    # One-hot encoding for categorical variables
    # ---------------------------------------------------------------
    categorical_cols = [
        c
        for c in ["team_name", "driverId", "constructorId", "raceId"]
        if c in X.columns
    ]

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Final numeric safety
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    years = df["year"].values  # needed for temporal split
    return X, y, years


# ---------------------------------------------------------------------
# Helpers: flatten report + confusion matrix into readable columns
# ---------------------------------------------------------------------
def _flatten_report_dict(report: dict) -> dict:
    """
    classification_report(output_dict=True) -> flat columns (for classes 0/1/2 + averages)
    """
    out = {}
    for cls in ["0", "1", "2"]:
        d = report.get(cls, {})
        out[f"precision_{cls}"] = d.get("precision", np.nan)
        out[f"recall_{cls}"] = d.get("recall", np.nan)
        out[f"f1_{cls}"] = d.get("f1-score", np.nan)
        out[f"support_{cls}"] = d.get("support", np.nan)

    # accuracy is sometimes a float in dict
    out["report_accuracy"] = report.get("accuracy", np.nan)

    for avg_key, prefix in [
        ("macro avg", "macro"),
        ("weighted avg", "weighted"),
    ]:
        d = report.get(avg_key, {})
        out[f"{prefix}_precision"] = d.get("precision", np.nan)
        out[f"{prefix}_recall"] = d.get("recall", np.nan)
        out[f"{prefix}_f1"] = d.get("f1-score", np.nan)
        out[f"{prefix}_support"] = d.get("support", np.nan)

    return out


def _flatten_cm(cm: np.ndarray) -> dict:
    """
    3x3 confusion matrix -> cm_00..cm_22
    """
    out = {}
    for i in range(3):
        for j in range(3):
            out[f"cm_{i}{j}"] = int(cm[i, j])
    return out


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------
def evaluate_model(name, tag, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # Force consistent 3x3 order for your multiclass labels
    labels = [0, 1, 2]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True, labels=labels)

    print(f"\n================ {name} ================")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy    : {acc:.3f}")
    print(f"Macro F1    : {macro_f1:.3f}")
    print(f"Weighted F1 : {weighted_f1:.3f}")

    row = {
        "model": name,
        "tag": tag,
        "test_year": 2023,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
        "random_state": RANDOM_STATE,
    }
    row.update(_flatten_cm(cm))
    row.update(_flatten_report_dict(report_dict))

    return model, row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading data...")
    X, y, years = load_data()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # TEMPORAL BACKTESTING SPLIT (deterministic)
    # Train: 2021–2022
    # Test : 2023
    # ---------------------------------------------------------------
    train_mask = years <= 2022
    test_mask = years == 2023

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print("\nBacktesting mode: temporal split")
    print(f"Train set (2021–2022): {X_train.shape[0]} rows")
    print(f"Test set  (2023): {X_test.shape[0]} rows")

    assert X_train.shape[0] > 0, "Train set is empty"
    assert X_test.shape[0] > 0, "Test set is empty"

    # ---------------------------------------------------------------
    # Models (fully reproducible)
    # ---------------------------------------------------------------
    log_reg = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=4000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE)

    rows = []

    log_reg, r1 = evaluate_model("Multinomial Logistic Regression", "lr", log_reg, X_train, X_test, y_train, y_test)
    rows.append(r1)

    rf, r2 = evaluate_model("Random Forest", "rf", rf, X_train, X_test, y_train, y_test)
    rows.append(r2)

    gb, r3 = evaluate_model("Gradient Boosting", "gb", gb, X_train, X_test, y_train, y_test)
    rows.append(r3)

    # ---------------------------------------------------------------
    # Save ONE CSV (consultable)
    # ---------------------------------------------------------------
    out_csv = RESULTS_DIR / "train_over_under_model_results_2023.csv"
    df_out = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ Saved: {out_csv.resolve()}")

    # ---------------------------------------------------------------
    # Save models (still needed for downstream scripts)
    # ---------------------------------------------------------------
    joblib.dump(log_reg, MODELS_DIR / "logistic_regression_multiclass.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest_multiclass.joblib")
    joblib.dump(gb, MODELS_DIR / "gradient_boosting_multiclass.joblib")

    print(f"\nModels saved to {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()