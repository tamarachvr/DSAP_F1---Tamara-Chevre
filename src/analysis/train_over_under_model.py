import joblib
from pathlib import Path

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
# Evaluation helper
# ---------------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n================ {name} ================")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy : {acc:.3f}")
    print(f"Macro F1 : {macro_f1:.3f}")

    return model, acc, macro_f1


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

    # Safety checks
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

    gb = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    log_reg, acc_log, f1_log = evaluate_model(
        "Multinomial Logistic Regression",
        log_reg,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    rf, acc_rf, f1_rf = evaluate_model(
        "Random Forest",
        rf,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    gb, acc_gb, f1_gb = evaluate_model(
        "Gradient Boosting",
        gb,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n================ Model comparison (TEST = 2023) ================")
    print(f"Logistic Regression - Accuracy: {acc_log:.3f}, Macro F1: {f1_log:.3f}")
    print(f"Random Forest       - Accuracy: {acc_rf:.3f}, Macro F1: {f1_rf:.3f}")
    print(f"Gradient Boosting   - Accuracy: {acc_gb:.3f}, Macro F1: {f1_gb:.3f}")

    # ---------------------------------------------------------------
    # Save models
    # ---------------------------------------------------------------
    joblib.dump(log_reg, MODELS_DIR / "logistic_regression_multiclass.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest_multiclass.joblib")
    joblib.dump(gb, MODELS_DIR / "gradient_boosting_multiclass.joblib")

    print(f"\nModels saved to {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()