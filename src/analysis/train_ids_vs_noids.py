from pathlib import Path

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
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Load data (safe, minimal, NO LEAKAGE)
# ---------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    y = df["label"].astype(int)
    years = df["year"].astype(int).values

    return df, y, years


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------
def evaluate(name, model, X_train, X_test, y_train, y_test):
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

    return macro_f1


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("Loading data for IDs vs no-IDs test...")
    df, y, years = load_data()

    # Temporal split
    train_mask = years <= 2022
    test_mask = years == 2023

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

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
        columns=["team_name", "driverId", "constructorId", "raceId"],
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

    f1_ids = evaluate(
        "Random Forest (WITH IDs)",
        rf_ids,
        X_train_ids,
        X_test_ids,
        y_train,
        y_test,
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

    f1_noids = evaluate(
        "Random Forest (NO IDs)",
        rf_noids,
        X_train_noids,
        X_test_noids,
        y_train,
        y_test,
    )

    # ===============================================================
    # FINAL COMPARISON
    # ===============================================================
    print("\n================ FINAL COMPARISON =================")
    print(f"Macro F1 WITH IDs : {f1_ids:.3f}")
    print(f"Macro F1 NO IDs   : {f1_noids:.3f}")
    print(f"Delta             : {f1_ids - f1_noids:.3f}")


if __name__ == "__main__":
    main()