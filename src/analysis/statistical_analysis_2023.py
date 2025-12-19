from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

RF_MODEL_FILE = MODELS_DIR / "random_forest_multiclass.joblib"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _clean_rainfall_any(df: pd.DataFrame) -> pd.DataFrame:
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
        )
        df["rainfall_any"] = df["rainfall_any"].astype(float).fillna(0).astype(int)
    return df


def _feature_engineering_prerace(df: pd.DataFrame) -> pd.DataFrame:
    if "grid" in df.columns:
        max_grid = df["grid"].replace(0, pd.NA).max()
        df["grid_norm"] = df["grid"] / max_grid if pd.notna(max_grid) and max_grid > 0 else 0.0

    if {"year", "constructorId", "grid"}.issubset(df.columns):
        df["team_avg_grid"] = df.groupby(["year", "constructorId"])["grid"].transform("mean")

    if {"raceId", "constructorId", "grid"}.issubset(df.columns):

        def teammate_grid(series: pd.Series) -> pd.Series:
            n = len(series)
            if n <= 1:
                return pd.Series([np.nan] * n, index=series.index)
            total = series.sum()
            return (total - series) / (n - 1)

        df["teammate_grid"] = df.groupby(["raceId", "constructorId"])["grid"].transform(teammate_grid)
        df["teammate_grid"] = df["teammate_grid"].fillna(df["grid"])

    return df


def _build_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    categorical_cols = [c for c in ["team_name", "driverId", "constructorId", "raceId"] if c in X.columns]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


def _align_columns(X: pd.DataFrame, reference_cols: list[str]) -> pd.DataFrame:
    for c in reference_cols:
        if c not in X.columns:
            X[c] = 0.0
    return X[reference_cols]


def _split_temporal_2023(df: pd.DataFrame):
    return df[df["year"].isin([2021, 2022])].copy(), df[df["year"] == 2023].copy()


def _safe_abs_delta(df: pd.DataFrame) -> pd.Series:
    if "delta" in df.columns:
        return df["delta"].astype(float).abs()
    if {"finish_position", "grid"}.issubset(df.columns):
        return (df["finish_position"] - df["grid"]).abs()
    return pd.Series([np.nan] * len(df), index=df.index)


def _finish_group(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 5, 10, 15, np.inf]
    labels = ["Top 5", "6-10", "11-15", "16+"]
    return pd.cut(series.astype(float), bins=bins, labels=labels)


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    aux_keep = [
        c
        for c in [
            "delta",
            "finish_position",
            "grid",
            "rainfall_any",
            "team_name",
            "driverId",
            "constructorId",
            "raceId",
            "round",
            "year",
        ]
        if c in df.columns
    ]

    df = _clean_rainfall_any(df)
    df = _feature_engineering_prerace(df)

    if "label" not in df.columns:
        raise ValueError("Missing 'label' in dataset")

    train_df, test_df = _split_temporal_2023(df)

    print(f"Train rows (2021-2022): {len(train_df)}")
    print(f"Test rows  (2023): {len(test_df)}")

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

    y_test = test_df["label"].astype(int)

    print("\nLoading Random Forest model...")
    rf = joblib.load(RF_MODEL_FILE)

    X_train_rf = _build_X(train_df, FEATURE_COLS)
    X_test_rf = _build_X(test_df, FEATURE_COLS)

    if hasattr(rf, "feature_names_in_"):
        X_test_rf = _align_columns(X_test_rf, list(rf.feature_names_in_))
    else:
        X_test_rf = _align_columns(X_test_rf, list(X_train_rf.columns))

    print("Computing permutation importance (macro-F1)...")
    perm = permutation_importance(
        rf,
        X_test_rf,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1_macro",
        n_jobs=-1,
    )

    imp_df = pd.DataFrame(
        {
            "feature": X_test_rf.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    imp_interpretable = imp_df[
        ~imp_df["feature"].str.startswith(("driverId_", "constructorId_", "raceId_"))
    ].copy()

    imp_df.to_csv(RESULTS_DIR / "rf_permutation_importance_2023_all.csv", index=False)
    imp_interpretable.to_csv(
        RESULTS_DIR / "rf_permutation_importance_2023_interpretable.csv", index=False
    )

    y_pred_rf = rf.predict(X_test_rf)
    analysis_df = test_df[aux_keep].copy()
    analysis_df["y_true"] = y_test.values
    analysis_df["y_pred"] = y_pred_rf
    analysis_df["error"] = (y_pred_rf != y_test.values).astype(int)

    analysis_df["abs_delta"] = _safe_abs_delta(test_df)
    if "finish_position" in analysis_df.columns:
        analysis_df["finish_group"] = _finish_group(analysis_df["finish_position"])

    overall_error = analysis_df["error"].mean()

    analysis_df.groupby("y_true")["error"].mean().reset_index().to_csv(
        RESULTS_DIR / "error_rate_by_true_class_2023.csv", index=False
    )
    analysis_df.groupby("rainfall_any")["error"].mean().reset_index().to_csv(
        RESULTS_DIR / "error_rate_by_rain_2023.csv", index=False
    )

    analysis_df["abs_delta_bin"] = pd.cut(
        analysis_df["abs_delta"],
        bins=[-np.inf, 2, 5, 10, 30, np.inf],
        labels=["<=2", "2-5", "5-10", "10-30", "30+"],
    )
    analysis_df.groupby("abs_delta_bin")["error"].mean().reset_index().to_csv(
        RESULTS_DIR / "error_rate_by_abs_delta_bin_2023.csv", index=False
    )

    if "finish_group" in analysis_df.columns:
        analysis_df.groupby("finish_group")["error"].mean().reset_index().to_csv(
            RESULTS_DIR / "error_rate_by_finish_group_2023.csv", index=False
        )

    pd.DataFrame(
        [{"metric": "overall_error_rate", "value": float(overall_error)}]
    ).to_csv(RESULTS_DIR / "error_analysis_summary_2023.csv", index=False)

    analysis_df.to_csv(RESULTS_DIR / "error_rows_2023_rf.csv", index=False)

    print("\n==================== 2023 ERROR ANALYSIS (RF) ====================")
    print("Confusion matrix (2023):")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nMacro F1 (2023):", f"{f1_score(y_test, y_pred_rf, average='macro'):.3f}")
    print("\nOverall error rate:", f"{overall_error:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()