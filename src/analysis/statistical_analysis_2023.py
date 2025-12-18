from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

LR_MODEL_FILE = MODELS_DIR / "logistic_regression_multiclass.joblib"
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
    # grid_norm
    if "grid" in df.columns:
        max_grid = df["grid"].replace(0, pd.NA).max()
        if pd.notna(max_grid) and max_grid > 0:
            df["grid_norm"] = df["grid"] / max_grid
        else:
            df["grid_norm"] = 0.0

    # team_avg_grid (year + constructor)
    if {"year", "constructorId", "grid"}.issubset(df.columns):
        df["team_avg_grid"] = df.groupby(["year", "constructorId"])["grid"].transform("mean")

    # teammate_grid (same race + constructor)
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


def _build_X(df: pd.DataFrame, feature_cols: list[str], with_ids: bool) -> pd.DataFrame:
    X = df[feature_cols].copy()

    if with_ids:
        categorical_cols = [c for c in ["team_name", "driverId", "constructorId", "raceId"] if c in X.columns]
    else:
        # no IDs: keep team_name if you consider it an "explanatory" variable;
        # if you want a STRICT no-identity model, you can remove team_name too.
        categorical_cols = [c for c in ["team_name"] if c in X.columns]

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X


def _align_columns(X: pd.DataFrame, reference_cols: list[str]) -> pd.DataFrame:
    # Add missing
    for c in reference_cols:
        if c not in X.columns:
            X[c] = 0.0
    # Drop extra
    X = X[reference_cols]
    return X


def _split_temporal_2023(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["year"].isin([2021, 2022])].copy()
    test_df = df[df["year"] == 2023].copy()
    return train_df, test_df


def _safe_abs_delta(df: pd.DataFrame) -> pd.Series:
    # delta may not exist if dataset changed; compute if possible
    if "delta" in df.columns:
        return df["delta"].astype(float).abs()
    if {"finish_position", "grid"}.issubset(df.columns):
        return (df["finish_position"].astype(float) - df["grid"].astype(float)).abs()
    return pd.Series([np.nan] * len(df), index=df.index)


def _finish_group(series: pd.Series) -> pd.Series:
    # groups: Top5, 6-10, 11-15, 16+
    s = series.astype(float)
    bins = [-np.inf, 5, 10, 15, np.inf]
    labels = ["Top 5", "6-10", "11-15", "16+"]
    return pd.cut(s, bins=bins, labels=labels)


# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    # keep aux columns if present
    aux_keep = [c for c in ["delta", "finish_position", "grid", "rainfall_any", "team_name", "driverId", "constructorId", "raceId", "round", "year"] if c in df.columns]

    df = _clean_rainfall_any(df)
    df = _feature_engineering_prerace(df)

    if "label" not in df.columns:
        raise ValueError("Missing 'label' in dataset")

    # Temporal split
    train_df, test_df = _split_temporal_2023(df)

    print(f"Train rows (2021-2022): {len(train_df)}")
    print(f"Test rows  (2023): {len(test_df)}")

    # Whitelist features (pre-race)
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

    # Targets
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)

    # ------------------------------------------------------------
    # 1) LR coefficients table (directional interpretation)
    #    Use WITH IDs design matrix because your trained LR likely used that.
    # ------------------------------------------------------------
    print("\nLoading Logistic Regression model...")
    lr = joblib.load(LR_MODEL_FILE)

    # Build X with IDs then align to model input columns
    X_train_lr = _build_X(train_df, FEATURE_COLS, with_ids=True)
    X_test_lr = _build_X(test_df, FEATURE_COLS, with_ids=True)

    # Align to model training columns (best effort)
    if hasattr(lr, "feature_names_in_"):
        ref_cols = list(lr.feature_names_in_)
        X_test_lr = _align_columns(X_test_lr, ref_cols)
    else:
        # fallback: align to train columns we computed
        ref_cols = list(X_train_lr.columns)
        X_test_lr = _align_columns(X_test_lr, ref_cols)

    # Predict & sanity
    y_pred_lr = lr.predict(X_test_lr)
    macro_f1_lr = f1_score(y_test, y_pred_lr, average="macro")
    print(f"LR Macro F1 on 2023 (sanity): {macro_f1_lr:.3f}")

    # Extract coefficients
    coef = lr.coef_  # shape: (n_classes, n_features)
    classes = list(getattr(lr, "classes_", [0, 1, 2]))
    feat_names = ref_cols

    coef_df = pd.DataFrame(coef, columns=feat_names)
    coef_df.insert(0, "class", classes)

    # Create a “long” table for readability: top +/- per class
    long_rows = []
    for i, cls in enumerate(classes):
        row = coef_df[coef_df["class"] == cls].drop(columns=["class"]).iloc[0]
        # Keep only non-ID-ish features for interpretability:
        # (We drop one-hot columns that look like driverId_, constructorId_, raceId_)
        keep = row.index[
            ~row.index.str.startswith("driverId_")
            & ~row.index.str.startswith("constructorId_")
            & ~row.index.str.startswith("raceId_")
        ]
        row2 = row[keep].sort_values()
        # take extremes
        neg = row2.head(15)
        pos = row2.tail(15)

        for var, val in neg.items():
            long_rows.append({"class": cls, "direction": "neg", "feature": var, "coef": float(val)})
        for var, val in pos.items():
            long_rows.append({"class": cls, "direction": "pos", "feature": var, "coef": float(val)})

    coef_long = pd.DataFrame(long_rows).sort_values(["class", "direction", "coef"])
    coef_path = RESULTS_DIR / "lr_coefficients_top_2023.csv"
    coef_long.to_csv(coef_path, index=False)
    print(f"Saved LR coefficients table to: {coef_path}")

    # ------------------------------------------------------------
    # 2) RF permutation importance on 2023 test (macro-F1)
    # ------------------------------------------------------------
    print("\nLoading Random Forest model...")
    rf = joblib.load(RF_MODEL_FILE)

    # Build X with IDs because your RF was trained with IDs in your pipeline
    X_train_rf = _build_X(train_df, FEATURE_COLS, with_ids=True)
    X_test_rf = _build_X(test_df, FEATURE_COLS, with_ids=True)

    # Align columns to model expected
    if hasattr(rf, "feature_names_in_"):
        ref_cols_rf = list(rf.feature_names_in_)
        X_test_rf = _align_columns(X_test_rf, ref_cols_rf)
    else:
        ref_cols_rf = list(X_train_rf.columns)
        X_test_rf = _align_columns(X_test_rf, ref_cols_rf)

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
            "feature": ref_cols_rf,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # For readability, create an "interpretable" version without IDs
    imp_interpretable = imp_df[
        ~imp_df["feature"].str.startswith("driverId_")
        & ~imp_df["feature"].str.startswith("constructorId_")
        & ~imp_df["feature"].str.startswith("raceId_")
    ].copy()

    imp_path_all = RESULTS_DIR / "rf_permutation_importance_2023_all.csv"
    imp_path_int = RESULTS_DIR / "rf_permutation_importance_2023_interpretable.csv"
    imp_df.to_csv(imp_path_all, index=False)
    imp_interpretable.to_csv(imp_path_int, index=False)
    print(f"Saved RF permutation importance to: {imp_path_all}")
    print(f"Saved RF interpretable importance to: {imp_path_int}")

    # ------------------------------------------------------------
    # 3) Descriptive conditional error tables on 2023 (using RF preds)
    # ------------------------------------------------------------
    y_pred_rf = rf.predict(X_test_rf)
    err = (y_pred_rf != y_test.values).astype(int)

    # Make an analysis dataframe with aux columns
    analysis_df = test_df[aux_keep].copy()
    analysis_df["y_true"] = y_test.values
    analysis_df["y_pred"] = y_pred_rf
    analysis_df["error"] = err

    # abs_delta + groups
    analysis_df["abs_delta"] = _safe_abs_delta(test_df)
    if "finish_position" in analysis_df.columns:
        analysis_df["finish_group"] = _finish_group(analysis_df["finish_position"])
    else:
        analysis_df["finish_group"] = np.nan

    # ---- Tables ----
    overall_error = analysis_df["error"].mean()

    by_class = analysis_df.groupby("y_true")["error"].mean()

    by_rain = None
    if "rainfall_any" in analysis_df.columns:
        by_rain = analysis_df.groupby("rainfall_any")["error"].mean()

    # abs_delta bins
    # bins chosen to match what you started doing earlier
    abs_delta_bins = pd.cut(
        analysis_df["abs_delta"],
        bins=[-np.inf, 2, 5, 10, 30, np.inf],
        labels=["<=2", "2-5", "5-10", "10-30", "30+"],
    )
    analysis_df["abs_delta_bin"] = abs_delta_bins
    by_abs_delta = analysis_df.groupby("abs_delta_bin")["error"].mean()

    by_finish = None
    if "finish_group" in analysis_df.columns:
        by_finish = analysis_df.groupby("finish_group")["error"].mean()

    # Save summary tables
    summary_rows = [{"metric": "overall_error_rate", "value": float(overall_error)}]
    summary_path = RESULTS_DIR / "error_analysis_summary_2023.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    by_class_path = RESULTS_DIR / "error_rate_by_true_class_2023.csv"
    by_class.reset_index().rename(columns={"y_true": "true_class", "error": "error_rate"}).to_csv(by_class_path, index=False)

    if by_rain is not None:
        by_rain_path = RESULTS_DIR / "error_rate_by_rain_2023.csv"
        by_rain.reset_index().rename(columns={"rainfall_any": "rainfall_any", "error": "error_rate"}).to_csv(by_rain_path, index=False)
    else:
        by_rain_path = None

    by_abs_delta_path = RESULTS_DIR / "error_rate_by_abs_delta_bin_2023.csv"
    by_abs_delta.reset_index().rename(columns={"abs_delta_bin": "abs_delta_bin", "error": "error_rate"}).to_csv(by_abs_delta_path, index=False)

    if by_finish is not None:
        by_finish_path = RESULTS_DIR / "error_rate_by_finish_group_2023.csv"
        by_finish.reset_index().rename(columns={"finish_group": "finish_group", "error": "error_rate"}).to_csv(by_finish_path, index=False)
    else:
        by_finish_path = None

    # Save row-level analysis for deeper digging
    rows_path = RESULTS_DIR / "error_rows_2023_rf.csv"
    analysis_df.to_csv(rows_path, index=False)

    # Print key results nicely
    print("\n==================== 2023 ERROR ANALYSIS (RF) ====================")
    print("Confusion matrix (2023):")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nMacro F1 (2023):", f"{f1_score(y_test, y_pred_rf, average='macro'):.3f}")
    print("\nOverall error rate:", f"{overall_error:.3f}")
    print("\nError rate by true class:")
    print(by_class)

    if by_rain is not None:
        print("\nError rate by rain (0=dry,1=rain):")
        print(by_rain)

    print("\nError rate by abs(delta) bin:")
    print(by_abs_delta)

    if by_finish is not None:
        print("\nError rate by finish group:")
        print(by_finish)

    print("\nSaved error analysis tables:")
    print(" -", summary_path)
    print(" -", by_class_path)
    if by_rain_path:
        print(" -", by_rain_path)
    print(" -", by_abs_delta_path)
    if by_finish_path:
        print(" -", by_finish_path)
    print(" -", rows_path)

    print("\nDone.")


if __name__ == "__main__":
    main()