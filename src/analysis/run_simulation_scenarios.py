from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

N_SIMULATIONS = 10_000
RANDOM_SEED = 42


# ---------------------------------------------------------------------
# Data preparation (same core logic as training, but NO leakage columns)
# ---------------------------------------------------------------------
def load_and_prepare_data():
    """
    Load the dataset and apply the same preprocessing style as training:
      - clean rainfall_any
      - feature engineering (grid_norm, team_avg_grid, teammate_grid)
      - one-hot encoding
    Returns:
        df : original dataframe (readable columns like year, round, driverId, team_name, etc.)
        X  : feature matrix ready for the model
    """
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    # Clean rainfall_any -> 0/1
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

    # Feature engineering (pre-race style)
    # 1) grid_norm
    if "grid" in df.columns:
        max_grid = df["grid"].replace(0, pd.NA).max()
        if pd.notna(max_grid) and max_grid > 0:
            df["grid_norm"] = df["grid"] / max_grid
        else:
            df["grid_norm"] = 0.0

    # 2) team_avg_grid (by year + constructor)
    if {"constructorId", "grid"}.issubset(df.columns):
        group_cols = ["constructorId"]
        if "year" in df.columns:
            group_cols = ["year", "constructorId"]
        df["team_avg_grid"] = df.groupby(group_cols)["grid"].transform("mean")

    # 3) teammate_grid
    if {"raceId", "constructorId", "grid"}.issubset(df.columns):

        def teammate_grid(series):
            n = len(series)
            if n <= 1:
                return pd.Series([np.nan] * n, index=series.index)
            total = series.sum()
            return (total - series) / (n - 1)

        df["teammate_grid"] = (
            df.groupby(["raceId", "constructorId"])["grid"].transform(teammate_grid)
        )
        df["teammate_grid"] = df["teammate_grid"].fillna(df["grid"])

    if "label" not in df.columns:
        raise ValueError("Column 'label' is missing from the dataset.")

    # Build feature matrix X
    # Drop target + any post-race/leakage columns if present
    drop_cols = ["label"]
    if "delta" in df.columns:
        drop_cols.append("delta")
    if "finish_position" in df.columns:
        drop_cols.append("finish_position")
    # Just in case older dataset still has it
    if "is_sim_race" in df.columns:
        drop_cols.append("is_sim_race")

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    categorical_cols = [
        c for c in ["team_name", "driverId", "constructorId", "raceId"] if c in X.columns
    ]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df, X


def load_random_forest():
    model_path = MODELS_DIR / "random_forest_multiclass.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"RandomForest model not found at: {model_path}. "
            "Please run train_over_under_model.py first."
        )
    return joblib.load(model_path)


def align_features_with_model(X: pd.DataFrame, model):
    """
    Ensure X columns match exactly what the model expects (same columns, same order).
    """
    if not hasattr(model, "feature_names_in_"):
        print("Warning: model has no 'feature_names_in_' -> cannot perfectly align.")
        return X

    model_cols = list(model.feature_names_in_)

    # Add missing columns as zeros
    missing = [c for c in model_cols if c not in X.columns]
    for c in missing:
        X[c] = 0.0

    # Drop extra columns
    extra = [c for c in X.columns if c not in model_cols]
    if extra:
        X = X.drop(columns=extra)

    # Reorder
    X = X[model_cols]
    return X


def get_race_subset(df: pd.DataFrame, X: pd.DataFrame, year: int, round_: int):
    mask = (df["year"] == year) & (df["round"] == round_)
    race_df = df.loc[mask].copy()
    if race_df.empty:
        raise ValueError(f"No race found for year={year}, round={round_}")
    race_X = X.loc[race_df.index]
    return race_df, race_X


# ---------------------------------------------------------------------
# Pretty printing for race-level summary (key: value)
# ---------------------------------------------------------------------
def print_race_summary_kv(race_summary: pd.DataFrame):
    """
    Print the single-row race_summary as key: value (no alignment issues).
    """
    if race_summary.empty:
        print("Race-level summary: <empty>")
        return

    row = race_summary.iloc[0].to_dict()

    order = [
        "scenario",
        "year",
        "round",
        "rainfall_any",
        "n_drivers",
        "n_simulations",
        "expected_under",
        "expected_neutral",
        "expected_over",
        "mc_mean_under",
        "mc_mean_neutral",
        "mc_mean_over",
    ]

    print("\nRace-level summary (expected from model vs Monte Carlo mean counts):")
    for k in order:
        v = row.get(k, None)
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


# ---------------------------------------------------------------------
# Monte Carlo: sample labels using model probabilities
# ---------------------------------------------------------------------
def monte_carlo_label_sampling(probs: np.ndarray, n_simulations: int, random_seed: int):
    """
    probs: shape (n_drivers, 3) for classes [0,1,2]
    Returns:
        mc_prob_under/neut/over per driver (means over simulations)
    """
    rng = np.random.default_rng(random_seed)
    n_drivers = probs.shape[0]
    classes = np.array([0, 1, 2])

    # all_labels: (n_drivers, n_simulations)
    all_labels = np.zeros((n_drivers, n_simulations), dtype=np.int8)

    for i in range(n_drivers):
        all_labels[i, :] = rng.choice(classes, size=n_simulations, p=probs[i])

    mc_prob_under = (all_labels == 0).mean(axis=1)
    mc_prob_neutral = (all_labels == 1).mean(axis=1)
    mc_prob_over = (all_labels == 2).mean(axis=1)

    return mc_prob_under, mc_prob_neutral, mc_prob_over


def run_distribution_simulation_for_race(
    model,
    df: pd.DataFrame,
    X: pd.DataFrame,
    year: int,
    round_: int,
    scenario_name: str,
    n_simulations: int = 10_000,
    random_seed: int = 42,
):
    """
    For a given race:
      - compute model probabilities p_under/p_neutral/p_over per driver
      - Monte Carlo label sampling to estimate mc_prob_* per driver
      - print ALL drivers in terminal (prof-friendly)
      - save drivers table + race summary to CSV
    """
    print("\n" + "=" * 90)
    print(f"=== DISTRIBUTION simulation: {scenario_name} (year={year}, round={round_}) ===")

    race_df, race_X = get_race_subset(df, X, year, round_)

    # Sanity: features must match
    if hasattr(model, "n_features_in_") and race_X.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: model expects {model.n_features_in_}, "
            f"but race_X has {race_X.shape[1]}."
        )

    probs = model.predict_proba(race_X)  # (n_drivers, 3)
    p_under = probs[:, 0]
    p_neutral = probs[:, 1]
    p_over = probs[:, 2]

    mc_prob_under, mc_prob_neutral, mc_prob_over = monte_carlo_label_sampling(
        probs=probs,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )

    # Driver-level output (NO rainfall_any; it stays in race summary only)
    out_drivers = pd.DataFrame(
        {
            "driverId": race_df["driverId"].values,
            "team_name": race_df["team_name"].values if "team_name" in race_df.columns else "NA",
            "p_under": p_under,
            "p_neutral": p_neutral,
            "p_over": p_over,
            "mc_prob_under": mc_prob_under,
            "mc_prob_neutral": mc_prob_neutral,
            "mc_prob_over": mc_prob_over,
        }
    ).reset_index(drop=True)

    # Race-level summary: expected counts (model) vs MC mean counts
    expected_under = float(p_under.sum())
    expected_neutral = float(p_neutral.sum())
    expected_over = float(p_over.sum())

    mc_mean_under = float(mc_prob_under.sum())
    mc_mean_neutral = float(mc_prob_neutral.sum())
    mc_mean_over = float(mc_prob_over.sum())

    rainfall_any_val = (
        int(race_df["rainfall_any"].iloc[0]) if "rainfall_any" in race_df.columns else np.nan
    )

    race_summary = pd.DataFrame(
        [
            {
                "scenario": scenario_name,
                "year": year,
                "round": round_,
                "rainfall_any": rainfall_any_val,
                "n_drivers": int(len(out_drivers)),
                "n_simulations": int(n_simulations),
                "expected_under": expected_under,
                "expected_neutral": expected_neutral,
                "expected_over": expected_over,
                "mc_mean_under": mc_mean_under,
                "mc_mean_neutral": mc_mean_neutral,
                "mc_mean_over": mc_mean_over,
            }
        ]
    )

    # -------------------- TERMINAL OUTPUT (prof-friendly) --------------------
    # Print race summary as key:value
    print_race_summary_kv(race_summary)

    # Print all drivers sorted by p_over (model) - keep your table style
    print("\nAll drivers sorted by p_over (model):")
    print(
        out_drivers.sort_values("p_over", ascending=False)[
            [
                "driverId",
                "team_name",
                "p_under",
                "p_neutral",
                "p_over",
                "mc_prob_under",
                "mc_prob_neutral",
                "mc_prob_over",
            ]
        ].to_string(index=False)
    )

    # -------------------- SAVE CSVs --------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    out_drivers_path = RESULTS_DIR / f"simulation_distribution_drivers_{scenario_name}.csv"
    out_race_path = RESULTS_DIR / f"simulation_distribution_race_{scenario_name}.csv"

    out_drivers.to_csv(out_drivers_path, index=False)
    race_summary.to_csv(out_race_path, index=False)

    print(f"\nSaved drivers table: {out_drivers_path}")
    print(f"Saved race summary : {out_race_path}")


def main():
    print("Loading data / features...")
    df, X = load_and_prepare_data()
    print(f"X shape before alignment: {X.shape}")

    print("Loading Random Forest model...")
    model = load_random_forest()

    print("Aligning X columns to model...")
    X = align_features_with_model(X, model)
    print(f"X shape after alignment : {X.shape}")

    # Your 4 scenarios (2023 only)
    scenarios = [
        ("azerbaijan_2023_street_dry", 2023, 4),
        ("monaco_2023_street_wet", 2023, 6),
        ("austrian_2023_perm_wet", 2023, 9),
        ("british_2023_perm_dry", 2023, 10),
    ]

    for scenario_name, year, rnd in scenarios:
        run_distribution_simulation_for_race(
            model=model,
            df=df,
            X=X,
            year=year,
            round_=rnd,
            scenario_name=scenario_name,
            n_simulations=N_SIMULATIONS,
            random_seed=RANDOM_SEED,
        )

    print("\nSimulations done. Check: results/")


if __name__ == "__main__":
    main()