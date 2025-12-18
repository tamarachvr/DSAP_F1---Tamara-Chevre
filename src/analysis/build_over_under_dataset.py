import pandas as pd
from pathlib import Path

# Paths (adapt if your filenames are different)
INPUT_FILE = Path("data/processed/f1_finalcomplete_global_sc_weather_standings_pitstop.csv")
OUTPUT_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")


def create_over_under_dataset():
    # Load the merged race-level dataset
    df = pd.read_csv(INPUT_FILE)

    # Clean column names: remove leading / trailing spaces
    df.columns = df.columns.str.strip()

    print(f"Total rows in base dataset: {len(df)}")

    # ------------------------------------
    # Create delta and multiclass label
    # ------------------------------------
    # Delta: finish - grid (post-race, do NOT use as feature)
    df["delta"] = df["finish_position"] - df["grid"]

    # Multiclass label:
    # 0 = underperformance (loses at least 3 positions)
    # 1 = neutral (-2, -1, 0, +1, +2)
    # 2 = overperformance (gains at least 3 positions)
    def label_from_delta(x: int | float) -> int:
        if x <= -3:
            return 0
        elif x >= 3:
            return 2
        return 1

    df["label"] = df["delta"].apply(label_from_delta).astype(int)

    # ------------------------------------
    # Select feature columns allowed for prediction (pre-race only)
    # ------------------------------------
    feature_cols = [
        "year",
        "round",
        "raceId",
        "driverId",
        "constructorId",
        "team_name",
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

    # ------------------------------------
    # Auxiliary columns kept for analysis/debug (NOT features)
    # ------------------------------------
    aux_cols = [
        "finish_position",  # useful for interpretation / sanity checks
        "delta",            # useful for analysis of errors; never use as feature
    ]

    # Safety check: make sure all required columns exist
    required_cols = feature_cols + aux_cols + ["label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")

    # Keep only the features + aux columns + label
    model_df = df[required_cols].copy()

    # Save to a new CSV that will be used for modelling
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved model dataset to: {OUTPUT_FILE}")
    print(f"Shape: {model_df.shape}")
    print("Label distribution:")
    print(model_df["label"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    create_over_under_dataset()