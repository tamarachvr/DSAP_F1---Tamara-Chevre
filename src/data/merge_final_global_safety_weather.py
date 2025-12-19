import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # DSAP_F1/
DATA = ROOT / "data"
GLOBAL = DATA / "global"
PROCESSED = DATA / "processed"

BASE_FILE = PROCESSED / "f1_with_safetycar_2021_2023.csv"
WEATHER_FILE = PROCESSED / "weather_2021_2023_by_round.csv"
RACES_FILE = GLOBAL / "races.csv"
OUTPUT_FILE = PROCESSED / "f1_final_global_safety_weather_2021_2023.csv"


def main():
    print("=== Loading datasets ===")
    base = pd.read_csv(BASE_FILE)
    weather = pd.read_csv(WEATHER_FILE)
    races = pd.read_csv(RACES_FILE)

    # Clean column names
    base.columns = [c.strip() for c in base.columns]
    weather.columns = [c.strip() for c in weather.columns]
    races.columns = [c.strip() for c in races.columns]

    # Keep only needed race columns
    races_small = races[["raceId", "year", "round"]]

    print("\n=== Adding 'round' to base ===")
    base_with_round = base.merge(
        races_small,
        on=["raceId", "year"],
        how="left",
        validate="m:m",
    )

    # Validate round merge
    if base_with_round["round"].isna().any():
        missing = base_with_round[base_with_round["round"].isna()][["year", "raceId"]].drop_duplicates()
        raise ValueError(f"Missing rounds:\n{missing}")

    # Ensure integer types
    base_with_round["year"] = base_with_round["year"].astype("int64")
    base_with_round["round"] = base_with_round["round"].astype("int64")
    weather["year"] = weather["year"].astype("int64")
    weather["round"] = weather["round"].astype("int64")

    print("\n=== Merging with weather ===")
    merged = base_with_round.merge(
        weather,
        on=["year", "round"],
        how="left",
        suffixes=("", "_weather"),
    )

    print("Merged dataset shape:", merged.shape)

    # -----------------------------------------
    # Remove unwanted columns
    # -----------------------------------------
    cols_to_drop = [
        "race_name_clean",
        "race_name_sc",
        "first_deploy_lap",
        "last_retreated_lap",
        "airtemp_min",
        "airtemp_max",
    ]
    merged = merged.drop(columns=cols_to_drop, errors="ignore")

    # Remove useless duplicate round column from weather
    if "round_weather" in merged.columns:
        merged = merged.drop(columns=["round_weather"])

    # -----------------------------------------
    # 📌 Sort dataset as before
    # -----------------------------------------
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values(
        by=["year", "date", "finish_position"],
        ascending=[True, True, True],
    )

    # -----------------------------------------
    # Reorder columns: year → date → round → rest
    # -----------------------------------------
    priority = ["year", "date", "round"]
    rest = [c for c in merged.columns if c not in priority]
    merged = merged[priority + rest]

    print("Columns after reordering:", merged.columns.tolist())

    # Save file
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Final merged & cleaned dataset saved to:\n{OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()