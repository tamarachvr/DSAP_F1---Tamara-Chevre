import pandas as pd
from pathlib import Path

# === Paths ===
BASE_FILE = Path("DSAP_F1/data/processed/f1_with_safetycar_2021_2023.csv")
WEATHER_FILE = Path("DSAP_F1/data/processed/weather_2021_2023_by_round.csv")
RACES_FILE = Path("DSAP_F1/data/global/races.csv")  # change if your races file has a different name
OUTPUT_FILE = Path("DSAP_F1/data/processed/f1_final_global_safety_weather_2021_2023.csv")


def main():
    print("=== Loading datasets ===")
    base = pd.read_csv(BASE_FILE)
    weather = pd.read_csv(WEATHER_FILE)
    races = pd.read_csv(RACES_FILE)

    print(f"Base dataset shape: {base.shape}")
    print(f"Weather dataset shape: {weather.shape}")
    print(f"Races dataset shape: {races.shape}")

    # Clean column names (strip accidental spaces)
    base.columns = [c.strip() for c in base.columns]
    weather.columns = [c.strip() for c in weather.columns]
    races.columns = [c.strip() for c in races.columns]

    print("\nBase columns:", list(base.columns))
    print("Weather columns:", list(weather.columns))
    print("Races columns:", list(races.columns))

    # Keep only what we need from races: link raceId + year → round
    races_small = races[["raceId", "year", "round"]]

    # Add "round" to the base dataset using raceId + year
    print("\n=== Adding 'round' to base using races table ===")
    base_with_round = base.merge(
        races_small,
        how="left",
        on=["raceId", "year"],
        validate="m:m",  # many drivers per race
    )

    # Safety check: all rows should have a round after the merge
    if base_with_round["round"].isna().any():
        missing = (
            base_with_round[base_with_round["round"].isna()][["year", "raceId"]]
            .drop_duplicates()
        )
        raise ValueError(
            "Some rows in base could not be matched to a round via races.csv:\n"
            f"{missing}"
        )

    # Ensure numeric dtypes for merge
    base_with_round["year"] = base_with_round["year"].astype("int64")
    base_with_round["round"] = base_with_round["round"].astype("int64")
    weather["year"] = weather["year"].astype("int64")
    weather["round"] = weather["round"].astype("int64")

    print("\n=== Merging base+safetycar with weather on (year, round) ===")
    merged = base_with_round.merge(
        weather,
        how="left",  # keep all rows from base
        on=["year", "round"],
        suffixes=("", "_weather"),
    )

    print("Merged dataset shape:", merged.shape)

    # Save final dataset
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Final merged dataset saved to:\n  {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()