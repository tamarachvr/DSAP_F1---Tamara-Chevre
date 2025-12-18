import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
# This file lives in: DSAP_F1/src/data/
# ROOT must point to: DSAP_F1/
ROOT = Path(__file__).resolve().parent.parent.parent  # -> DSAP_F1/
DATA = ROOT / "data"
SAFETYCAR = DATA / "safetycar"
PROCESSED = DATA / "processed"

# Make sure processed/ exists
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_raw_safetycar() -> pd.DataFrame:
    """
    Load the raw safety car CSV from the Kaggle dataset.
    """
    df = pd.read_csv(SAFETYCAR / "safety_cars.csv")
    print("\n✅ Raw safety car data loaded:", df.shape)
    return df


def prepare_safetycar_for_merge() -> pd.DataFrame:
    """
    Prepare the safety car dataset so that it can be merged
    with the main F1 results dataset (Ergast).

    Steps:
    - Extract the year and the race name from the 'Race' column
      (e.g. '1998 Canadian Grand Prix' -> year=1998, race_name='Canadian Grand Prix')
    - Keep only seasons 2021–2023
    - Aggregate one row per (year, race_name)
    """
    df = load_raw_safetycar()

    # Split 'Race' into year and race_name
    split = df["Race"].str.split(" ", n=1, expand=True)
    df["year"] = split[0].astype(int)
    df["race_name"] = split[1].str.strip()

    print("After extracting year and race_name:", df.shape)

    # Keep only 2021–2023
    df_2123 = df[df["year"].between(2021, 2023)].copy()
    print("Safety car events 2021–2023 (raw events):", df_2123.shape)

    if df_2123.empty:
        print("\n⚠️ No safety car events found for 2021–2023 in this dataset.")
        print("   Returning empty frame with correct structure.")
        return df_2123[["year", "race_name"]].drop_duplicates()

    # Aggregate to one row per race
    safety_by_race = (
        df_2123
        .groupby(["year", "race_name"], as_index=False)
        .agg(
            safetycar_events=("Race", "size"),
            first_deploy_lap=("Deployed", "min"),
            last_retreated_lap=("Retreated", "max"),
            total_safetycar_laps=("FullLaps", "sum"),
        )
    )

    print("Safety car summary per race (2021–2023):", safety_by_race.shape)
    return safety_by_race


def main():
    safety_for_merge = prepare_safetycar_for_merge()

    out_path = PROCESSED / "safetycar_2021_2023_by_race.csv"
    safety_for_merge.to_csv(out_path, index=False)
    print(f"\n💾 Saved safety car summary to: {out_path}")


if __name__ == "__main__":
    main()