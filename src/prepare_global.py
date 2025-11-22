import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parent.parent  # -> DSAP_F1/
DATA = ROOT / "data"
GLOBAL = DATA / "global"
PROCESSED = DATA / "processed"

# Make sure processed/ exists
PROCESSED.mkdir(parents=True, exist_ok=True)

print("Root:", ROOT)
print("Data:", DATA)
print("Global data folder:", GLOBAL)
print("Processed folder:", PROCESSED)


def load_global_data():
    """
    Load core F1 tables from the global dataset.
    """
    races = pd.read_csv(GLOBAL / "races.csv")
    results = pd.read_csv(GLOBAL / "results.csv")
    drivers = pd.read_csv(GLOBAL / "drivers.csv")
    constructors = pd.read_csv(GLOBAL / "constructors.csv")

    print("\n✅ Global data loaded:")
    print("  races       :", races.shape)
    print("  results     :", results.shape)
    print("  drivers     :", drivers.shape)
    print("  constructors:", constructors.shape)

    return races, results, drivers, constructors


def build_base_results_2021_2023():
    """
    Build a clean base dataset for seasons 2021–2023.

    One row = one driver in one race, with:
    - grid position
    - finish position
    - driver + team information
    - basic race info (year, date, race name)
    """
    races, results, drivers, constructors = load_global_data()

    # 1) Filter races for 2021–2023
    races_2123 = races[(races["year"] >= 2021) & (races["year"] <= 2023)].copy()
    print("\nRaces 2021–2023:", races_2123.shape)

    # 2) Keep only results for those races
    results_2123 = results.merge(
        races_2123[["raceId", "year", "name", "circuitId", "date"]],
        on="raceId",
        how="inner",
    )
    print("Results 2021–2023 after merge with races:", results_2123.shape)

    # 3) Add driver information
    drivers_small = drivers[["driverId", "forename", "surname", "code", "nationality"]]
    merged = results_2123.merge(drivers_small, on="driverId", how="left")

    # 4) Add constructor (team) information
    constructors_small = constructors[["constructorId", "name", "nationality"]]
    merged = merged.merge(
        constructors_small,
        on="constructorId",
        how="left",
        suffixes=("", "_team"),
    )

    print("After adding drivers + constructors:", merged.shape)

    # 5) Keep only relevant columns
    cols = [
        "year",
        "date",
        "raceId",
        "name",            # race name
        "driverId",
        "forename",
        "surname",
        "code",
        "constructorId",
        "name_team",       # team name
        "grid",            # starting position
        "positionOrder",   # numeric finish position
        "positionText",    # raw finish text (e.g. DNF)
        "statusId",
        "points",
    ]

    base = merged[cols].copy()

    # 6) Rename some columns to clearer names
    base = base.rename(
        columns={
            "name": "race_name",
            "name_team": "team_name",
            "positionOrder": "finish_position",
            "positionText": "finish_text",
        }
    )

    # 7) Sort for readability
    base = base.sort_values(["year", "raceId", "grid"]).reset_index(drop=True)

    print("\n✅ Base dataset 2021–2023 ready:", base.shape)
    return base


def main():
    base = build_base_results_2021_2023()

    # Save to data/processed
    out_path = PROCESSED / "f1_base_results_2021_2023.csv"
    base.to_csv(out_path, index=False)
    print(f"\n💾 Saved base dataset to: {out_path}")


if __name__ == "__main__":
    main()