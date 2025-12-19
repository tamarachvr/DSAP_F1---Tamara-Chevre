import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
# This file lives in: DSAP_F1/src/analysis/
# ROOT must point to: DSAP_F1/
ROOT = Path(__file__).resolve().parents[2]  # -> DSAP_F1/

DATA = ROOT / "data"
GLOBAL = DATA / "global"
PROCESSED = DATA / "processed"

FINAL_BASE_FILE = PROCESSED / "f1_final_global_safety_weather_2021_2023.csv"
OUTPUT_FILE = PROCESSED / "f1_finalcomplete_global_sc_weather_standings_pitstop.csv"


def load_core_datasets():
    """Load final merged dataset + global tables needed to compute features."""
    df = pd.read_csv(FINAL_BASE_FILE)

    # Clean column names: remove leading/trailing spaces
    df.columns = [c.strip() for c in df.columns]

    # Drop useless 'round' column coming from weather merge, if present
    # (we keep the main 'round' from the base dataset)
    for col in list(df.columns):
        if col.strip().lower() == "round_weather":
            df = df.drop(columns=[col])
            break

    races = pd.read_csv(GLOBAL / "races.csv")
    driver_standings = pd.read_csv(GLOBAL / "driver_standings.csv")
    constructor_standings = pd.read_csv(GLOBAL / "constructor_standings.csv")
    pit_stops = pd.read_csv(GLOBAL / "pit_stops.csv")

    # Clean columns for all
    races.columns = [c.strip() for c in races.columns]
    driver_standings.columns = [c.strip() for c in driver_standings.columns]
    constructor_standings.columns = [c.strip() for c in constructor_standings.columns]
    pit_stops.columns = [c.strip() for c in pit_stops.columns]

    return df, races, driver_standings, constructor_standings, pit_stops


def build_pre_race_driver_standings(races, driver_standings):
    """Compute driver standings BEFORE each race."""
    races_small = races[["raceId", "year", "round"]].copy()

    ds = driver_standings.merge(
        races_small,
        on="raceId",
        how="left",
    )
    ds = ds[ds["year"].between(2021, 2023)].copy()
    ds = ds.sort_values(["year", "driverId", "round"])

    ds["driver_standings_position_pre_race"] = (
        ds.groupby(["year", "driverId"])["position"].shift(1)
    )
    ds["driver_standings_points_pre_race"] = (
        ds.groupby(["year", "driverId"])["points"].shift(1)
    )

    keep_cols = [
        "year",
        "round",
        "driverId",
        "driver_standings_position_pre_race",
        "driver_standings_points_pre_race",
    ]
    return ds[keep_cols]


def build_pre_race_constructor_standings(races, constructor_standings):
    """Compute constructor standings BEFORE each race."""
    races_small = races[["raceId", "year", "round"]].copy()

    cs = constructor_standings.merge(
        races_small,
        on="raceId",
        how="left",
    )
    cs = cs[cs["year"].between(2021, 2023)].copy()
    cs = cs.sort_values(["year", "constructorId", "round"])

    cs["constructor_standings_position_pre_race"] = (
        cs.groupby(["year", "constructorId"])["position"].shift(1)
    )
    cs["constructor_standings_points_pre_race"] = (
        cs.groupby(["year", "constructorId"])["points"].shift(1)
    )

    keep_cols = [
        "year",
        "round",
        "constructorId",
        "constructor_standings_position_pre_race",
        "constructor_standings_points_pre_race",
    ]
    return cs[keep_cols]


def build_pitstop_features(pit_stops, races):
    """Compute pit stop features."""
    races_small = races[["raceId", "year", "round"]].copy()

    ps = pit_stops.merge(
        races_small,
        on="raceId",
        how="left",
    )
    ps = ps[ps["year"].between(2021, 2023)].copy()

    agg = (
        ps.groupby(["year", "round", "driverId"], as_index=False)
        .agg(
            pitstop_count=("stop", "nunique"),
            pitstop_total_ms=("milliseconds", "sum"),
        )
    )

    return agg


def main():
    print("=== Loading base + global tables ===")
    df, races, driver_standings, constructor_standings, pit_stops = load_core_datasets()
    print("Base final dataset:", df.shape)

    print("\n=== Computing pre-race driver standings ===")
    pre_driver = build_pre_race_driver_standings(races, driver_standings)
    print("Pre-race driver standings:", pre_driver.shape)

    print("\n=== Computing pre-race constructor standings ===")
    pre_constructor = build_pre_race_constructor_standings(races, constructor_standings)
    print("Pre-race constructor standings:", pre_constructor.shape)

    print("\n=== Computing pit stop features ===")
    pit_features = build_pitstop_features(pit_stops, races)
    print("Pit stop features:", pit_features.shape)

    print("\n=== Merging pre-race driver standings into final dataset ===")
    df = df.merge(
        pre_driver,
        on=["year", "round", "driverId"],
        how="left",
    )
    print("After driver standings merge:", df.shape)

    print("\n=== Merging pre-race constructor standings into final dataset ===")
    df = df.merge(
        pre_constructor,
        on=["year", "round", "constructorId"],
        how="left",
    )
    print("After constructor standings merge:", df.shape)

    print("\n=== Merging pit stop features into final dataset ===")
    df = df.merge(
        pit_features,
        on=["year", "round", "driverId"],
        how="left",
    )
    print("After pit stop merge:", df.shape)

    # -----------------------------------------
    # Sort final dataset by year, date, finish position
    # -----------------------------------------
    df.columns = [c.strip() for c in df.columns]  # one last cleanup
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        by=["year", "date", "finish_position"],
        ascending=[True, True, True],
    )

    print("Saving final dataset...")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Final complete dataset saved to:")
    print("  ", OUTPUT_FILE)
    print(df.head())


if __name__ == "__main__":
    main()