import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
# This file lives in: DSAP_F1/src/data/
# ROOT must point to: DSAP_F1/
ROOT = Path(__file__).resolve().parent.parent.parent  # -> DSAP_F1/
DATA = ROOT / "data"
WEATHER = DATA / "weather"
PROCESSED = DATA / "processed"

# Make sure processed/ exists
PROCESSED.mkdir(parents=True, exist_ok=True)


def load_raw_weather() -> pd.DataFrame:
    """
    Load the raw weather CSV from the Kaggle dataset.

    Expected columns (example):
    Time, AirTemp, Humidity, Pressure, Rainfall,
    TrackTemp, WindDirection, WindSpeed, Round Number, Year
    """
    df = pd.read_csv(WEATHER / "weather.csv")
    print("\n✅ Raw weather data loaded:", df.shape)
    return df


def prepare_weather_for_merge() -> pd.DataFrame:
    """
    Aggregate weather data at race level so that it can be merged
    with the main F1 dataset.

    The raw file has one row per timestamp.
    We aggregate by (Year, Round) and compute summary statistics
    for each Grand Prix.
    """
    df = load_raw_weather()

    # ---- Ensure Year and Round are numeric ----
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Round"] = pd.to_numeric(df["Round Number"], errors="coerce").astype("Int64")

    # Drop the original "Round Number" column with the space in its name
    df = df.drop(columns=["Round Number"])

    # Keep only rows with valid Year and Round
    df = df[df["Year"].notna() & df["Round"].notna()].copy()

    # ---- Convert numeric weather columns safely ----
    for col in ["AirTemp", "Humidity", "Pressure", "TrackTemp", "WindSpeed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Rainfall: build a boolean flag "did it rain at any time?" ----
    df["Rain_flag"] = (
        df["Rainfall"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )

    # ---- Aggregate by Year + Round (one row per race) ----
    weather_by_race = (
        df.groupby(["Year", "Round"], as_index=False)
        .agg(
            airtemp_mean=("AirTemp", "mean"),
            airtemp_min=("AirTemp", "min"),
            airtemp_max=("AirTemp", "max"),
            tracktemp_mean=("TrackTemp", "mean"),
            humidity_mean=("Humidity", "mean"),
            pressure_mean=("Pressure", "mean"),
            windspeed_mean=("WindSpeed", "mean"),
            rainfall_any=("Rain_flag", "max"),  # True if it rained at any point
        )
    )

    print("Weather summary per race (all years):", weather_by_race.shape)

    # ---- Restrict to 2021–2023 only ----
    weather_2123 = weather_by_race[
        weather_by_race["Year"].between(2021, 2023)
    ].copy()

    # Rename keys to match the main dataset (year, round)
    weather_2123 = weather_2123.rename(
        columns={
            "Year": "year",
            "Round": "round",
        }
    )

    print("Weather summary per race (2021–2023):", weather_2123.shape)
    return weather_2123


def main():
    weather_for_merge = prepare_weather_for_merge()

    out_path = PROCESSED / "weather_2021_2023_by_round.csv"
    weather_for_merge.to_csv(out_path, index=False)
    print(f"\n💾 Saved weather summary to: {out_path}")


if __name__ == "__main__":
    main()