import pandas as pd
from pathlib import Path

# -----------------------------
# Paths configuration
# -----------------------------

# ROOT = folder where this script lives (DSAP_F1/)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

GLOBAL = DATA / "global"
WEATHER = DATA / "weather"
SAFETY = DATA / "safetycar"

print("Root:", ROOT)
print("Data folder:", DATA)


# -----------------------------
# 1) Load core F1 data
# -----------------------------

races = pd.read_csv(GLOBAL / "races.csv")
results = pd.read_csv(GLOBAL / "results.csv")
pit_stops = pd.read_csv(GLOBAL / "pit_stops.csv")

print("\n✅ Global data loaded:")
print("  races     :", races.shape)
print("  results   :", results.shape)
print("  pit_stops :", pit_stops.shape)


# -----------------------------
# 2) Load safety car data
# -----------------------------

safety = pd.read_csv(SAFETY / "safety_cars.csv")
print("\n✅ Safety car data loaded:")
print("  safety_cars:", safety.shape)


# -----------------------------
# 3) Load weather data
# -----------------------------

weather = pd.read_csv(WEATHER / "weather.csv")
print("\n✅ Weather data loaded:")
print("  weather:", weather.shape)
