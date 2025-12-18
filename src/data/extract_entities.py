import pandas as pd

# === 1) Load the final merged dataset
# IMPORTANT: update the file name if yours is different
df = pd.read_csv("data/processed/f1_finalcomplete_global_sc_weather_standings_pitstop.csv")

# === 2) Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()

# === 3) Extract the list of constructors (teams)
teams_table = (
    df[["constructorId", "team_name"]]
    .drop_duplicates()
    .sort_values(["team_name", "constructorId"])
)

print("\n=== LIST OF TEAMS ===")
print(teams_table.to_string(index=False))

# === 4) Extract the list of drivers
drivers_table = (
    df[["driverId", "code", "forename", "surname"]]
    .drop_duplicates()
    .sort_values(["surname", "forename"])
)

print("\n=== LIST OF DRIVERS ===")
print(drivers_table.to_string(index=False))

# === 5) Optional: List of drivers by team
drivers_by_team = (
    df[["team_name", "driverId", "code", "forename", "surname"]]
    .drop_duplicates()
    .sort_values(["team_name", "driverId"])
)

print("\n=== DRIVERS BY TEAM ===")
print(drivers_by_team.to_string(index=False))