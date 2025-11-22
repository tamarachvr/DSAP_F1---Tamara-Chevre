import pandas as pd
from pathlib import Path

ROOT = Path("DSAP_F1")
DATA = ROOT / "data"
PROCESSED = DATA / "processed"


def load_base():
    """Load the main merged F1 dataset (2021–2023)."""
    base_path = PROCESSED / "f1_base_results_2021_2023.csv"
    df = pd.read_csv(base_path)
    print("Base F1 data loaded:", df.shape)
    return df


def load_safetycar():
    """Load the safety car summary per race."""
    sc_path = PROCESSED / "safetycar_2021_2023_by_race.csv"
    df = pd.read_csv(sc_path)
    print("Safety car summary loaded:", df.shape)
    return df


def merge_safetycar(base, safety):
    """
    Merge safety car data with the main F1 base.
    
    Matching keys:
    - year
    - race_name (harmonized)
    """

    # Standardize race names for matching
    base["race_name_clean"] = (
        base["race_name"]
        .str.replace("Grand Prix", "", regex=False)
        .str.strip()
        .str.lower()
    )

    safety["race_name_clean"] = (
        safety["race_name"]
        .str.replace("Grand Prix", "", regex=False)
        .str.strip()
        .str.lower()
    )

    # Merge
    merged = base.merge(
        safety,
        on=["year", "race_name_clean"],
        how="left",
        suffixes=("", "_sc"),
    )

    print("\nAfter merging with safety car:", merged.shape)

    # Save
    out_path = PROCESSED / "f1_with_safetycar_2021_2023.csv"
    merged.to_csv(out_path, index=False)
    print("Saved merged dataset to:", out_path)

    return merged


def main():
    base = load_base()
    safety = load_safetycar()
    merged = merge_safetycar(base, safety)
    print("\nPreview of merged data:")
    print(merged.head(20))


if __name__ == "__main__":
    main()