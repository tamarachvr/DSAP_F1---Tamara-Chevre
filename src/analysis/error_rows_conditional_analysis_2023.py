from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROWS_FILE = Path("results/error_rows_2023_rf.csv")
OUT_FILE = Path("results/error_rows_conditional_summary_2023.csv")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {ROWS_FILE}: {missing}")


def _make_bins(series: pd.Series, bins, labels) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)


def _summarize_group(df: pd.DataFrame, group_col: str, analysis_name: str) -> pd.DataFrame:
    # Drop NA groups to avoid junk rows
    tmp = df.copy()
    tmp[group_col] = tmp[group_col].astype("object")
    tmp = tmp[~tmp[group_col].isna()]

    if tmp.empty:
        return pd.DataFrame(columns=[
            "analysis", "group_col", "group_value",
            "n_obs", "n_errors", "error_rate",
            "share_of_all_errors", "share_of_all_obs"
        ])

    g = tmp.groupby(group_col, dropna=True).agg(
        n_obs=("error", "size"),
        n_errors=("error", "sum"),
        error_rate=("error", "mean"),
    ).reset_index().rename(columns={group_col: "group_value"})

    total_obs = len(df)
    total_err = float(df["error"].sum()) if total_obs > 0 else 0.0

    g["share_of_all_obs"] = g["n_obs"] / total_obs if total_obs else np.nan
    g["share_of_all_errors"] = g["n_errors"] / total_err if total_err else np.nan

    g.insert(0, "group_col", group_col)
    g.insert(0, "analysis", analysis_name)

    # Sort: biggest error_rate first, then most errors (useful for reading)
    g = g.sort_values(["error_rate", "n_errors"], ascending=[False, False])

    return g


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not ROWS_FILE.exists():
        raise FileNotFoundError(f"Cannot find: {ROWS_FILE}")

    df = pd.read_csv(ROWS_FILE)
    df.columns = df.columns.str.strip()

    # Must-have columns from your pipeline
    _ensure_cols(df, ["error", "y_true", "y_pred"])

    # Make sure error is 0/1 int
    df["error"] = pd.to_numeric(df["error"], errors="coerce").fillna(0).astype(int)

    # ------------------------------------------------------------
    # Create some useful derived columns (only if possible)
    # ------------------------------------------------------------
    # Grid bins (very relevant for "From Grid to Flag")
    if "grid" in df.columns:
        df["grid_bin"] = _make_bins(
            df["grid"],
            bins=[-np.inf, 5, 10, 15, 20, np.inf],
            labels=["P1-5", "P6-10", "P11-15", "P16-20", "P21+"],
        )

    # Round bins (moment of season)
    if "round" in df.columns:
        df["season_phase"] = _make_bins(
            df["round"],
            bins=[-np.inf, 7, 15, np.inf],
            labels=["early (1-7)", "mid (8-15)", "late (16+)"],
        )

    # Abs(delta) bins if present
    if "abs_delta" in df.columns:
        df["abs_delta_bin2"] = _make_bins(
            df["abs_delta"],
            bins=[-np.inf, 2, 5, 10, np.inf],
            labels=["<=2", "2-5", "5-10", "10+"],
        )

    # ------------------------------------------------------------
    # Analyses to run (only those whose columns exist)
    # ------------------------------------------------------------
    candidates = [
        ("By true class", "y_true"),
        ("By predicted class", "y_pred"),
        ("By driver", "driverId"),
        ("By team", "team_name"),
        ("By circuit/race", "raceId"),
        ("By round", "round"),
        ("By season phase", "season_phase"),
        ("By rainfall", "rainfall_any"),
        ("By finish group", "finish_group"),
        ("By abs(delta) bin", "abs_delta_bin"),
        ("By abs(delta) bin (alt)", "abs_delta_bin2"),
        ("By grid bin", "grid_bin"),
    ]

    all_tables = []
    for analysis_name, col in candidates:
        if col in df.columns:
            all_tables.append(_summarize_group(df, col, analysis_name))

    # One single CSV (long format)
    out = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"Saved: {OUT_FILE}")
    print(f"Rows: {len(out)}")
    print("Done.")


if __name__ == "__main__":
    main()