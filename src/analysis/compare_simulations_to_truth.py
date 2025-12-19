from pathlib import Path
import numpy as np
import pandas as pd

DATA_FILE = Path("data/processed/f1_over_under_multiclass_dataset.csv")
RESULTS_DIR = Path("results")

SCENARIOS = [
    ("azerbaijan_2023_street_dry", 2023, 4),
    ("monaco_2023_street_wet", 2023, 6),
    ("austrian_2023_perm_wet", 2023, 9),
    ("british_2023_perm_dry", 2023, 10),
]

# Change only if your label mapping is different
LABEL_MAP = {0: "under", 1: "neutral", 2: "over"}


def observed_counts_for_race(df: pd.DataFrame, year: int, round_: int) -> dict:
    race = df[(df["year"] == year) & (df["round"] == round_)]
    if race.empty:
        raise ValueError(f"No rows found for year={year}, round={round_}")

    counts = race["label"].value_counts().to_dict()
    return {
        "observed_under": float(counts.get(0, 0)),
        "observed_neutral": float(counts.get(1, 0)),
        "observed_over": float(counts.get(2, 0)),
        "n_drivers_observed": int(len(race)),
    }


def add_error_metrics_course_level(out_course: pd.DataFrame) -> pd.DataFrame:
    """
    Adds analytic error columns per race:
      - delta_* (model - observed)
      - abs_error_* = |delta_*|
      - squared_error_* = delta_*^2
      - rmse_* per class (same as sqrt(squared_error_*) at race level)
      - total_abs_error = sum(abs errors across classes)
    """
    for cls in ["under", "neutral", "over"]:
        out_course[f"delta_{cls}"] = out_course[f"expected_{cls}"] - out_course[f"observed_{cls}"]
        out_course[f"abs_error_{cls}"] = out_course[f"delta_{cls}"].abs()
        out_course[f"squared_error_{cls}"] = out_course[f"delta_{cls}"] ** 2
        out_course[f"rmse_{cls}"] = np.sqrt(out_course[f"squared_error_{cls}"])

    out_course["total_abs_error"] = (
        out_course["abs_error_under"] + out_course["abs_error_neutral"] + out_course["abs_error_over"]
    )
    return out_course


def make_error_summary(out_course: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a compact summary across the scenario races:
      - MAE per class (mean absolute error)
      - RMSE per class
      - mean total_abs_error per race
      - max total_abs_error and which scenario it happens in
      - bias per class (mean delta)
    """
    rows = []

    for cls in ["under", "neutral", "over"]:
        mae = float(out_course[f"abs_error_{cls}"].mean())
        rmse = float(np.sqrt(out_course[f"squared_error_{cls}"].mean()))
        bias = float(out_course[f"delta_{cls}"].mean())

        rows.append({"metric": f"MAE_{cls}", "value": mae})
        rows.append({"metric": f"RMSE_{cls}", "value": rmse})
        rows.append({"metric": f"Bias_mean_delta_{cls}", "value": bias})

    rows.append({"metric": "mean_total_abs_error_per_race", "value": float(out_course["total_abs_error"].mean())})

    worst_idx = out_course["total_abs_error"].idxmax()
    worst_row = out_course.loc[worst_idx]
    rows.append({"metric": "max_total_abs_error", "value": float(worst_row["total_abs_error"])})
    rows.append({"metric": "max_total_abs_error_scenario", "value": str(worst_row["scenario"])})

    return pd.DataFrame(rows)


def make_calibration_table_over(driver_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Simple calibration table for the 'over' class:
      - bin predicted p_over into n_bins
      - compare mean predicted probability vs observed frequency of true over

    This is useful in a report to discuss calibration (does p=0.7 mean ~70%?).
    """
    df = driver_df.copy()

    # True event: over == label 2
    df["true_over"] = (df["label"] == 2).astype(int)

    # Create bins on p_over
    df["p_over_bin"] = pd.cut(df["p_over"], bins=n_bins, include_lowest=True)

    calib = (
        df.groupby("p_over_bin")
        .agg(
            n=("true_over", "size"),
            mean_p_over=("p_over", "mean"),
            observed_over_rate=("true_over", "mean"),
        )
        .reset_index()
    )

    # Add a simple "calibration gap"
    calib["calibration_gap"] = calib["mean_p_over"] - calib["observed_over_rate"]
    return calib


def main():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    required = {"year", "round", "driverId", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Course-level comparison: observed counts vs expected counts
    # ------------------------------------------------------------------
    course_rows = []
    for scenario, year, rnd in SCENARIOS:
        sim_race_path = RESULTS_DIR / f"simulation_distribution_race_{scenario}.csv"
        if not sim_race_path.exists():
            raise FileNotFoundError(f"Missing simulation race file: {sim_race_path}")

        sim_race = pd.read_csv(sim_race_path).iloc[0].to_dict()
        obs = observed_counts_for_race(df, year, rnd)

        row = {
            "scenario": scenario,
            "year": year,
            "round": rnd,
            "rainfall_any": sim_race.get("rainfall_any", None),
            "n_drivers_sim": sim_race.get("n_drivers", None),
            **obs,
            "expected_under": float(sim_race.get("expected_under", 0)),
            "expected_neutral": float(sim_race.get("expected_neutral", 0)),
            "expected_over": float(sim_race.get("expected_over", 0)),
        }
        course_rows.append(row)

    out_course = pd.DataFrame(course_rows)

    # Save the “base” version (clean comparison)
    base_course_path = RESULTS_DIR / "simulation_vs_truth_course_2023.csv"
    out_course.to_csv(base_course_path, index=False)
    print(f"Saved: {base_course_path}")

    # Add analytic error metrics & save
    out_course_err = add_error_metrics_course_level(out_course)
    out_course_err_path = RESULTS_DIR / "simulation_vs_truth_course_2023_with_errors.csv"
    out_course_err.to_csv(out_course_err_path, index=False)
    print(f"Saved: {out_course_err_path}")

    # Summary analytics CSV
    summary = make_error_summary(out_course_err)
    summary_path = RESULTS_DIR / "simulation_error_summary_2023.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # ------------------------------------------------------------------
    # 2) Driver-level comparison (argmax class vs true label)
    # ------------------------------------------------------------------
    driver_tables = []
    for scenario, year, rnd in SCENARIOS:
        sim_driver_path = RESULTS_DIR / f"simulation_distribution_drivers_{scenario}.csv"
        if not sim_driver_path.exists():
            raise FileNotFoundError(f"Missing simulation drivers file: {sim_driver_path}")

        sim_dr = pd.read_csv(sim_driver_path)

        truth = df[(df["year"] == year) & (df["round"] == rnd)][["driverId", "label"]].copy()
        merged = sim_dr.merge(truth, on="driverId", how="left")

        if merged["label"].isna().any():
            raise ValueError(
                f"Some driverIds in {scenario} not found in dataset truth for year={year}, round={rnd}"
            )

        merged["pred_class"] = merged[["p_under", "p_neutral", "p_over"]].values.argmax(axis=1)
        merged["pred_label_name"] = merged["pred_class"].map(LABEL_MAP)
        merged["true_label_name"] = merged["label"].map(LABEL_MAP)
        merged["correct"] = (merged["pred_class"] == merged["label"]).astype(int)

        merged.insert(0, "scenario", scenario)
        merged.insert(1, "year", year)
        merged.insert(2, "round", rnd)

        driver_tables.append(
            merged[
                [
                    "scenario",
                    "year",
                    "round",
                    "driverId",
                    "team_name",
                    "p_under",
                    "p_neutral",
                    "p_over",
                    "pred_class",
                    "pred_label_name",
                    "label",
                    "true_label_name",
                    "correct",
                ]
            ]
        )

    out_driver = pd.concat(driver_tables, ignore_index=True)
    out_driver_path = RESULTS_DIR / "simulation_vs_truth_driver_2023.csv"
    out_driver.to_csv(out_driver_path, index=False)
    print(f"Saved: {out_driver_path}")

    # ------------------------------------------------------------------
    # 3) Calibration analysis (simple, report-friendly) for class "over"
    # ------------------------------------------------------------------
    calib_over = make_calibration_table_over(out_driver, n_bins=5)
    calib_path = RESULTS_DIR / "calibration_table_over_2023.csv"
    calib_over.to_csv(calib_path, index=False)
    print(f"Saved: {calib_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()