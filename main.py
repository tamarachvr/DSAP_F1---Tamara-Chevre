"""
DSAP F1 project - main entry point.

Core pipeline (reproducible):
1) Build consolidated dataset
2) Build over/under-performance modelling dataset
3) Train models + run baselines + ids-vs-noids
3bis) Extra 2023 analyses (RF error analysis + conditional concentration)
4) Run simulations + compare simulations vs truth + analytics

Note:
- We keep full metric blocks (confusion matrix, classification report) because
  partial filtering breaks table readability.
"""

from pathlib import Path
import io
import re
import warnings
from contextlib import redirect_stdout, redirect_stderr

from src.data.prepare_global import main as prepare_global_main
from src.data.prepare_safetycar import main as prepare_safetycar_main
from src.data.prepare_weather import main as prepare_weather_main
from src.data.merge_safetycar_with_base import main as merge_safetycar_main
from src.data.merge_final_global_safety_weather import (
    main as merge_final_global_weather_main,
)

from src.analysis.add_standings_and_pitstops import main as finalcomplete_main
from src.analysis.build_over_under_dataset import (
    create_over_under_dataset as build_over_under_main,
)

from src.analysis.train_over_under_model import main as train_over_under_main
from src.analysis.run_simulation_scenarios import main as run_simulation_scenarios_main


# ---------------------------------------------------------------------
# Output files used to optionally skip heavy steps (FAST_MODE)
# ---------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")

FINAL_DATASET = PROCESSED_DIR / "f1_finalcomplete_global_sc_weather_standings_pitstop.csv"
OVER_UNDER_DATASET = PROCESSED_DIR / "f1_over_under_multiclass_dataset.csv"

MODEL_LR = MODELS_DIR / "logistic_regression_multiclass.joblib"
MODEL_RF = MODELS_DIR / "random_forest_multiclass.joblib"
MODEL_GB = MODELS_DIR / "gradient_boosting_multiclass.joblib"


def exists_all(paths):
    return all(p.exists() for p in paths)


# ---------------------------------------------------------------------
# Output filtering
# ---------------------------------------------------------------------
_DROP_PATTERNS = [
    r"FutureWarning",
    r"UserWarning",
    r"DeprecationWarning",
    r"^Traceback",
    r"File \".*\", line \d+",
]


def _is_noise(line: str) -> bool:
    s = line.rstrip("\n")
    if any(re.search(p, s, flags=re.IGNORECASE) for p in _DROP_PATTERNS):
        return True
    return False


def run_quiet(step_title: str, fn):
    """
    Runs fn() while capturing stdout/stderr.

    Goal:
    - Remove noisy warnings/traceback spam
    - Keep FULL metric blocks (confusion matrix + classification report)
      so tables stay readable.
    - If a step fails, show a short error and continue.
    """
    print(f"\n  ── {step_title}")
    buf_out, buf_err = io.StringIO(), io.StringIO()

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            fn()
    except Exception as e:
        print(f"  [FAILED] {step_title}: {e}")
        print("  -> Pipeline continues.\n")
        return

    lines = (buf_out.getvalue() + "\n" + buf_err.getvalue()).splitlines()

    kept = [ln for ln in lines if not _is_noise(ln)]
    kept = [ln for ln in kept if not ln.strip().lower().startswith("loading")]

    if any(ln.strip() for ln in kept):
        print("\n".join(kept).rstrip())
    else:
        print("  (No output captured.)")


# ---------------------------------------------------------------------
# Pretty terminal helpers (titles + printing full CSVs)
# ---------------------------------------------------------------------
def print_big_step(title: str):
    bar = "═" * 90
    print(f"\n{bar}\n{title}\n{bar}")


def print_sub_step(title: str):
    print(f"\n  ──────────────────────────────────────────────────────────────")
    print(f"  {title}")
    print(f"  ──────────────────────────────────────────────────────────────")


def _safe_read_csv(path: Path):
    try:
        import pandas as pd

        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def _print_df_full(df, title: str):
    """
    Print the FULL dataframe (no head/tail truncation).
    """
    print_sub_step(title)

    if df is None or len(df) == 0:
        print("  (No data / file missing.)")
        return

    try:
        import pandas as pd

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            200,
            "display.max_colwidth",
            None,
        ):
            print(df.to_string(index=False))
    except Exception:
        print(df)


# ---------------------------------------------------------------------
# Extra scripts (ONLY the ones we want)
# ---------------------------------------------------------------------
def run_train_baselines():
    from src.analysis.train_baselines import main as train_baselines_main

    train_baselines_main()


def run_train_ids_vs_noids():
    from src.analysis.train_ids_vs_noids import main as train_ids_vs_noids_main

    train_ids_vs_noids_main()


def run_statistical_analysis_2023():
    # Produces:
    # - error_analysis_summary_2023.csv
    # - error_rate_by_abs_delta_bin_2023.csv
    # - error_rate_by_finish_group_2023.csv
    # - error_rate_by_rain_2023.csv
    # - error_rate_by_true_class_2023.csv
    # - error_rows_2023_rf.csv
    # - rf_permutation_importance_2023_all.csv
    # - rf_permutation_importance_2023_interpretable.csv
    from src.analysis.statistical_analysis_2023 import main as statistical_analysis_main

    statistical_analysis_main()


def run_error_rows_conditional_analysis_2023():
    # Produces:
    # - error_rows_conditional_summary_2023.csv
    from src.analysis.error_rows_conditional_analysis_2023 import main as conditional_main

    conditional_main()


def run_compare_simulations_to_truth():
    # Produces:
    # - simulation_error_summary_2023.csv
    # - plus other simulation_vs_truth outputs
    from src.analysis.compare_simulations_to_truth import main as compare_main

    compare_main()


def print_key_2023_error_analysis_outputs_full():
    """
    Print FULL contents of the 2023 RF error-analysis outputs.
    (Note: we intentionally do NOT print error_rows_2023_rf.csv in main.)
    """
    print_big_step("📌 2023 RF ERROR ANALYSIS — FULL CSV OUTPUTS (results/)")

    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "error_analysis_summary_2023.csv"),
        "error_analysis_summary_2023.csv",
    )
    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "error_rate_by_true_class_2023.csv"),
        "error_rate_by_true_class_2023.csv",
    )
    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "error_rate_by_rain_2023.csv"),
        "error_rate_by_rain_2023.csv",
    )
    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "error_rate_by_finish_group_2023.csv"),
        "error_rate_by_finish_group_2023.csv",
    )
    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "error_rate_by_abs_delta_bin_2023.csv"),
        "error_rate_by_abs_delta_bin_2023.csv",
    )

    # NOTE: rf_permutation_importance_2023_all.csv is intentionally NOT printed (too heavy for main)
    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "rf_permutation_importance_2023_interpretable.csv"),
        "rf_permutation_importance_2023_interpretable.csv",
    )

    # Based on error_rows_2023_rf.csv (row-level errors file, not printed in main)
    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "error_rows_conditional_summary_2023.csv"),
        "error_rows_conditional_summary_2023.csv",
    )


def print_simulation_analytics_summary_full():
    print_big_step("📌 SIMULATION VALIDATION — FULL ANALYTICS SUMMARY (results/)")
    _print_df_full(
        _safe_read_csv(RESULTS_DIR / "simulation_error_summary_2023.csv"),
        "simulation_error_summary_2023.csv",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")

    print_big_step("🏁 DSAP F1 PIPELINE — START")
    FAST_MODE = True

    # =========================================================
    # 1) DATA PREPARATION
    # =========================================================
    print_big_step("1️⃣  DATA PREPARATION")

    if FAST_MODE and FINAL_DATASET.exists():
        print("  ✅ SKIP — final processed dataset already exists")
        print(f"  ↳ {FINAL_DATASET}")
    else:
        print("  ▶ Build consolidated dataset")
        prepare_global_main()
        prepare_safetycar_main()
        prepare_weather_main()
        merge_safetycar_main()
        merge_final_global_weather_main()
        finalcomplete_main()
        print("  ✅ Done")
        print(f"  ↳ Output: {FINAL_DATASET}")

    # =========================================================
    # 2) MODELLING DATASET
    # =========================================================
    print_big_step("2️⃣  MODELLING DATASET (OVER/UNDER)")

    if FAST_MODE and OVER_UNDER_DATASET.exists():
        print("  ✅ SKIP — over/under dataset already exists")
        print(f"  ↳ {OVER_UNDER_DATASET}")
    else:
        print("  ▶ Build over/under dataset")
        build_over_under_main()
        print("  ✅ Done")
        print(f"  ↳ Output: {OVER_UNDER_DATASET}")

    # =========================================================
    # 3) TRAINING + BASELINES + IDS VS NO-IDS
    # =========================================================
    print_big_step("3️⃣  TRAINING + MODEL EVALUATION")

    run_quiet("train_over_under_model.py  → models + metrics", train_over_under_main)
    run_quiet("train_baselines.py        → baselines", run_train_baselines)
    run_quiet("train_ids_vs_noids.py     → identity vs no-identity", run_train_ids_vs_noids)

    # =========================================================
    # 3bis) EXTRA: 2023 ERROR ANALYSIS + CONDITIONAL CONCENTRATION
    # =========================================================
    print_big_step("3️⃣🅱️  EXTRA — 2023 ERROR ANALYSIS (RF) + CONDITIONAL CONCENTRATION")

    run_quiet(
        "statistical_analysis_2023.py → RF importance + error tables + error_rows",
        run_statistical_analysis_2023,
    )
    run_quiet(
        "error_rows_conditional_analysis_2023.py → conditional results based on error_rows_2023_rf.csv",
        run_error_rows_conditional_analysis_2023,
    )

    # Print FULL CSV outputs (no truncation)
    print_key_2023_error_analysis_outputs_full()

    # =========================================================
    # 4) SIMULATIONS + VALIDATION VS TRUTH
    # =========================================================
    print_big_step("4️⃣  SIMULATIONS + VALIDATION VS TRUTH")

    run_quiet(
        "run_simulation_scenarios.py  → scenario simulations (Monte Carlo)",
        run_simulation_scenarios_main,
    )
    run_quiet(
        "compare_simulations_to_truth.py → compare simulations vs observed + analytics",
        run_compare_simulations_to_truth,
    )

    # Print FULL simulation analytics summary (no truncation)
    print_simulation_analytics_summary_full()

    # =========================================================
    # DONE
    # =========================================================
    print_big_step("✅ PIPELINE DONE — OUTPUTS")
    print(f"  results/ folder: {RESULTS_DIR.resolve()}")
    print("\n🏁 DSAP F1 pipeline: done")


if __name__ == "__main__":
    main()
