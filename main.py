"""
DSAP F1 project - main entry point.

Core pipeline (reproducible):
1) Build consolidated dataset
2) Build over/under-performance modelling dataset
3) Train models + run baselines + ids-vs-noids (clean-ish output)
4) Run simulations

Note:
- statistical_analysis_2023.py and error_analysis_2023.py are NOT run from main.
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
    print(f"\n=== {step_title} ===")
    buf_out, buf_err = io.StringIO(), io.StringIO()

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            fn()
    except Exception as e:
        print(f"[FAILED] {step_title}: {e}")
        print("-> Pipeline continues.\n")
        return

    lines = (buf_out.getvalue() + "\n" + buf_err.getvalue()).splitlines()

    # Keep almost everything, only drop obvious noise
    kept = [ln for ln in lines if not _is_noise(ln)]

    # Also drop super generic "Loading ..." lines to keep it clean
    kept = [ln for ln in kept if not ln.strip().lower().startswith("loading")]

    # Print
    if any(ln.strip() for ln in kept):
        print("\n".join(kept).rstrip())
    else:
        print("(No output captured.)")


# ---------------------------------------------------------------------
# Extra scripts (ONLY the ones you want)
# ---------------------------------------------------------------------
def run_train_baselines():
    from src.analysis.train_baselines import main as train_baselines_main

    train_baselines_main()


def run_train_ids_vs_noids():
    from src.analysis.train_ids_vs_noids import main as train_ids_vs_noids_main

    train_ids_vs_noids_main()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")

    print("=== DSAP F1 pipeline: start ===")
    FAST_MODE = True

    # ------------------------------
    # 1) Data preparation
    # ------------------------------
    if FAST_MODE and FINAL_DATASET.exists():
        print("[1/4] Data prep: SKIP (final processed dataset already exists)")
    else:
        print("[1/4] Data prep: build consolidated dataset")
        prepare_global_main()
        prepare_safetycar_main()
        prepare_weather_main()
        merge_safetycar_main()
        merge_final_global_weather_main()
        finalcomplete_main()
        print(f"Data prep done. Output: {FINAL_DATASET}")

    # ------------------------------
    # 2) Modelling dataset build
    # ------------------------------
    if FAST_MODE and OVER_UNDER_DATASET.exists():
        print("[2/4] Modelling dataset: SKIP (over/under dataset already exists)")
    else:
        print("[2/4] Modelling dataset: build over/under dataset")
        build_over_under_main()
        print(f"Dataset done. Output: {OVER_UNDER_DATASET}")

    # ------------------------------
    # 3) Training + baselines + ids-vs-noids (before simulations)
    # ------------------------------
    print("[3/4] Training/analysis: print results (readable tables)")
    run_quiet("train_over_under_model.py (models + metrics)", train_over_under_main)
    run_quiet("train_baselines.py (baselines)", run_train_baselines)
    run_quiet("train_ids_vs_noids.py (ids vs no-ids)", run_train_ids_vs_noids)

    # ------------------------------
    # 4) Simulations
    # ------------------------------
    print("\n[4/4] Simulations: run scenario simulations (simple + Monte Carlo)")
    run_simulation_scenarios_main()
    print(f"Simulations done. Check: {RESULTS_DIR.resolve()}")

    print("=== DSAP F1 pipeline: done ===")


if __name__ == "__main__":
    main()