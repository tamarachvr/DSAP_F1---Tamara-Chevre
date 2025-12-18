# DSAP Final Project — Formula 1 Performance Analysis

This project analyzes the relationship between starting grid position and race performance
in Formula 1 across the 2021–2023 seasons.

The objective is to predict whether a driver overperforms or underperforms relative to
their starting position using machine learning models.

## Research Question

To what extent does starting grid position influence race outcome in Formula 1, and
can machine learning models predict driver overperformance or underperformance
based on pre-race and race-related features?

## Methodology

- Construction of a consolidated dataset combining:
  - race results
  - weather conditions
  - safety car events
  - pit stops and standings
- Feature engineering including normalized grid position and contextual variables
- Definition of performance labels (overperformance / underperformance)
- Training and evaluation of multiple classification models
- Scenario-based race simulations (simple and Monte Carlo)

## Project Structure

DSAP_F1/
├── main.py                # Entry point (runs the full pipeline)
├── environment.yml        # Conda dependencies
├── README.md              # Project documentation
├── src/                    # Source code
│   ├── data/               # Data preparation scripts
│   ├── analysis/           # Modeling, evaluation, simulations
│   └── utils/              # Utility functions
├── data/
│   ├── global/             # Raw race data (not tracked)
│   ├── weather/            # Raw weather data (not tracked)
│   ├── safetycar/          # Raw safety car data (not tracked)
│   └── processed/          # Processed datasets
├── results/                # Model outputs and simulation results
└── notebooks/              # Exploratory notebooks

## Setup

Create the conda environment:

conda env create -f environment.yml
conda activate dsap-f1

## Usage

Run the full pipeline with:

python main.py

## Expected Output

- Model performance metrics saved in `results/`
- Feature importance analysis (Random Forest)
- Race simulation results (simple and Monte Carlo)

## Reproducibility

All models and simulations use fixed random seeds (`random_state=42`)
to ensure reproducible results.

## Requirements

- Python 3.10+
- Conda
