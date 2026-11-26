# Synthetic Financial Time-Series Generator for Stress Testing

A beginner-friendly Streamlit showcase project for loading financial data, generating synthetic time-series data, applying stress scenarios, and analyzing risk.

## Project Structure

- `app.py` - Streamlit application entry point and user interface.
- `modules/data_loader.py` - Loads financial time-series data from files or external sources.
- `modules/preprocessing.py` - Cleans and prepares raw data for analysis and generation.
- `modules/synthetic_generator.py` - Generates synthetic financial time-series data.
- `modules/stress_scenarios.py` - Defines and applies market stress scenarios.
- `modules/validation.py` - Checks generated data for validity and compares it with source statistics.
- `modules/risk_analysis.py` - Calculates risk metrics and summarizes risk results.
- `requirements.txt` - Python dependencies for the project.
- `.gitignore` - Files and folders that Git should ignore.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

This is the initial project scaffold. Python module implementations will be added incrementally.
