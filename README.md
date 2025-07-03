# Volatility Prediction App

This app forecasts stock prices using GARCH-family models (GARCH, EGARCH, TGARCH) and displays predictions for the next 10 days.

## How to Deploy on Streamlit Cloud

1. **Upload the repository to GitHub.**
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud) and create a new app.**
3. **Set `app.py` as the main file.**
4. **Ensure `requirements.txt` is present (see below).**

## Requirements
The following Python packages are required (see `requirements.txt`):
- streamlit
- pandas
- numpy
- arch

## Usage
- Upload a CSV file with columns: `Date`, `Close Price`.
- The app will fit GARCH, EGARCH, and TGARCH models, select the best one, and forecast the next 10 days. 