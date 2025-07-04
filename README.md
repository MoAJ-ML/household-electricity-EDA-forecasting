# Electricity Demand Forecasting & Household Energy Analysis

This project performs end-to-end exploratory data analysis (EDA), clustering, and time-series forecasting on the UCI Individual household electric power consumption dataset (2008).

## Project Structure
- `electricity_analysis.py`: Main analysis and forecasting script
- `requirements.txt`: Python dependencies
- `summary.txt`: EDA summary and key findings
- `visualization/`: All generated plots (.png)

## Steps Performed
1. **Data Loading & Cleaning:** Loads and filters data for 2008, handles missing values
2. **EDA:** Visualizes trends, peaks, distributions, anomalies
3. **Clustering:** Identifies typical daily usage patterns
4. **Forecasting:** SARIMA and Prophet models for daily demand
5. **Results:** All plots saved to `visualization/`, summary in `summary.txt`

## Usage
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run main script:
   ```sh
   python electricity_analysis.py
   ```
3. Review plots in `visualization/` and findings in `summary.txt`

## Dataset
- [UCI: Individual household electric power consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

## Extending
- Add LSTM or dashboard integration as needed.
