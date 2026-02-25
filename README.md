# EV Maintenance PoC

Ready-made example of a predictive maintenance system for electric vehicles.  
Python code with machine learning for forecasts and risk calculation.

## What it does
- Loads data from file or API
- Trains a GRU model
- Predicts battery degradation for the next 5 days
- Calculates breakdown risk
- Builds an interactive graph

## How to run
```bash
pip install numpy torch scipy matplotlib pandas
python ev_predictive_maintenance.py
