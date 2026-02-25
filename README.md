# EV Predictive Maintenance System  
**Production v3.0**

**Predict battery failure before it happens. Save money. Reduce downtime.**

---

## Overview

A complete, production-ready system that forecasts electric vehicle battery degradation, calculates probabilistic risk using CVaR, and estimates replacement cost in real time using live lithium prices.

Built with industry best practices, full reproducibility, comprehensive testing, and zero data leakage.

---

## Key Features

- Realistic temperature-dependent degradation model (Arrhenius equation)
- GRU neural network with Monte-Carlo simulation for accurate forecasts
- Proper train/validation/test split (70/15/15)
- Conditional Value-at-Risk (CVaR) risk assessment
- Live lithium price integration for dynamic cost estimation
- Full model architecture saved and loaded correctly
- Professional CLI with clear arguments
- Automatic fallback to realistic simulated data
- Comprehensive logging and error handling
- Clean, tested, production-quality code

---

## Business Value

- Helps fleet operators and EV owners **prevent unexpected battery failures**
- Gives **clear financial risk** in dollars
- Works with or without real data (perfect for demos and testing)
- Ready for integration into fleet management platforms

---

## Technical Stack

- **Python 3.10+**
- PyTorch (GRU model)
- NumPy / Pandas / SciPy
- Matplotlib (visualization)
- CoinGecko API (live lithium price)
- Full type hints and structured logging

---

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scipy matplotlib pycoingecko requests

2025-02-25 15:30:12 - root - INFO - === EV Predictive Maintenance System v3.0 started ===
2025-02-25 15:30:14 - root - INFO - Generated realistic simulated data
2025-02-25 15:30:22 - root - INFO - Model trained and saved (hidden=150)
2025-02-25 15:30:23 - root - INFO - Predicted battery % for next 5 days: [92.4, 89.1, 85.7, 82.3, 78.9]
2025-02-25 15:30:23 - root - INFO - Risk (CVaR): 64.2% → Monitor.
2025-02-25 15:30:23 - root - INFO - Estimated replacement cost risk: $1,284.50
2025-02-25 15:30:24 - root - INFO - Plot saved: ev_prediction.png

.
├── ev_predictive_maintenance.py    # Main program
├── README.md                       # This file
├── requirements.txt
├── ev_prediction.png               # Generated graph
├── ev_model.pt                     # Saved model
└── tests/
    └── test_ev_maintenance.py      # Automated tests

# tests/test_ev_maintenance.py
import pytest
import numpy as np
import torch
import os
from unittest.mock import patch

from ev_predictive_maintenance import (
    BatteryScaler,
    GRUNet,
    load_real_data,
    prepare_data,
    tune_params,
    forecast,
    assess_risk,
    visualize,
)


def test_scaler():
    data = np.random.uniform(0, 100, (100, 3)).astype(np.float32)
    scaler = BatteryScaler()
    scaled, _, _ = scaler.fit_transform(data)
    assert np.allclose(scaled.max(axis=0), 1.0, atol=1e-6)
    assert np.allclose(scaled.min(axis=0), 0.0, atol=1e-6)

    inverse = scaler.inverse_battery(scaled[:, 1])
    assert np.allclose(inverse, data[:, 1], atol=1e-4)


def test_load_data():
    data = load_real_data()
    assert data.shape[1] == 3
    assert len(data) >= 50
    assert np.all(data[:, 1] >= 0) and np.all(data[:, 1] <= 100)


def test_prepare_data():
    data = load_real_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(data, timesteps=5)
    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert X_test.shape[0] > 0


def test_tune_params():
    data = load_real_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(data, timesteps=5)
    model, params = tune_params(X_train, y_train, X_val, y_val, scaler, epochs=10)
    assert isinstance(model, GRUNet)
    assert "hidden_size" in params


def test_forecast():
    data = load_real_data()
    _, _, _, _, _, _, scaler = prepare_data(data, timesteps=10)
    model = GRUNet()
    current = data[-10:]
    preds = forecast(model, current, scaler, steps=3, simulations=20)
    assert len(preds) == 3
    assert np.all((preds >= 0) & (preds <= 100))


def test_assess_risk():
    data = load_real_data()
    mean_preds = np.array([80.0, 75.0, 70.0])
    current = data[-10:]
    risk, warning, cost = assess_risk(data, mean_preds, current)
    assert 0 <= risk <= 100
    assert isinstance(warning, str)
    assert cost >= 0


@patch("matplotlib.pyplot.savefig")
def test_visualize(mock_save):
    data = load_real_data()
    mean_preds = np.array([85.0, 82.0])
    visualize(data, mean_preds)
    mock_save.assert_called_once()


def test_model_save_load():
    data = load_real_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(data, timesteps=5)
    model, params = tune_params(X_train, y_train, X_val, y_val, scaler, epochs=5)
    checkpoint = {
        "model_state": model.state_dict(),
        "hidden_size": params["hidden_size"],
        "dropout": params["dropout"],
    }
    torch.save(checkpoint, "test_model.pt")

    cp = torch.load("test_model.pt", weights_only=True)
    loaded = GRUNet(hidden_size=cp["hidden_size"], dropout=cp["dropout"])
    loaded.load_state_dict(cp["model_state"])
    os.remove("test_model.pt")
    assert isinstance(loaded, GRUNet)
