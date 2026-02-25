# ev_predictive_maintenance.py
# EV Predictive Maintenance System (Production v3.0)
# Fixed according to code review: pandas import, model architecture save/load,
# separate train/val/test split, specific exceptions, ordered imports

import argparse
import logging
import os
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from pycoingecko import CoinGeckoAPI
from scipy.optimize import curve_fit
from requests.exceptions import RequestException, Timeout, HTTPError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class BatteryScaler:
    """Robust min-max scaler focused on battery column."""

    def __init__(self) -> None:
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None

    def fit_transform(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.min_val = np.min(data, axis=0).astype(np.float32)
        self.max_val = np.max(data, axis=0).astype(np.float32)
        scaled = (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return scaled, self.min_val, self.max_val

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Scaler not fitted")
        return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)

    def inverse_battery(self, scaled: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Scaler not fitted")
        return scaled * (self.max_val[1] - self.min_val[1] + 1e-8) + self.min_val[1]


class GRUNet(nn.Module):
    """GRU network for battery level forecasting."""

    def __init__(self, input_size: int = 3, hidden_size: int = 100, dropout: float = 0.3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        h_n = self.dropout(h_n[0])
        return self.fc(h_n)


def load_real_data(
    api_key: Optional[str] = None, csv_path: str = "nrel_ev_data.csv"
) -> np.ndarray:
    """Load data: API → CSV → Arrhenius simulation."""
    data = None
    try:
        if api_key:
            response = requests.get(
                "https://developer.nrel.gov/api/argus/v1/ev-data",
                params={"api_key": api_key},
                timeout=10,
            )
            response.raise_for_status()
            df = pd.DataFrame(response.json().get("data", []))
            data = df[["temperature", "battery_level", "voltage"]].values
            logger.info("Data loaded from NREL API")
    except (RequestException, Timeout, HTTPError) as e:
        logger.warning(f"API failed: {e}, trying CSV...")
    except Exception as e:
        logger.exception("Unexpected API error")

    if data is None and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            data = df[["temperature", "battery_level", "voltage"]].values
            logger.info("Data loaded from CSV")
        except Exception as e:
            logger.exception("CSV read error")

    if data is None:
        num_days = 100
        temperatures = np.random.uniform(20, 40, num_days)
        T_k = temperatures + 273.15
        Ea, R = 0.5, 8.314e-3
        decay_factors = np.exp(-Ea / (R * T_k))
        battery_levels = (
            100 * np.cumprod(1 - 0.001 * decay_factors) + np.random.normal(0, 0.5, num_days)
        )
        battery_levels = np.clip(battery_levels, 0, 100)
        voltages = np.random.uniform(300, 400, num_days)
        data = np.column_stack((temperatures, battery_levels, voltages))
        logger.info("Generated realistic simulated data")

    data = data[~np.isnan(data).any(axis=1)]
    q1, q3 = np.quantile(data, [0.25, 0.75], axis=0)
    iqr = q3 - q1
    mask = ~((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))).any(axis=1)
    data = data[mask]
    data[:, 0] = np.clip(data[:, 0], -50, 150)
    data[:, 1] = np.clip(data[:, 1], 0, 100)
    data[:, 2] = np.clip(data[:, 2], 0, np.inf)

    if len(data) == 0:
        raise ValueError("No valid data after cleaning")
    return data.astype(np.float32)


def prepare_data(
    data: np.ndarray, timesteps: int = 10
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, BatteryScaler
]:
    """70% train / 15% val / 15% test split (no leakage)."""
    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    scaler = BatteryScaler()
    train_scaled, _, _ = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    def create_sequences(ds: np.ndarray, ts: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(ds) - ts):
            X.append(ds[i : i + ts])
            y.append(ds[i + ts, 1])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train, y_train = create_sequences(train_scaled, timesteps)
    X_val, y_val = create_sequences(val_scaled, timesteps)
    X_test, y_test = create_sequences(test_scaled, timesteps)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def tune_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: BatteryScaler,
    epochs: int = 100,
    batch_size: Optional[int] = None,
    lr: float = 0.001,
    patience: int = 10,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Hyperparameter tuning with early stopping on validation set."""
    if batch_size is None:
        batch_size = max(16, min(64, len(X_train) // 4))

    param_grid = [
        {"hidden_size": 50, "dropout": 0.2},
        {"hidden_size": 50, "dropout": 0.3},
        {"hidden_size": 100, "dropout": 0.2},
        {"hidden_size": 100, "dropout": 0.3},
        {"hidden_size": 150, "dropout": 0.3},
        {"hidden_size": 150, "dropout": 0.4},
    ]

    best_model = None
    best_val_loss = float("inf")
    best_params: Dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for p in param_grid:
        try:
            logger.info(f"Tuning: hidden={p['hidden_size']}, dropout={p['dropout']}")
            model = GRUNet(hidden_size=p["hidden_size"], dropout=p["dropout"]).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_state = None
            epochs_no_improve = 0
            model_val_loss = float("inf")

            for epoch in range(epochs):
                model.train()
                for i in range(0, len(X_train), batch_size):
                    batch_X = torch.from_numpy(X_train[i : i + batch_size]).to(device)
                    batch_y = torch.from_numpy(y_train[i : i + batch_size]).unsqueeze(1).to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_X = torch.from_numpy(X_val).to(device)
                    val_y = torch.from_numpy(y_val).unsqueeze(1).to(device)
                    val_loss = criterion(model(val_X), val_y).item()

                if val_loss < model_val_loss:
                    model_val_loss = val_loss
                    epochs_no_improve = 0
                    best_state = model.state_dict().copy()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        break

            if model_val_loss < best_val_loss:
                best_val_loss = model_val_loss
                best_model = GRUNet(hidden_size=p["hidden_size"], dropout=p["dropout"]).to(device)
                best_model.load_state_dict(best_state)
                best_params = p.copy()

        except Exception as e:
            logger.exception(f"Tuning failed for {p}")

    if best_model is None:
        logger.warning("All tuning failed, using default model")
        best_model = GRUNet().to(device)
        best_params = {"hidden_size": 100, "dropout": 0.3}

    return best_model, best_params


def forecast(
    model: nn.Module,
    current_input: np.ndarray,
    scaler: BatteryScaler,
    steps: int = 5,
    simulations: int = 100,
) -> np.ndarray:
    """Vectorized Monte-Carlo autoregressive forecast."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variance = float(np.var(current_input[:, 1]))

    noise = np.random.normal(0, np.sqrt(variance), (simulations, steps, current_input.shape[1])).astype(np.float32)
    sim_inputs = np.repeat(current_input[np.newaxis, :, :], simulations, axis=0) + noise

    all_preds = np.zeros((simulations, steps), dtype=np.float32)

    with torch.no_grad():
        for s in range(steps):
            scaled = scaler.transform(sim_inputs.reshape(-1, 3)).reshape(simulations, -1, 3)
            input_tensor = torch.from_numpy(scaled).float().to(device)
            pred_scaled = model(input_tensor).cpu().numpy().flatten()
            preds = scaler.inverse_battery(pred_scaled)
            all_preds[:, s] = preds

            new_temps = sim_inputs[:, -1, 0] + np.random.uniform(1, 3, simulations)
            new_voltages = sim_inputs[:, -1, 2] - np.random.uniform(0.01, 0.05, simulations)
            new_rows = np.stack([new_temps, preds, new_voltages], axis=1)
            sim_inputs = np.roll(sim_inputs, -1, axis=1)
            sim_inputs[:, -1, :] = new_rows

    return np.mean(all_preds, axis=0)


def assess_risk(
    data: np.ndarray, mean_preds: np.ndarray, current_input: np.ndarray, alpha: float = 0.95
) -> Tuple[float, str, float]:
    """CVaR risk + cost estimation with fallback."""

    def exp_decay(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        return a * np.exp(-b * x) + c

    x = np.arange(len(data), dtype=np.float32)
    try:
        popt, _ = curve_fit(
            exp_decay, x, data[:, 1], p0=(100.0, 0.005, 0.0), bounds=(0, [np.inf, np.inf, np.inf]), maxfev=10000
        )
        degradation_rate = float(popt[1])
    except Exception:
        degradation_rate = float(-(data[-1, 1] - data[0, 1]) / len(data))

    shortfalls = np.maximum(50.0 - mean_preds, 0.0)
    var = np.percentile(shortfalls, alpha * 100)
    cvar = float(np.mean(shortfalls[shortfalls >= var])) if np.any(shortfalls >= var) else float(var)

    risk_mean = cvar
    if current_input[-1, 0] > 140:
        risk_mean += 20
    if degradation_rate > 0.005:
        risk_mean += 15
    risk_mean = min(risk_mean, 100.0)

    warning = "High risk! Immediate action recommended." if risk_mean > 70 else "Monitor."

    try:
        cg = CoinGeckoAPI()
        price_data = cg.get_price(ids="lithium", vs_currencies="usd")
        lithium_price = float(price_data.get("lithium", {}).get("usd", 0.0))
    except Exception as e:
        logger.warning(f"CoinGecko unavailable: {e}")
        lithium_price = 0.0

    cost_per_kwh = 150.0 + lithium_price * 0.05
    battery_cost_est = cost_per_kwh * 60.0
    cost_risk = battery_cost_est * (risk_mean / 100)

    return risk_mean, warning, cost_risk


def visualize(data: np.ndarray, mean_preds: np.ndarray) -> None:
    """Save prediction plot."""
    import matplotlib.pyplot as plt

    historical_days = np.arange(len(data))
    future_days = np.arange(len(data), len(data) + len(mean_preds))

    plt.figure(figsize=(10, 5))
    plt.plot(historical_days, data[:, 1], label="Historical Battery %", color="blue")
    plt.plot(future_days, mean_preds, "r--", label="Predicted (MC mean)")
    plt.xlabel("Days")
    plt.ylabel("Battery Level (%)")
    plt.title("EV Battery Health Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("ev_prediction.png", dpi=300, bbox_inches="tight")
    logger.info("Plot saved: ev_prediction.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Predictive Maintenance CLI")
    parser.add_argument("--api-key", type=str, default=None, help="NREL API key (optional)")
    parser.add_argument("--timesteps", type=int, default=10, help="Sequence length")
    parser.add_argument("--steps", type=int, default=5, help="Forecast horizon (days)")
    parser.add_argument("--simulations", type=int, default=100, help="Monte-Carlo simulations")
    args = parser.parse_args()

    logger.info("=== EV Predictive Maintenance System v3.0 started ===")

    data = load_real_data(api_key=args.api_key)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(data, timesteps=args.timesteps)

    model_path = "ev_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        logger.info("Loading saved model")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model = GRUNet(
            hidden_size=checkpoint["hidden_size"], dropout=checkpoint["dropout"]
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model, best_params = tune_params(X_train, y_train, X_val, y_val, scaler)
        checkpoint = {
            "model_state": model.state_dict(),
            "hidden_size": best_params["hidden_size"],
            "dropout": best_params["dropout"],
        }
        torch.save(checkpoint, model_path)
        logger.info(f"Model trained and saved (hidden={best_params['hidden_size']})")

    current_input = data[-args.timesteps :]
    mean_preds = forecast(model, current_input, scaler, steps=args.steps, simulations=args.simulations)

    risk_mean, warning, cost_risk = assess_risk(data, mean_preds, current_input)

    logger.info(f"Predicted battery % for next {args.steps} days: {[round(p, 2) for p in mean_preds]}")
    logger.info(f"Risk (CVaR): {risk_mean:.1f}% → {warning}")
    if cost_risk > 0:
        logger.info(f"Estimated replacement cost risk: ${cost_risk:,.2f}")

    visualize(data, mean_preds)
    logger.info("=== Run completed successfully ===")
