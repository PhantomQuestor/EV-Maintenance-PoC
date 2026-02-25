# EV Predictive Maintenance System 
Ready-made example of a predictive maintenance system for electric vehicles. Python code with machine learning. For buying full version or modifications - write to X (@PhantomQuestor)

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
