# Dynamic Delta Hedging Simulation

This repository contains the Python codebase for the empirical simulation of Delta hedging under realistic transaction costs, evaluating the Black-Scholes-Merton (BSM) framework.

## Project Structure
* `GBM_Simulation.py`: Generates discrete geometric brownian motion price paths.
* `BSM_Engine.py`: Object-oriented BSM pricing and Delta calculation engine.
* `Hedging.py`: Simulates continuous (daily) and periodic (weekly) Delta hedging with frictional decay.
* `delta band hedging.py`: Implements the advanced Delta-Band threshold optimization strategy.
