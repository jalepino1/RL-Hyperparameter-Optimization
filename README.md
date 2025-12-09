# RL_Final_Project

Tested with Python 3.12 + packages from requirements.txt (installed globally).
On some macOS setups, a venv with LightGBM/OpenMP may cause a segmentation fault; using a global env or rebuilding LightGBM fixes it.


============================================================
RL FINAL PROJECT - LIMITED-BUDGET HYPERPARAMETER OPTIMIZATION
============================================================

This project implements a Reinforcement Learning (RL) approach for
hyperparameter optimization of a LightGBM forecasting model using a
Deep Q-Network (DQN) agent. The RL method is directly compared against
a traditional Grid Search, with both approaches restricted to the same
evaluation budget.

The system includes:
 - A custom Gymnasium RL environment
 - A DQN agent with replay buffer and target network
 - A LightGBM evaluation pipeline
 - A full limited-budget RL vs Grid Search comparison
 - Logging of training histories, RMSE trends, chosen actions,
   and stability adjustments

The executable entry point is `src/main.py`.

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------

RL_Final_Project/
 |
 ├── data/
 │    └── train.csv               # Dataset used for example
 |
 ├── results/                     # contains results given from every execution of main.py
 │  
 ├── src/
 │    ├── __init__.py             # Makes src a Python package
 │    ├── main.py                 # EXECUTABLE: Runs RL + Grid Search
 │    ├── agent.py                # DQN agent implementation
 │    ├── data_loader.py          # Loads and processes dataset
 │    ├── env.py                  # Custom RL environment
 │    ├── search.py               # Search algorithms + comparison
 │    
 |
 ├── notebooks/
 │    └── RL_Implementation.ipynb   # Development notebook
 |
 ├── requirements.txt
 └── readme.txt (this file)


------------------------------------------------------------
HOW TO RUN THE PROJECT
------------------------------------------------------------

1. Create and activate a virtual environment:

       python3 -m venv venv
       source venv/bin/activate

   macOS Apple Silicon users:
       python3 -m venv venv --system-site-packages

   (This avoids LightGBM OpenMP segmentation faults.)


2. Install required packages:

       pip install -r requirements.txt


3. Ensure the dataset exists:

       data/train.csv


4. Run the RL vs Grid Search comparison:

       python -m src.main --data data/train.csv --max-evals 60 --max-steps 20

Arguments:
   --data       Path to dataset CSV
   --max-evals  Maximum total LightGBM evaluations allowed
   --max-steps  Max actions taken by RL per episode


------------------------------------------------------------
HOW THE SYSTEM WORKS (ACCORDING TO CURRENT SOURCE CODE)
------------------------------------------------------------

------------------------------------------------------------
(1) DATA LOADING (src/data_loader.py)
------------------------------------------------------------

Your file performs:
 - CSV loading
 - Timestamp conversion
 - Feature extraction:
       year, month, day, hour, dayofweek, dayofyear
 - An 80/20 train/validation split
 - Returns:
       rl_train, rl_val, features, target

target is hardcoded as:
       "load"

------------------------------------------------------------
(2) RL ENVIRONMENT (src/env.py)
------------------------------------------------------------

The `EnergyForecastingEnv` defines:
 - Discrete action space of size 256 for hyperparameter selection
 - State vector of 12 elements including:
        rmse, mae, previous rmse, stability factor,
        action index encoding, step counters, trends

 - Hyperparameter grid:
        learning_rate ∈ {0.01, 0.02, 0.05, 0.1}
        max_depth ∈ {3, 5, 7, 10}
        num_leaves ∈ {10, 15, 20, 31}
        min_child_samples ∈ {10, 20, 30, 40}

 - LightGBM model training inside step()
 - Reward function includes:
        Negative RMSE
        + Improvement bonus
        + Stability adjustments (your env has include_stability flag)
        + Small action penalty/bonus

The environment tracks:
 - Current step
 - RMSE trend
 - Previous action
 - Stability dampening term

Episode ends when max_steps is reached.


------------------------------------------------------------
(3) DQN AGENT (src/agent.py)
------------------------------------------------------------

The implementation includes:
 - Neural network with:
       Linear → LayerNorm → ReLU → Dropout
       Linear → LayerNorm → ReLU → Dropout
       Linear(output)
 - Epsilon-greedy exploration
 - Replay buffer (deque, size=10,000)
 - Double-DQN target computation:
       next action chosen by online net
       Q-value from target net
 - Gradient clipping (`clip_grad_norm_`)
 - SmoothL1Loss (Huber)
 - Target network updates
 - Epsilon decay toward a minimum

The agent stores:
 - q_network
 - target_network
 - optimizer
 - replay memory
 - epsilon schedule


------------------------------------------------------------
(4) SEARCH METHODS (src/search.py)
------------------------------------------------------------

Includes:

1. evaluate_lgbm_config()
     - Builds a LightGBM model with given parameters
     - Computes RMSE and MAE
     - Used both by RL and Grid Search

2. run_limited_grid_search()
     - Iterates through parameter combinations
     - Stops when max_evals is reached
     - Tracks:
          best RMSE
          best parameters
          full evaluation history

3. run_limited_budget_comparison()
     - Creates RL environment + DQN agent
     - Runs RL episodes
     - Tracks:
          RL eval history
          RL best result
          Grid Search best result
     - Both RL and Grid Search share the same evaluation cap

Return dictionary includes:
     rl_best, grid_best, rl_history, grid_history, step_by_step logs


------------------------------------------------------------
(5) EXECUTABLE SCRIPT (src/main.py)
------------------------------------------------------------

main.py does the following:

   1. Parses command-line arguments
   2. Loads and prepares the dataset via data_loader
   3. Calls run_limited_budget_comparison()
   4. Saves results to results/results.json
   5. Prints summary information to console

This is the ONLY file run by the user.

Run via:

       python -m src.main ...


------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

The program produces:
 - RL best hyperparameters
 - Grid Search best hyperparameters
 - RMSE and MAE values
 - Action and reward histories
 - Evaluation histories
 - Stability trend data
 - Full results saved to:

       results/results.json