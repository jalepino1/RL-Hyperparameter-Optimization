import pandas as pd
from typing import List, Tuple


def load_energy_data(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """
    Load the energy forecasting dataset and create train/val splits.

    Returns
    -------
    rl_train : DataFrame
    rl_val   : DataFrame
    features : list of feature column names
    target   : target column name ("load")
    """
    train = pd.read_csv(csv_path)

    # ---- Feature engineering (copied from your notebook) ----
    train["time"] = pd.to_datetime(train["time"])
    train["year"] = train["time"].dt.year
    train["month"] = train["time"].dt.month
    train["day"] = train["time"].dt.day
    train["hour"] = train["time"].dt.hour
    train["dayofweek"] = train["time"].dt.dayofweek
    train["dayofyear"] = train["time"].dt.dayofyear

    # Use the same features list you used in the notebook
    features = [
        "year",
        "month",
        "day",
        "hour",
        "dayofweek",
        "dayofyear",
        "Gb(i)",
        "Gd(i)",
        "H_sun",
        "T2m",
        "WS10m",
    ]

    target = "load"

    # 80/20 split (same logic you used)
    split_idx = int(len(train) * 0.8)
    rl_train = train.iloc[:split_idx].copy()
    rl_val = train.iloc[split_idx:].copy()

    return rl_train, rl_val, features, target
