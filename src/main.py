# src/main.py

import argparse
from pathlib import Path
import json

from .data_loader import load_energy_data
from .search import run_limited_budget_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Limited-budget RL vs Grid Search for LightGBM hyperparameters."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/train.csv",
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=60,
        help="Maximum total model evaluations for RL and grid search.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=5,
        help="Maximum RL steps per episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file at {data_path}")

    print(f"Loading data from: {data_path}")
    rl_train, rl_val, features, target = load_energy_data(str(data_path))

    print("Running limited-budget comparison (RL vs Grid Search)...")
    results = run_limited_budget_comparison(
        rl_train=rl_train,
        rl_val=rl_val,
        features=features,
        target=target,
        max_evals=args.max_evals,
        max_steps_per_episode=args.max_steps_per_episode,
        verbose=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
