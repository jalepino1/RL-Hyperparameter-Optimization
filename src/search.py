from typing import Dict, List, Tuple
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .env import EnergyForecastingEnv
from .agent import DQNAgent

def evaluate_lgbm_config(
    train_data,
    val_data,
    features: List[str],
    target: str,
    params: Dict,
    verbose: bool = False,
) -> float:
    """Train LightGBM with given hyperparameters and return RMSE."""
    X_train = train_data[features]
    y_train = train_data[target]
    X_val = val_data[features]
    y_val = val_data[target]

    model = lgb.LGBMRegressor(
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"],
        n_estimators=100,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # Older sklearn: mean_squared_error(...) doesn't accept squared=False
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)

    if verbose:
        print(f"Params: {params} | RMSE: {rmse:.4f}")

    return rmse

def generate_param_grid_from_env(env) -> List[Dict]:
    """Use the discrete hyperparameter grids defined in the environment.

    This guarantees that the grid search explores the **same space** as the RL
    environment.
    """
    grid = []
    for lr in env.lr_values:
        for depth in env.depth_values:
            for leaves in env.n_leaves_values:
                for mcs in env.min_child_samples_values:
                    grid.append(
                        {
                            "learning_rate": lr,
                            "max_depth": depth,
                            "num_leaves": leaves,
                            "min_child_samples": mcs,
                        }
                    )
    return grid


def run_limited_grid_search(
    train_data,
    val_data,
    features: List[str],
    target: str = "load",
    max_evals: int = 30,
    env: EnergyForecastingEnv | None = None,
    verbose: bool = True,
):
    """Grid search baseline under a fixed evaluation budget.

    Parameters
    ----------
    train_data, val_data : pd.DataFrame
        Split used for training/validation.
    features : list of str
        Input feature columns.
    target : str
        Target column name.
    max_evals : int
        Maximum number of model evaluations allowed.
    env : EnergyForecastingEnv or None
        If provided, its discrete hyperparameter values are used so that the
        grid search and RL share the same search space.
    """
    if env is not None:
        grid = generate_param_grid_from_env(env)
    else:
        # Fallback grid (kept small); you can tune these if needed
        lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
        depth_values = [3, 5, 7, 10]
        leaves_values = [15, 31, 63, 100]
        mcs_values = [5, 10, 20]
        grid = [
            {
                "learning_rate": lr,
                "max_depth": depth,
                "num_leaves": leaves,
                "min_child_samples": mcs,
            }
            for lr in lr_values
            for depth in depth_values
            for leaves in leaves_values
            for mcs in mcs_values
        ]

    best_rmse = float("inf")
    best_params = None
    results: List[Dict] = []
    eval_count = 0

    for params in grid:
        rmse = evaluate_lgbm_config(train_data, val_data, features, target, params, verbose=False)
        results.append({"params": params, "rmse": rmse})
        eval_count += 1

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

        if verbose and eval_count % 10 == 0:
            print(f"[Grid] Evaluation {eval_count}/{max_evals} | Best RMSE so far: {best_rmse:.4f}")

        if eval_count >= max_evals:
            break

    if verbose:
        print(f"\n[Grid] Done. Used {eval_count} evaluations.")
        print(f"[Grid] Best RMSE: {best_rmse:.4f}")
        print(f"[Grid] Best params: {best_params}")

    return best_rmse, best_params, results, eval_count


def train_rl_with_eval_budget(
    env: EnergyForecastingEnv,
    max_evals: int = 50,
    target_update_freq: int = 5,
    epsilon_decay: float = 0.99,
    verbose: bool = True,
):
    """Train a fresh DQN agent on the given environment under a hard budget
    on the number of model evaluations (environment steps).

    Returns
    -------
    best_rmse : float
    best_config : dict
    episode_rmses : list[float]
        Best RMSE observed in each episode (for plotting/comparison).
    step_history : list[dict]
        Per‑step records (eval index, reward, RMSE, epsilon, etc.).
    eval_count : int
        Total number of environment steps / model evaluations used.
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=0.0005,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=epsilon_decay,
        buffer_size=10000,
        batch_size=64,
        grad_clip=1.0,
    )

    eval_count = 0
    step_history: List[Dict] = []
    best_rmse = float("inf")
    best_config = None
    episode = 0

    while eval_count < max_evals:
        state, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done and eval_count < max_evals:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            _ = agent.train_step()  # loss not strictly needed here for reporting

            state = next_state
            episode_reward += reward
            eval_count += 1

            rmse = info.get("rmse", None)
            if rmse is not None and rmse < best_rmse:
                best_rmse = rmse
                best_config = {
                    "lr": info["lr"],
                    "max_depth": info["max_depth"],
                    "num_leaves": info["num_leaves"],
                    "min_child_samples": info["min_child_samples"],
                }

            step_history.append(
                {
                    "eval": eval_count,
                    "episode": episode + 1,
                    "reward": reward,
                    "cumulative_reward": episode_reward,
                    "rmse": rmse,
                    "epsilon": agent.epsilon,
                }
            )

        # Episode finished
        agent.update_target_network()
        agent.decay_epsilon()
        episode += 1

        if verbose:
            print(
                f"[RL] Episode {episode:3d} finished | "
                f"last RMSE: {env.current_rmse:.4f} | "
                f"best RMSE so far: {best_rmse:.4f} | "
                f"evals used: {eval_count}/{max_evals}"
            )

    if verbose:
        print(f"\n[RL] Done. Used {eval_count} evaluations.")
        print(f"[RL] Best RMSE: {best_rmse:.4f}")
        print(f"[RL] Best config: {best_config}")

    # Derive a simple per‑episode RMSE sequence for compatibility with compare_rl_vs_baseline
    episode_rmses: List[float] = []
    if step_history:
        last_ep = step_history[0]["episode"]
        rmses_in_ep: List[float] = []
        for s in step_history:
            if s["episode"] != last_ep:
                if rmses_in_ep:
                    episode_rmses.append(min(rmses_in_ep))
                rmses_in_ep = []
                last_ep = s["episode"]
            if s["rmse"] is not None:
                rmses_in_ep.append(s["rmse"])
        if rmses_in_ep:
            episode_rmses.append(min(rmses_in_ep))

    return best_rmse, best_config, episode_rmses, step_history, eval_count

def run_limited_budget_comparison(
    rl_train,
    rl_val,
    features: List[str],
    target: str = "load",
    max_evals: int = 60,
    max_steps_per_episode: int = 5,
    verbose: bool = True,
):
    """Compare RL vs Grid Search under the same evaluation budget.

    Assumes you have already defined `rl_train`, `rl_val`, and `features`
    (for example, by running STEP 1 & STEP 2 in the main notebook).
    """
    print("=" * 80)
    print("LIMITED‑BUDGET COMPARISON: RL vs Grid Search")
    print("=" * 80)

    # Environment with a small max_steps to keep RL relatively fast
    env = EnergyForecastingEnv(
        train_data=rl_train,
        val_data=rl_val,
        features=features,
        target=target,
        max_steps=max_steps_per_episode,
        include_stability_penalty=True,
    )

    # RL under budget
    rl_best_rmse, rl_best_config, rl_episode_rmses, rl_step_history, rl_eval_count = train_rl_with_eval_budget(
        env,
        max_evals=max_evals,
        verbose=verbose,
    )

    # Grid search under the same evaluation budget, using the same discrete hyperparameters
    grid_best_rmse, grid_best_params, grid_results, grid_eval_count = run_limited_grid_search(
        rl_train,
        rl_val,
        features,
        target=target,
        max_evals=max_evals,
        env=env,
        verbose=verbose,
    )

    print("\n" + "=" * 80)
    print("SUMMARY (Limited budget)")
    print("=" * 80)
    print(f"Budget (max evaluations):           {max_evals}")
    print(f"RL evaluations actually used:       {rl_eval_count}")
    print(f"Grid search evaluations used:       {grid_eval_count}")
    print(f"RL best RMSE under budget:          {rl_best_rmse:.4f}")
    print(f"Grid search best RMSE under budget: {grid_best_rmse:.4f}")

    return {
        "rl_best_rmse": rl_best_rmse,
        "rl_best_config": rl_best_config,
        "rl_eval_count": rl_eval_count,
        "grid_best_rmse": grid_best_rmse,
        "grid_best_params": grid_best_params,
        "grid_eval_count": grid_eval_count,
        "rl_episode_rmses": rl_episode_rmses,
        "rl_step_history": rl_step_history,
        "grid_results": grid_results,
    }


