import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import yaml


TOTAL_TIMESTEPS = 900_000
EVAL_INTERVAL = 100_000
CHECKPOINT_INTERVAL = 100_000

CLASSICAL_RUNS = [
    {
        "kind": "config",
        "label": "ppo",
        "path": "configs/benchmarks/ppo_classical_lunarlander.yaml",
    },
    {
        "kind": "script",
        "label": "ppo_tiny",
        "agent": "PPO_tiny_classical",
        "trial_name": "ppo_tiny_classical_900k",
    },
    {
        "kind": "config",
        "label": "dqn",
        "path": "configs/benchmarks/dqn_classical_lunarlander.yaml",
    },
]


def read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_completed_run(run_path, trial_name, target_timesteps):
    result_path = os.path.join(run_path, "result.json")
    config_path = os.path.join(run_path, "config.yaml")
    if not os.path.exists(config_path):
        raise RuntimeError(f"{trial_name} did not write config.yaml")
    if not os.path.exists(result_path):
        raise RuntimeError(f"{trial_name} did not write result.json")

    records = read_jsonl(result_path)
    max_timestep = max(
        [
            int(record.get("training_timestep", record.get("global_step", 0)) or 0)
            for record in records
        ]
        or [0]
    )
    if max_timestep < target_timesteps:
        raise RuntimeError(
            f"{trial_name} stopped at {max_timestep}, expected {target_timesteps}"
        )


def align_ppo_rollout_batch(config, total_timesteps):
    if not str(config.get("agent", "")).lower().startswith("ppo"):
        return

    num_envs = int(config["num_envs"])
    original_num_steps = int(config["num_steps"])
    num_minibatches = int(config["num_minibatches"])
    original_batch_size = num_envs * original_num_steps
    if total_timesteps % original_batch_size == 0:
        return

    if total_timesteps % num_envs != 0:
        raise ValueError(
            f"{config['agent']} cannot run exactly {total_timesteps} steps with "
            f"{num_envs} envs. Choose a total timestep count divisible by {num_envs}."
        )

    per_env_timesteps = total_timesteps // num_envs
    candidates = []
    for candidate_num_steps in range(1, per_env_timesteps + 1):
        candidate_batch_size = num_envs * candidate_num_steps
        if per_env_timesteps % candidate_num_steps != 0:
            continue
        if candidate_batch_size % num_minibatches != 0:
            continue
        candidates.append(candidate_num_steps)

    if not candidates:
        raise ValueError(
            f"{config['agent']} cannot run exactly {total_timesteps} steps while "
            f"keeping num_envs={num_envs} and num_minibatches={num_minibatches}."
        )

    num_steps = min(candidates, key=lambda value: (abs(value - original_num_steps), value))
    config["num_steps"] = num_steps
    print(
        f"Adjusted {config['agent']} num_steps from {original_num_steps} to {num_steps} "
        f"so {num_envs * num_steps} step rollouts divide {total_timesteps} exactly."
    )


def configure_common_lunarlander_fields(config, seed, total_timesteps):
    from cleanqrl.experiment import standardize_lunarlander_config

    config["seed"] = seed
    config["training_budget_timesteps"] = total_timesteps
    config["total_timesteps"] = total_timesteps
    config["eval_interval"] = EVAL_INTERVAL
    config["checkpoint_interval"] = CHECKPOINT_INTERVAL
    config["eval_episodes"] = 10
    config["standardize_lunarlander"] = True
    config["observation_preprocessing"] = "none"
    config["wandb"] = False
    standardize_lunarlander_config(config)
    align_ppo_rollout_batch(config, total_timesteps)


def run_config(config_path, seed, repo_root, total_timesteps):
    from cleanqrl_utils.train import train_agent

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    configure_common_lunarlander_fields(config, seed, total_timesteps)
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    base_name = config["trial_name"]
    config["trial_name"] = f"{timestamp}_{base_name}_900k_seed{seed}"
    config["path"] = os.path.join(repo_root, "logs", config["trial_name"])

    os.makedirs(config["path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(config["path"], "source_config.yaml"))
    with open(os.path.join(config["path"], "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    print(f"\n=== Running {config['agent']} seed={seed} for {total_timesteps} steps ===")
    train_agent(config)
    verify_completed_run(config["path"], config["trial_name"], total_timesteps)


def run_tiny(seed, repo_root, total_timesteps):
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    trial_name = f"{timestamp}_ppo_tiny_classical_900k_seed{seed}"
    run_path = os.path.join(repo_root, "logs", trial_name)
    print(f"\n=== Running PPO_tiny_classical seed={seed} for {total_timesteps} steps ===")
    subprocess.run(
        [
            sys.executable,
            os.path.join(repo_root, "ppo_classical_lunarlander_tinyparam.py"),
            "--seed",
            str(seed),
            "--trial-name",
            trial_name,
            "--path",
            run_path,
            "--total-timesteps",
            str(total_timesteps),
            "--eval-interval",
            str(EVAL_INTERVAL),
            "--checkpoint-interval",
            str(CHECKPOINT_INTERVAL),
        ],
        check=True,
    )
    verify_completed_run(run_path, trial_name, total_timesteps)


def main():
    parser = argparse.ArgumentParser(
        description="Run PPO, PPO-tiny, and DQN on LunarLander-v3 at 900k steps."
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0],
        help="Seeds to run for every classical agent.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=TOTAL_TIMESTEPS,
        help="Shared training budget for every classical agent.",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    for seed in args.seeds:
        for run_spec in CLASSICAL_RUNS:
            if run_spec["kind"] == "script":
                run_tiny(seed, repo_root, args.total_timesteps)
            else:
                run_config(
                    os.path.join(repo_root, run_spec["path"]),
                    seed,
                    repo_root,
                    args.total_timesteps,
                )

    print(f"\nAll requested {args.total_timesteps} step classical LunarLander runs completed.")


if __name__ == "__main__":
    main()
