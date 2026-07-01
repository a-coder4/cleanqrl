import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: PyYAML. Install it with:\n"
        "  python -m pip install PyYAML\n"
        "or, if you use the Windows launcher:\n"
        "  py -m pip install PyYAML"
    ) from exc

from cleanqrl.experiment import LUNARLANDER_SEEDS, standardize_lunarlander_config
from cleanqrl_utils.train import train_agent


STANDARDIZED_RUNS = [
    {"kind": "config", "path": "configs/benchmarks/ppo_classical_lunarlander.yaml"},
    {"kind": "script", "agent": "PPO_tiny_classical", "trial_name": "ppo_tiny_classical"},
    {"kind": "config", "path": "configs/benchmarks/dqn_classical_lunarlander.yaml"},
    {"kind": "config", "path": "configs/benchmarks/dqn_quantum_lunarlander.yaml"},
    {"kind": "config", "path": "configs/benchmarks/ppo_quantum_lunarlander.yaml"},
]


def run_config(config_path: str, seed: int, repo_root: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = seed
    standardize_lunarlander_config(config)
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    config["trial_name"] = f"{timestamp}_{config['trial_name']}_seed{seed}"
    config["path"] = os.path.join(repo_root, "logs", config["trial_name"])

    os.makedirs(config["path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(config["path"], "source_config.yaml"))
    with open(os.path.join(config["path"], "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    train_agent(config)
    verify_completed_run(config["path"], config["trial_name"])


def run_tiny(seed: int, repo_root: str):
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    trial_name = f"{timestamp}_ppo_tiny_classical_seed{seed}"
    run_path = os.path.join(repo_root, "logs", trial_name)
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
        ],
        check=True,
    )
    verify_completed_run(run_path, trial_name)


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


def verify_completed_run(run_path: str, trial_name: str):
    config_path = os.path.join(run_path, "config.yaml")
    result_path = os.path.join(run_path, "result.json")
    if not os.path.exists(config_path):
        raise RuntimeError(f"{trial_name} did not write config.yaml")
    if not os.path.exists(result_path):
        raise RuntimeError(f"{trial_name} did not write result.json")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    records = read_jsonl(result_path)
    target_timesteps = int(config.get("total_timesteps", 2_000_000))
    interval = int(config.get("eval_interval", 100_000))
    max_timestep = max(
        [
            int(record.get("training_timestep", record.get("global_step", 0)) or 0)
            for record in records
        ]
        or [0]
    )
    eval_rows = [record for record in records if record.get("metric_type") == "evaluation"]
    eval_steps = sorted({int(record.get("training_timestep", 0)) for record in eval_rows})
    checkpoint_steps = []
    for name in os.listdir(run_path):
        if not name.endswith(".cleanqrl_model") or "_step" not in name:
            continue
        step_text = name.split("_step", 1)[1].split(".", 1)[0]
        if step_text.isdigit():
            checkpoint_steps.append(int(step_text))
    checkpoint_steps = sorted(set(checkpoint_steps))

    expected_steps = list(range(interval, target_timesteps + 1, interval))
    failures = []
    if max_timestep < target_timesteps:
        failures.append(f"max timestep {max_timestep}, expected {target_timesteps}")
    if eval_steps != expected_steps:
        failures.append(f"eval steps {eval_steps}, expected {expected_steps}")
    if checkpoint_steps != expected_steps:
        failures.append(f"checkpoint steps {checkpoint_steps}, expected {expected_steps}")
    if failures:
        raise RuntimeError(f"Incomplete fair-comparison run {trial_name}: " + "; ".join(failures))


def main():
    parser = argparse.ArgumentParser(
        description="Run the standardized LunarLander-v3 fair comparison."
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=LUNARLANDER_SEEDS,
        help="Seed set shared by every agent.",
    )
    parser.add_argument(
        "--skip-tiny",
        action="store_true",
        help="Skip the standalone PPO-tiny script.",
    )
    parser.add_argument(
        "--fairness-check",
        action="store_true",
        help="Generate fairness_check_report.md after all requested runs finish.",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Generate aggregate CSV, plots, and summary tables after requested runs finish.",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    for seed in args.seeds:
        for run_spec in STANDARDIZED_RUNS:
            if run_spec["kind"] == "script":
                if args.skip_tiny:
                    continue
                run_tiny(seed, repo_root)
            else:
                run_config(os.path.join(repo_root, run_spec["path"]), seed, repo_root)
    if args.fairness_check:
        subprocess.run(
            [
                sys.executable,
                os.path.join(repo_root, "analyze_lunarlander_fairness.py"),
            ],
            check=True,
        )
    if args.aggregate_results:
        subprocess.run(
            [
                sys.executable,
                os.path.join(repo_root, "aggregate_lunarlander_results.py"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
