"""Launch the exact matched 900k LunarLander-v3 experiment cohort."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cleanqrl.experiment import standardize_lunarlander_config
from lunarlander_900k_common import (
    AGENT_ORDER,
    AGENT_SPECS,
    CHECKPOINT_INTERVAL,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    EXPECTED_STEPS,
    REPO_ROOT,
    SEEDS,
    SUCCESS_THRESHOLD,
    TOTAL_TIMESTEPS,
    display_combo,
    validate_candidate,
    validate_cohort,
)


def align_ppo_rollout_batch(config: dict[str, Any], total_timesteps: int) -> None:
    """Choose the closest rollout length whose batch divides the exact budget."""
    if not str(config.get("agent", "")).lower().startswith("ppo"):
        return

    num_envs = int(config["num_envs"])
    original_num_steps = int(config["num_steps"])
    num_minibatches = int(config["num_minibatches"])
    original_batch_size = num_envs * original_num_steps
    if total_timesteps % original_batch_size == 0:
        config["batch_size"] = original_batch_size
        config["minibatch_size"] = original_batch_size // num_minibatches
        return
    if total_timesteps % num_envs:
        raise ValueError(
            f"{config['agent']} cannot run exactly {total_timesteps} interactions "
            f"with num_envs={num_envs}."
        )

    per_env_timesteps = total_timesteps // num_envs
    candidates = []
    for candidate_num_steps in range(1, per_env_timesteps + 1):
        batch_size = num_envs * candidate_num_steps
        if per_env_timesteps % candidate_num_steps == 0:
            if batch_size % num_minibatches == 0:
                candidates.append(candidate_num_steps)
    if not candidates:
        raise ValueError(
            f"No rollout length preserves num_envs={num_envs}, "
            f"num_minibatches={num_minibatches}, and an exact {total_timesteps} budget."
        )

    num_steps = min(candidates, key=lambda value: (abs(value - original_num_steps), value))
    batch_size = num_envs * num_steps
    config["num_steps"] = num_steps
    config["batch_size"] = batch_size
    config["minibatch_size"] = batch_size // num_minibatches
    config["rollout_alignment_note"] = (
        f"num_steps adjusted from {original_num_steps} to {num_steps}; "
        f"batch_size={batch_size} divides total_timesteps={total_timesteps} exactly"
    )


def _load_base_config(friendly_agent: str) -> dict[str, Any]:
    if friendly_agent == "ppo_tiny":
        from ppo_classical_lunarlander_tinyparam import CONFIG

        return copy.deepcopy(CONFIG)
    config_path = REPO_ROOT / AGENT_SPECS[friendly_agent]["config"]
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_run_config(
    friendly_agent: str,
    seed: int,
    run_name: str,
    run_path: Path,
) -> dict[str, Any]:
    config = _load_base_config(friendly_agent)
    config.update(
        {
            "agent": AGENT_SPECS[friendly_agent]["agent_key"],
            "env_id": "LunarLander-v3",
            "seed": seed,
            "seed_set": list(SEEDS),
            "training_budget_timesteps": TOTAL_TIMESTEPS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_interval": EVAL_INTERVAL,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "eval_episodes": EVAL_EPISODES,
            "success_reward_threshold": SUCCESS_THRESHOLD,
            "standardize_lunarlander": True,
            "observation_preprocessing": "none",
            "action_remapping": "none",
            "evaluation_policy": "deterministic_greedy",
            "evaluation_exploration": False,
            "protocol_name": "matched_lunarlander_900k_v1",
            "trial_name": run_name,
            "path": str(run_path.resolve()),
            "wandb": False,
            "save_model": True,
        }
    )
    standardize_lunarlander_config(config)
    align_ppo_rollout_batch(config, TOTAL_TIMESTEPS)
    if friendly_agent == "qrl":
        from cleanqrl.ppo_quantum_hybrid import n_layers, n_qubits

        config["num_qubits"] = int(n_qubits)
        config["num_layers"] = int(n_layers)
    return config


def _new_run_identity(friendly_agent: str, seed: int) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    name = f"{timestamp}_matched_900k_{friendly_agent}_seed{seed}"
    return name, REPO_ROOT / "logs" / name


def _print_config(friendly_agent: str, seed: int, config: dict[str, Any]) -> None:
    label = AGENT_SPECS[friendly_agent]["label"]
    print(f"\n--- {label} seed {seed} ---")
    keys = [
        "agent",
        "seed",
        "env_id",
        "total_timesteps",
        "num_envs",
        "num_steps",
        "batch_size",
        "num_minibatches",
        "minibatch_size",
        "num_qubits",
        "num_layers",
        "eval_interval",
        "eval_episodes",
        "checkpoint_interval",
        "observation_preprocessing",
        "action_remapping",
        "evaluation_policy",
        "evaluation_exploration",
        "success_reward_threshold",
        "path",
    ]
    for key in keys:
        if key in config:
            print(f"{key}: {config[key]}")
    print("evaluation_checkpoints: " + ", ".join(str(step) for step in EXPECTED_STEPS))
    if config.get("rollout_alignment_note"):
        print(config["rollout_alignment_note"])


def _write_run_config(
    friendly_agent: str, config: dict[str, Any], run_path: Path
) -> None:
    run_path.mkdir(parents=True, exist_ok=False)
    source = REPO_ROOT / AGENT_SPECS[friendly_agent]["config"]
    if source.is_file():
        shutil.copy2(source, run_path / source.name)
    with (run_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)


def _run_tiny(config: dict[str, Any], run_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "ppo_classical_lunarlander_tinyparam.py"),
            "--seed",
            str(config["seed"]),
            "--trial-name",
            str(config["trial_name"]),
            "--path",
            str(run_path),
            "--config",
            str(run_path / "config.yaml"),
            "--total-timesteps",
            str(TOTAL_TIMESTEPS),
            "--eval-interval",
            str(EVAL_INTERVAL),
            "--checkpoint-interval",
            str(CHECKPOINT_INTERVAL),
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def _run_config_agent(config: dict[str, Any]) -> None:
    from cleanqrl_utils.train import train_agent

    train_agent(config)


def run_one(friendly_agent: str, seed: int) -> Path:
    run_name, run_path = _new_run_identity(friendly_agent, seed)
    config = build_run_config(friendly_agent, seed, run_name, run_path)
    print(f"\n=== Running {AGENT_SPECS[friendly_agent]['label']} seed {seed} ===")
    _write_run_config(friendly_agent, config, run_path)
    if friendly_agent == "ppo_tiny":
        # The standalone script writes its final effective configuration again.
        _run_tiny(config, run_path)
    else:
        _run_config_agent(config)

    validation = validate_candidate(run_path)
    if not validation.valid:
        raise RuntimeError(
            f"Completed process did not produce a valid 900k run at {run_path}: "
            + validation.reason
        )
    print(f"Validated completed run: {run_path}")
    return run_path


def _deduplicate(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=AGENT_ORDER,
        default=list(AGENT_ORDER),
        help="Friendly agent names to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=SEEDS,
        default=list(SEEDS),
        help="Matched cohort seeds.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip only agent/seed pairs with a completely valid dedicated 900k run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print effective configurations without creating directories or training.",
    )
    args = parser.parse_args()
    agents = _deduplicate(args.agents)
    seeds = _deduplicate(args.seeds)

    cohort = validate_cohort(REPO_ROOT / "logs")
    requested = [(agent, seed) for agent in agents for seed in seeds]
    missing_requested = [combo for combo in requested if combo not in cohort.selected]
    print("Current exact missing run set for this request:")
    if missing_requested:
        for combo in missing_requested:
            print(f"  - {display_combo(combo)}")
    else:
        print("  none")

    for friendly_agent, seed in requested:
        if args.skip_completed and (friendly_agent, seed) in cohort.selected:
            selected = cohort.selected[(friendly_agent, seed)]
            print(
                f"Skipping {display_combo((friendly_agent, seed))}: "
                f"validated run {selected.path}"
            )
            continue
        if args.dry_run:
            run_name, run_path = _new_run_identity(friendly_agent, seed)
            config = build_run_config(friendly_agent, seed, run_name, run_path)
            _print_config(friendly_agent, seed, config)
        else:
            run_one(friendly_agent, seed)

    if args.dry_run:
        print("\nDry run only: no log directories were created and no training was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
