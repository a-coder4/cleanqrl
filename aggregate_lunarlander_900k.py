"""Build the matched-900k LunarLander dataset, statistics, tables, and text."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from lunarlander_900k_common import (
    AGENT_ORDER,
    AGENT_SPECS,
    EVAL_EPISODES,
    EXPECTED_STEPS,
    REPO_ROOT,
    SEEDS,
    SUCCESS_THRESHOLD,
    TOTAL_TIMESTEPS,
    CohortValidation,
    CandidateValidation,
    as_float,
    display_combo,
    record_timestep,
    validate_cohort,
)


ASSET_DIR = Path("lunarlander_matched_900k_assets")
T_CRITICAL_DF2 = 4.302652729911275
Z_975 = 1.959963984540054


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def sample_std(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.stdev(values) if len(values) >= 2 else None


def student_t_ci(values: Iterable[float]) -> tuple[float | None, float | None]:
    values = list(values)
    if len(values) != 3:
        return None, None
    center = statistics.mean(values)
    spread = statistics.stdev(values)
    half_width = T_CRITICAL_DF2 * spread / math.sqrt(3)
    return center - half_width, center + half_width


def wilson_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    proportion = successes / trials
    denominator = 1 + Z_975**2 / trials
    center = (proportion + Z_975**2 / (2 * trials)) / denominator
    half_width = (
        Z_975
        * math.sqrt(proportion * (1 - proportion) / trials + Z_975**2 / (4 * trials**2))
        / denominator
    )
    return center - half_width, center + half_width


def _latest_value(records: list[dict[str, Any]], key: str) -> float | None:
    candidates = [
        (record_timestep(record), as_float(record.get(key)))
        for record in records
        if as_float(record.get(key)) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda pair: (-1 if pair[0] is None else pair[0]))[1]


def _max_value(records: list[dict[str, Any]], key: str) -> float | None:
    values = [as_float(record.get(key)) for record in records]
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _final_evaluation(candidate: CandidateValidation) -> dict[str, Any]:
    rows = [
        record
        for record in candidate.records
        if record.get("metric_type") == "evaluation"
        and record_timestep(record) == TOTAL_TIMESTEPS
    ]
    if len(rows) != 1:
        raise ValueError(f"{candidate.run_name} does not have exactly one final evaluation")
    return rows[0]


def _evaluation_success_count(row: dict[str, Any]) -> int:
    explicit = row.get("evaluation_successes")
    if explicit is not None:
        return int(explicit)
    rate = as_float(row.get("success_rate"))
    episodes = int(row.get("eval_episodes", EVAL_EPISODES))
    if rate is None:
        raise ValueError("evaluation row lacks success_rate")
    return int(round(rate * episodes))


def _run_metrics(candidate: CandidateValidation) -> dict[str, Any]:
    final_eval = _final_evaluation(candidate)
    train_rows = sorted(
        [record for record in candidate.records if record.get("metric_type") == "train_episode"],
        key=lambda record: record_timestep(record) or -1,
    )
    training_rewards = [
        value
        for record in train_rows
        if (value := as_float(record.get("episode_reward"))) is not None
    ]
    last_100 = training_rewards[-100:]
    wall_clock = _max_value(candidate.records, "wall_clock_time")
    if wall_clock is None:
        wall_clock = _max_value(candidate.records, "elapsed_time")
    sps = _latest_value(candidate.records, "SPS")
    circuits = _max_value(candidate.records, "circuit_evaluations")
    eval_episodes = int(final_eval.get("eval_episodes", EVAL_EPISODES))
    successes = _evaluation_success_count(final_eval)
    return {
        "final_evaluation_reward": as_float(final_eval.get("episode_reward")),
        "final_evaluation_successes": successes,
        "final_evaluation_episodes": eval_episodes,
        "final_evaluation_success_rate": successes / eval_episodes,
        "training_last_100_mean": statistics.mean(last_100) if last_100 else None,
        "training_last_100_std": statistics.stdev(last_100) if len(last_100) >= 2 else None,
        "training_max_reward": max(training_rewards) if training_rewards else None,
        "wall_clock_seconds": wall_clock,
        "wall_clock_hours": wall_clock / 3600 if wall_clock is not None else None,
        "SPS": sps,
        "circuit_evaluations": circuits,
    }


def instantiate_and_count_models() -> list[dict[str, Any]]:
    """Instantiate the exact classes used for training and count named parameters."""
    cleanqrl_path = str(REPO_ROOT / "cleanqrl")
    if cleanqrl_path not in sys.path:
        sys.path.insert(0, cleanqrl_path)
    import gymnasium as gym
    import numpy as np
    import torch.nn as nn

    from cleanqrl.dqn_classical import DQNAgentClassical
    from cleanqrl.ppo_classical import PPOAgentClassical
    from cleanqrl.ppo_quantum_hybrid import Agent as HybridAgent
    from cleanqrl.ppo_quantum_hybrid import n_layers, n_qubits
    from ppo_classical_lunarlander_tinyparam import TinyClassicalAgent

    envs = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        ),
        single_action_space=gym.spaces.Discrete(4),
    )
    models = {
        "ppo": PPOAgentClassical(envs),
        "qrl": HybridAgent(envs),
        "ppo_tiny": TinyClassicalAgent(envs),
        "dqn": DQNAgentClassical(8, 4),
    }

    def count(module: nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

    def macs(module: nn.Module) -> int:
        return sum(
            layer.in_features * layer.out_features
            for layer in module.modules()
            if isinstance(layer, nn.Linear)
        )

    rows: list[dict[str, Any]] = []
    for friendly in AGENT_ORDER:
        model = models[friendly]
        row: dict[str, Any] = {
            "agent": AGENT_SPECS[friendly]["label"],
            "friendly_agent": friendly,
            "total_trainable_parameters": count(model),
            "classical_encoder_parameters": 0,
            "quantum_circuit_parameters": 0,
            "scaling_output_parameters": 0,
            "actor_policy_parameters": 0,
            "critic_value_parameters": 0,
            "q_network_parameters": 0,
            "policy_classical_macs": 0,
            "value_classical_macs": 0,
            "classical_flops_convention": "2 FLOPs per linear-layer weight (multiply + add); biases and activations excluded",
            "qubits": "",
            "variational_layers": "",
            "source_class": "",
        }
        if friendly == "ppo":
            row.update(
                actor_policy_parameters=count(model.actor),
                critic_value_parameters=count(model.critic),
                policy_classical_macs=macs(model.actor),
                value_classical_macs=macs(model.critic),
                source_class="cleanqrl.ppo_classical.PPOAgentClassical",
            )
        elif friendly == "ppo_tiny":
            row.update(
                actor_policy_parameters=count(model.actor),
                critic_value_parameters=count(model.critic),
                policy_classical_macs=macs(model.actor),
                value_classical_macs=macs(model.critic),
                source_class="ppo_classical_lunarlander_tinyparam.TinyClassicalAgent",
            )
        elif friendly == "dqn":
            row.update(
                q_network_parameters=count(model.network),
                policy_classical_macs=macs(model.network),
                source_class="cleanqrl.dqn_classical.DQNAgentClassical",
            )
        elif friendly == "qrl":
            row.update(
                classical_encoder_parameters=count(model.network),
                quantum_circuit_parameters=count(model.quantum_layer),
                scaling_output_parameters=int(model.actor_scale.numel()),
                actor_policy_parameters=(
                    count(model.network) + count(model.quantum_layer) + int(model.actor_scale.numel())
                ),
                critic_value_parameters=count(model.critic),
                policy_classical_macs=macs(model.network),
                value_classical_macs=macs(model.critic),
                qubits=int(n_qubits),
                variational_layers=int(n_layers),
                source_class="cleanqrl.ppo_quantum_hybrid.Agent",
            )
        row["policy_classical_flops"] = 2 * int(row["policy_classical_macs"])
        row["value_classical_flops"] = 2 * int(row["value_classical_macs"])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_all_metrics(cohort: CohortValidation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for friendly in AGENT_ORDER:
        for seed in SEEDS:
            candidate = cohort.selected.get((friendly, seed))
            if candidate is None:
                continue
            for record_index, record in enumerate(candidate.records, start=1):
                rows.append(
                    {
                        "run_name": candidate.run_name,
                        "run_path": str(candidate.path),
                        "agent": AGENT_SPECS[friendly]["label"],
                        "friendly_agent": friendly,
                        "agent_key": candidate.agent_key,
                        "seed": seed,
                        "configured_total_timesteps": candidate.config.get("total_timesteps", ""),
                        "configured_eval_interval": candidate.config.get("eval_interval", ""),
                        "configured_eval_episodes": candidate.config.get("eval_episodes", ""),
                        "configured_checkpoint_interval": candidate.config.get("checkpoint_interval", ""),
                        "configured_success_threshold": candidate.config.get("success_reward_threshold", ""),
                        "observation_preprocessing": candidate.config.get("observation_preprocessing", ""),
                        "action_remapping": candidate.config.get("action_remapping", "none"),
                        "evaluation_policy": candidate.config.get("evaluation_policy", "deterministic_greedy"),
                        "evaluation_exploration": candidate.config.get("evaluation_exploration", False),
                        "protocol_name": candidate.config.get("protocol_name", "legacy_standardized_implementation"),
                        "record_index": record_index,
                        "metric_type": record.get("metric_type", ""),
                        "global_step": record.get("global_step", ""),
                        "training_timestep": record.get("training_timestep", ""),
                        "episode_reward": record.get("episode_reward", ""),
                        "episode_length": record.get("episode_length", ""),
                        "success_rate": record.get("success_rate", ""),
                        "eval_episodes": record.get("eval_episodes", ""),
                        "evaluation_successes": record.get("evaluation_successes", ""),
                        "wall_clock_time": record.get("wall_clock_time", record.get("elapsed_time", "")),
                        "SPS": record.get("SPS", ""),
                        "circuit_evaluations": record.get("circuit_evaluations", ""),
                        "evaluation_episode_rewards_json": json.dumps(
                            record.get("evaluation_episode_rewards", "")
                        ),
                        "evaluation_episode_lengths_json": json.dumps(
                            record.get("evaluation_episode_lengths", "")
                        ),
                    }
                )
    return rows


def build_seed_report(
    cohort: CohortValidation, parameter_by_agent: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metrics: dict[tuple[str, int], dict[str, Any]] = {}
    for friendly in AGENT_ORDER:
        for seed in SEEDS:
            candidate = cohort.selected.get((friendly, seed))
            base = {
                "agent": AGENT_SPECS[friendly]["label"],
                "friendly_agent": friendly,
                "agent_key": AGENT_SPECS[friendly]["agent_key"],
                "seed": seed,
                "status": "PASS" if candidate else "MISSING",
                "run_name": candidate.run_name if candidate else "",
                "run_path": str(candidate.path) if candidate else "",
                "total_timesteps": TOTAL_TIMESTEPS if candidate else "",
                "trainable_parameters": parameter_by_agent[friendly]["total_trainable_parameters"],
            }
            if candidate:
                run_metrics = _run_metrics(candidate)
                metrics[(friendly, seed)] = run_metrics
                base.update(run_metrics)
            rows.append(base)
    return rows, metrics


def build_checkpoint_report(
    cohort: CohortValidation,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for friendly in AGENT_ORDER:
        by_seed: dict[int, dict[int, dict[str, Any]]] = {}
        for seed in SEEDS:
            candidate = cohort.selected.get((friendly, seed))
            by_seed[seed] = {}
            if candidate:
                for record in candidate.records:
                    if record.get("metric_type") == "evaluation":
                        step = record_timestep(record)
                        if step is not None:
                            by_seed[seed][step] = record
        for step in EXPECTED_STEPS:
            rewards = [
                value
                for seed in SEEDS
                if (row := by_seed[seed].get(step)) is not None
                and (value := as_float(row.get("episode_reward"))) is not None
            ]
            successes = sum(
                _evaluation_success_count(row)
                for seed in SEEDS
                if (row := by_seed[seed].get(step)) is not None
            )
            episodes = sum(
                int(row.get("eval_episodes", EVAL_EPISODES))
                for seed in SEEDS
                if (row := by_seed[seed].get(step)) is not None
            )
            ci_low, ci_high = student_t_ci(rewards)
            wilson_low, wilson_high = wilson_interval(successes, episodes)
            output = {
                "agent": AGENT_SPECS[friendly]["label"],
                "friendly_agent": friendly,
                "training_timestep": step,
                "seed_0_evaluation_reward": "",
                "seed_1_evaluation_reward": "",
                "seed_2_evaluation_reward": "",
                "seed_count": len(rewards),
                "mean_evaluation_reward": statistics.mean(rewards) if rewards else "",
                "sample_std_evaluation_reward": (
                    value if (value := sample_std(rewards)) is not None else ""
                ),
                "t95_ci_lower": "" if ci_low is None else ci_low,
                "t95_ci_upper": "" if ci_high is None else ci_high,
                "evaluation_successes": successes,
                "evaluation_episodes": episodes,
                "pooled_success_rate": successes / episodes if episodes else "",
                "wilson95_lower": "" if wilson_low is None else wilson_low,
                "wilson95_upper": "" if wilson_high is None else wilson_high,
            }
            for seed in SEEDS:
                record = by_seed[seed].get(step)
                if record:
                    output[f"seed_{seed}_evaluation_reward"] = record.get("episode_reward", "")
            rows.append(output)
    return rows


def build_model_summary(
    metrics: dict[tuple[str, int], dict[str, Any]],
    parameter_by_agent: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for friendly in AGENT_ORDER:
        values = [
            metrics[(friendly, seed)]["final_evaluation_reward"]
            for seed in SEEDS
            if (friendly, seed) in metrics
        ]
        success_count = sum(
            metrics[(friendly, seed)]["final_evaluation_successes"]
            for seed in SEEDS
            if (friendly, seed) in metrics
        )
        episode_count = sum(
            metrics[(friendly, seed)]["final_evaluation_episodes"]
            for seed in SEEDS
            if (friendly, seed) in metrics
        )
        ci_low, ci_high = student_t_ci(values)
        wilson_low, wilson_high = wilson_interval(success_count, episode_count)
        row = {
            "agent": AGENT_SPECS[friendly]["label"],
            "friendly_agent": friendly,
            "seed_0_final_evaluation_reward": metrics.get((friendly, 0), {}).get("final_evaluation_reward", ""),
            "seed_1_final_evaluation_reward": metrics.get((friendly, 1), {}).get("final_evaluation_reward", ""),
            "seed_2_final_evaluation_reward": metrics.get((friendly, 2), {}).get("final_evaluation_reward", ""),
            "seed_count": len(values),
            "mean_final_evaluation_reward": statistics.mean(values) if values else "",
            "sample_std_final_evaluation_reward": (
                value if (value := sample_std(values)) is not None else ""
            ),
            "t95_ci_lower": "" if ci_low is None else ci_low,
            "t95_ci_upper": "" if ci_high is None else ci_high,
            "final_evaluation_successes": success_count,
            "final_evaluation_episodes": episode_count,
            "final_evaluation_success_percentage": 100 * success_count / episode_count if episode_count else "",
            "wilson95_success_lower_percentage": "" if wilson_low is None else 100 * wilson_low,
            "wilson95_success_upper_percentage": "" if wilson_high is None else 100 * wilson_high,
            "trainable_parameters": parameter_by_agent[friendly]["total_trainable_parameters"],
            "status": "COMPLETE" if len(values) == 3 else f"INCOMPLETE ({len(values)}/3 seeds)",
        }
        rows.append(row)
    return rows


def build_compute_summary(
    metrics: dict[tuple[str, int], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for friendly in AGENT_ORDER:
        for seed in SEEDS:
            values = metrics.get((friendly, seed))
            rows.append(
                {
                    "row_type": "seed",
                    "agent": AGENT_SPECS[friendly]["label"],
                    "friendly_agent": friendly,
                    "seed": seed,
                    "wall_clock_seconds": values.get("wall_clock_seconds", "") if values else "",
                    "wall_clock_hours": values.get("wall_clock_hours", "") if values else "",
                    "SPS": values.get("SPS", "") if values else "",
                    "circuit_evaluations": values.get("circuit_evaluations", "") if values else "",
                    "seed_count": 1 if values else 0,
                }
            )
        available = [metrics[(friendly, seed)] for seed in SEEDS if (friendly, seed) in metrics]
        walls = [row["wall_clock_seconds"] for row in available if row["wall_clock_seconds"] is not None]
        sps_values = [row["SPS"] for row in available if row["SPS"] is not None]
        circuits = [row["circuit_evaluations"] for row in available if row["circuit_evaluations"] is not None]
        rows.append(
            {
                "row_type": "aggregate",
                "agent": AGENT_SPECS[friendly]["label"],
                "friendly_agent": friendly,
                "seed": "",
                "seed_count": len(available),
                "wall_clock_mean_seconds": statistics.mean(walls) if walls else "",
                "wall_clock_sample_std_seconds": (
                    value if (value := sample_std(walls)) is not None else ""
                ),
                "wall_clock_cumulative_seconds": sum(walls) if walls else "",
                "wall_clock_cumulative_hours": sum(walls) / 3600 if walls else "",
                "SPS_mean": statistics.mean(sps_values) if sps_values else "",
                "SPS_sample_std": (
                    value if (value := sample_std(sps_values)) is not None else ""
                ),
                "circuit_evaluations_mean": statistics.mean(circuits) if circuits else "",
                "circuit_evaluations_cumulative": sum(circuits) if circuits else "",
            }
        )
    return rows


def _fmt(value: Any, digits: int = 2) -> str:
    numeric = as_float(value)
    return "--" if numeric is None else f"{numeric:.{digits}f}"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_tables(
    model_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    parameter_by_agent: dict[str, dict[str, Any]],
) -> None:
    compute_aggregate = {
        row["friendly_agent"]: row for row in compute_rows if row["row_type"] == "aggregate"
    }
    summary_csv_rows = []
    latex_rows = []
    for row in model_rows:
        friendly = row["friendly_agent"]
        compute = compute_aggregate[friendly]
        complete = row["seed_count"] == 3
        reward = _fmt(row["mean_final_evaluation_reward"]) if complete else "pending"
        ci = (
            f"[{_fmt(row['t95_ci_lower'])}, {_fmt(row['t95_ci_upper'])}]"
            if complete
            else "pending"
        )
        success = (
            f"{int(row['final_evaluation_successes'])}/{int(row['final_evaluation_episodes'])} "
            f"({_fmt(row['final_evaluation_success_percentage'], 1)}%)"
            if complete
            else "pending"
        )
        wall = _fmt(as_float(compute.get("wall_clock_mean_seconds")) / 3600 if as_float(compute.get("wall_clock_mean_seconds")) is not None else None)
        sps = _fmt(compute.get("SPS_mean"), 1)
        output = {
            "Agent": row["agent"],
            "Final eval reward": reward,
            "95% t CI": ci,
            "Final eval success": success,
            "Parameters": row["trainable_parameters"],
            "Mean wall-clock hours": wall,
            "Mean SPS": sps,
        }
        summary_csv_rows.append(output)
        latex_rows.append(
            " & ".join(_latex_escape(str(output[key])) for key in output) + r" \\"
        )
    _write_csv(Path("table_lunarlander_900k_summary.csv"), summary_csv_rows)
    latex = "\n".join(
        [
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Agent & Final reward & 95\% CI & Success & Parameters & Time (h) & SPS \\",
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "% Reward uncertainty is a 95% Student-t CI across three seeds (df=2).",
        ]
    )
    Path("table_lunarlander_900k_summary.tex").write_text(latex + "\n", encoding="utf-8")

    qrl_seed_rows = [
        row for row in compute_rows if row["friendly_agent"] == "qrl" and row["row_type"] == "seed"
    ]
    qrl_aggregate = compute_aggregate["qrl"]
    qrl_params = parameter_by_agent["qrl"]
    quantum_csv = [
        {
            "QRL qubits": qrl_params["qubits"],
            "Variational layers": qrl_params["variational_layers"],
            "Quantum parameters": qrl_params["quantum_circuit_parameters"],
            "Seed 0 circuit evaluations": qrl_seed_rows[0].get("circuit_evaluations", ""),
            "Seed 1 circuit evaluations": qrl_seed_rows[1].get("circuit_evaluations", ""),
            "Seed 2 circuit evaluations": qrl_seed_rows[2].get("circuit_evaluations", ""),
            "Mean circuit evaluations": qrl_aggregate.get("circuit_evaluations_mean", ""),
            "Cumulative circuit evaluations": qrl_aggregate.get("circuit_evaluations_cumulative", ""),
            "Mean wall-clock hours": (
                as_float(qrl_aggregate.get("wall_clock_mean_seconds")) / 3600
                if as_float(qrl_aggregate.get("wall_clock_mean_seconds")) is not None
                else ""
            ),
            "Cumulative wall-clock hours": qrl_aggregate.get("wall_clock_cumulative_hours", ""),
        }
    ]
    _write_csv(Path("table_lunarlander_900k_quantum_compute.csv"), quantum_csv)
    q = quantum_csv[0]
    quantum_latex = "\n".join(
        [
            r"\begin{tabular}{lr}",
            r"\toprule",
            r"QRL resource & Value \\",
            r"\midrule",
            f"Qubits & {_latex_escape(str(q['QRL qubits']))} \\\\",
            f"Variational layers & {_latex_escape(str(q['Variational layers']))} \\\\",
            f"Trainable quantum parameters & {_latex_escape(str(q['Quantum parameters']))} \\\\",
            f"Circuit evaluations / seed & {_latex_escape(', '.join(_fmt(q[f'Seed {seed} circuit evaluations'], 0) for seed in SEEDS))} \\\\",
            f"Cumulative circuit evaluations & {_fmt(q['Cumulative circuit evaluations'], 0)} \\\\",
            f"Mean wall-clock (h) & {_fmt(q['Mean wall-clock hours'])} \\\\",
            f"Cumulative wall-clock (h) & {_fmt(q['Cumulative wall-clock hours'])} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    Path("table_lunarlander_900k_quantum_compute.tex").write_text(
        quantum_latex + "\n", encoding="utf-8"
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_readme(cohort: CohortValidation, asset_dir: Path, xlsx_created: bool) -> None:
    included = [
        f"- `{candidate.path}` ({display_combo(combo)})"
        for combo, candidate in sorted(
            cohort.selected.items(), key=lambda item: (AGENT_ORDER.index(item[0][0]), item[0][1])
        )
    ]
    excluded = []
    for candidate in cohort.candidates:
        if candidate.selected:
            continue
        reason = "; ".join(candidate.warnings) if candidate.valid else candidate.reason
        excluded.append(f"- `{candidate.path}` — {reason}")
    missing = [f"- {display_combo(combo)}" for combo in cohort.missing]
    content = f"""# Matched LunarLander-v3 900k dataset

Status: **{'COMPLETE' if cohort.passed else 'INCOMPLETE'}** ({len(cohort.selected)}/12 validated runs).

Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}

Source Git commit: `{_git_sha()}`

## Exact protocol

- Environment: `LunarLander-v3` with native discrete actions and no action remapping.
- Agents: PPO, QRL (the existing hybrid Config C architecture), PPO-tiny, and DQN.
- Seeds: 0, 1, and 2 for every agent.
- Configured/completed budget: exactly 900,000 environment interactions per run.
- Observation preprocessing: none.
- Evaluation: 10 deterministic greedy episodes, exploration disabled, at every 100,000 interactions from 100k through 900k.
- Checkpoints: every 100,000 interactions from 100k through 900k.
- Success: episode reward >= {SUCCESS_THRESHOLD:g}.
- The 95% reward intervals use Student-t with `df=2`, `t=4.3026527`; success intervals use Wilson 95% binomial intervals.
- Historical 1M/2M runs and runs merely containing a 900k checkpoint are excluded.

## Included run paths

{chr(10).join(included) if included else 'None.'}

## Missing required runs

{chr(10).join(missing) if missing else 'None.'}

## Excluded candidate paths

{chr(10).join(excluded) if excluded else 'None.'}

## Files

- `matched_900k_all_metrics.csv`: normalized selected-run metrics plus explicit protocol/configuration columns.
- `matched_900k_seed_report.csv`: one row per required agent/seed, including explicit missing rows.
- `matched_900k_model_summary.csv`: final reward, Student-t CI, pooled evaluation success, and Wilson interval.
- `matched_900k_parameter_summary.csv`: programmatically instantiated parameter decomposition and classical MAC/FLOP estimates.
- `matched_900k_compute_summary.csv`: per-seed and aggregate wall-clock, SPS, and QRL circuit workload.
- `matched_900k_evaluation_checkpoints.csv`: raw seed values and checkpoint-level statistics.
- `matched_900k_data.xlsx`: {'created' if xlsx_created else 'not created because the workbook artifact runtime is unavailable to this repository script and no project XLSX dependency is declared'}.

Classical FLOPs use exactly two FLOPs per linear-layer weight (one multiply plus one add); biases and activation costs are excluded. Quantum resources are reported independently and are never converted to FLOPs.

## Regeneration commands (PowerShell)

```powershell
& 'C:\\Users\\Aadesh\\anaconda3\\envs\\cleanqrl\\python.exe' validate_lunarlander_900k.py
& 'C:\\Users\\Aadesh\\anaconda3\\envs\\cleanqrl\\python.exe' aggregate_lunarlander_900k.py
& 'C:\\Users\\Aadesh\\anaconda3\\envs\\cleanqrl\\python.exe' plot_lunarlander_900k_paper_figures.py
```

While the cohort is incomplete, `aggregate_lunarlander_900k.py --allow-incomplete` may be used to regenerate this clearly marked provisional inventory. Plot generation intentionally refuses incomplete data.
"""
    (asset_dir / "README.md").write_text(content, encoding="utf-8")


def write_report_ready_section(
    cohort: CohortValidation,
    model_rows: list[dict[str, Any]],
    compute_rows: list[dict[str, Any]],
    parameter_by_agent: dict[str, dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    run_metrics: dict[tuple[str, int], dict[str, Any]],
) -> None:
    path = Path("lunarlander_900k_report_ready_section.md")
    if not cohort.passed:
        content = "\n".join(
            [
                "# Matched LunarLander-v3 900k Results",
                "",
                "**Results text withheld: the primary cohort is incomplete.**",
                "",
                f"The validator currently selects {len(cohort.selected)}/12 required runs. "
                "Paper-ready conclusions must not be written until all twelve dedicated 900k runs pass.",
                "",
                "Missing runs:",
                "",
                *[f"- {display_combo(combo)}" for combo in cohort.missing],
                "",
                "Run `python validate_lunarlander_900k.py` after training. This file will then be regenerated from validated values.",
                "",
            ]
        )
        path.write_text(content, encoding="utf-8")
        return

    model_by_agent = {row["friendly_agent"]: row for row in model_rows}
    compute_by_agent = {
        row["friendly_agent"]: row for row in compute_rows if row["row_type"] == "aggregate"
    }
    result_lines = []
    for friendly in AGENT_ORDER:
        row = model_by_agent[friendly]
        result_lines.append(
            f"- {row['agent']}: seed rewards "
            f"{_fmt(row['seed_0_final_evaluation_reward'])}, "
            f"{_fmt(row['seed_1_final_evaluation_reward'])}, and "
            f"{_fmt(row['seed_2_final_evaluation_reward'])}; mean "
            f"{_fmt(row['mean_final_evaluation_reward'])} (95% Student-t CI "
            f"[{_fmt(row['t95_ci_lower'])}, {_fmt(row['t95_ci_upper'])}], "
            f"sample SD {_fmt(row['sample_std_final_evaluation_reward'])}); "
            f"{row['final_evaluation_successes']}/{row['final_evaluation_episodes']} successful evaluations "
            f"({_fmt(row['final_evaluation_success_percentage'], 1)}%)."
        )
    checkpoint_by_agent = {
        friendly: [row for row in checkpoint_rows if row["friendly_agent"] == friendly]
        for friendly in AGENT_ORDER
    }
    learning_lines = []
    for friendly in AGENT_ORDER:
        rows = sorted(checkpoint_by_agent[friendly], key=lambda row: row["training_timestep"])
        mean_solved = next(
            (row["training_timestep"] for row in rows if row["mean_evaluation_reward"] >= SUCCESS_THRESHOLD),
            None,
        )
        first_success = next(
            (row["training_timestep"] for row in rows if row["evaluation_successes"] > 0),
            None,
        )
        learning_lines.append(
            f"- {AGENT_SPECS[friendly]['label']}: first nonzero deterministic-evaluation success at "
            f"{f'{first_success // 1000}k' if first_success is not None else 'no checkpoint'}; "
            f"three-seed mean first reached 200 at "
            f"{f'{mean_solved // 1000}k' if mean_solved is not None else 'no checkpoint'}."
        )
    ranking = sorted(
        model_rows, key=lambda row: float(row["mean_final_evaluation_reward"]), reverse=True
    )
    ranking_text = " > ".join(
        f"{row['agent']} ({_fmt(row['mean_final_evaluation_reward'])})" for row in ranking
    )
    qrl_mean = float(model_by_agent["qrl"]["mean_final_evaluation_reward"])
    qrl_comparisons = []
    for other in ("ppo", "ppo_tiny", "dqn"):
        other_mean = float(model_by_agent[other]["mean_final_evaluation_reward"])
        delta = qrl_mean - other_mean
        relation = "higher" if delta > 0 else ("lower" if delta < 0 else "equal")
        qrl_comparisons.append(
            f"{abs(delta):.2f} points {relation} than {AGENT_SPECS[other]['label']}"
        )
    overlapping_pairs = []
    for index, left in enumerate(model_rows):
        for right in model_rows[index + 1 :]:
            if not (
                float(left["t95_ci_upper"]) < float(right["t95_ci_lower"])
                or float(right["t95_ci_upper"]) < float(left["t95_ci_lower"])
            ):
                overlapping_pairs.append(f"{left['agent']}/{right['agent']}")
    training_lines = []
    for friendly in AGENT_ORDER:
        values = [run_metrics[(friendly, seed)]["training_last_100_mean"] for seed in SEEDS]
        training_lines.append(
            f"- {AGENT_SPECS[friendly]['label']}: seed-level last-100 training means "
            + ", ".join(_fmt(value) for value in values)
            + "."
        )
    compute_lines = []
    for friendly in AGENT_ORDER:
        row = compute_by_agent[friendly]
        compute_lines.append(
            f"- {AGENT_SPECS[friendly]['label']}: mean wall-clock "
            f"{_fmt(as_float(row['wall_clock_mean_seconds']) / 3600)} h per seed, "
            f"{_fmt(row['wall_clock_cumulative_hours'])} h cumulative, and "
            f"{_fmt(row['SPS_mean'], 1)} mean SPS."
        )
    qrl_compute = compute_by_agent["qrl"]
    qrl_params = parameter_by_agent["qrl"]
    content = f"""# Matched LunarLander-v3 900k Results

## Protocol

PPO, QRL, PPO-tiny, and DQN were trained from the outset for exactly 900,000 LunarLander-v3 interactions using seeds 0, 1, and 2. Every run used unmodified observations, the native discrete action space, deterministic greedy evaluation over 10 episodes every 100,000 interactions, and a reward-at-least-200 success definition. Historical runs configured for other horizons were excluded.

## Final deterministic evaluation

{chr(10).join(result_lines)}

Intervals are 95% Student-t intervals across only three seeds (`df=2`); raw seed values are therefore shown explicitly. Overlapping intervals are not interpreted as evidence of a statistically significant difference.

The descriptive final-mean ordering was {ranking_text}. This ordering is descriptive, not a significance claim. {'Overlapping 95% intervals occurred for: ' + ', '.join(overlapping_pairs) + '.' if overlapping_pairs else 'None of the plotted 95% intervals overlapped, but n=3 is still too small for strong significance language without a prespecified test.'}

Relative to the classical three-seed means, the QRL mean was {', '.join(qrl_comparisons)}. These differences are reported without best-seed selection.

## Evaluation learning behavior

{chr(10).join(learning_lines)}

These statements use only the nine unsmoothed deterministic-evaluation checkpoints. They do not interpolate when a threshold was crossed.

## Training diagnostics (reported separately)

{chr(10).join(training_lines)}

Training returns are noisier on-policy diagnostics and are not substituted for deterministic evaluation results.

## Parameters and compute

The exact instantiated trainable parameter counts are {', '.join(f"{AGENT_SPECS[name]['label']}={parameter_by_agent[name]['total_trainable_parameters']:,}" for name in AGENT_ORDER)}. QRL contains {qrl_params['classical_encoder_parameters']:,} classical encoder, {qrl_params['quantum_circuit_parameters']:,} variational quantum, {qrl_params['scaling_output_parameters']:,} scaling/output, and {qrl_params['critic_value_parameters']:,} critic parameters. Parameter count alone does not establish an advantage over PPO-tiny.

{chr(10).join(compute_lines)}

QRL used a {qrl_params['qubits']}-qubit, {qrl_params['variational_layers']}-layer variational circuit. Its mean wall-clock cost was {_fmt(as_float(qrl_compute['wall_clock_mean_seconds']) / 3600 if as_float(qrl_compute['wall_clock_mean_seconds']) is not None else None)} hours per seed ({_fmt(qrl_compute['wall_clock_cumulative_hours'])} cumulative hours), with {_fmt(qrl_compute['circuit_evaluations_cumulative'], 0)} cumulative simulated circuit evaluations. Classical MAC/FLOP estimates and quantum resources are reported separately; no FLOP-to-qubit conversion is made.

## Limitations

Only three training seeds were measured, producing wide and unstable uncertainty estimates. Simulator wall-clock results are hardware- and implementation-dependent, and circuit-evaluation counts do not represent hardware FLOPs. Conclusions should emphasize the full three-seed cohort rather than a selected best QRL seed.
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Generate a clearly marked provisional dataset while runs are missing.",
    )
    args = parser.parse_args()
    cohort = validate_cohort(args.logs_dir)
    if not cohort.passed and not args.allow_incomplete:
        print(
            "Aggregation refused: validator has not passed. Missing: "
            + ", ".join(display_combo(combo) for combo in cohort.missing),
            file=sys.stderr,
        )
        return 1

    asset_dir = args.output_dir
    asset_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows = instantiate_and_count_models()
    parameter_by_agent = {row["friendly_agent"]: row for row in parameter_rows}
    all_metrics = build_all_metrics(cohort)
    seed_rows, metrics = build_seed_report(cohort, parameter_by_agent)
    checkpoint_rows = build_checkpoint_report(cohort)
    model_rows = build_model_summary(metrics, parameter_by_agent)
    compute_rows = build_compute_summary(metrics)

    outputs = {
        "matched_900k_all_metrics.csv": all_metrics,
        "matched_900k_seed_report.csv": seed_rows,
        "matched_900k_model_summary.csv": model_rows,
        "matched_900k_parameter_summary.csv": parameter_rows,
        "matched_900k_compute_summary.csv": compute_rows,
        "matched_900k_evaluation_checkpoints.csv": checkpoint_rows,
    }
    for filename, rows in outputs.items():
        _write_csv(asset_dir / filename, rows)
    # XLSX is optional in the experiment specification. The repository does not
    # declare an XLSX dependency, and workbook authoring belongs to the external
    # artifact runtime rather than this reproducibility script.
    xlsx_created = False
    write_readme(cohort, asset_dir, xlsx_created)
    write_tables(model_rows, compute_rows, parameter_by_agent)
    write_report_ready_section(
        cohort,
        model_rows,
        compute_rows,
        parameter_by_agent,
        checkpoint_rows,
        metrics,
    )
    print(f"Wrote matched 900k assets to {asset_dir}")
    print(f"Cohort status: {'PASS' if cohort.passed else 'INCOMPLETE'} ({len(cohort.selected)}/12)")
    if not xlsx_created:
        print("Skipped optional XLSX: no repository-supported workbook runtime is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
