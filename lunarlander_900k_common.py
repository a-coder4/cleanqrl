"""Shared protocol and validation helpers for the matched 900k experiment."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"
TOTAL_TIMESTEPS = 900_000
SEEDS = (0, 1, 2)
EVAL_INTERVAL = 100_000
CHECKPOINT_INTERVAL = 100_000
EVAL_EPISODES = 10
SUCCESS_THRESHOLD = 200.0
EXPECTED_STEPS = tuple(range(EVAL_INTERVAL, TOTAL_TIMESTEPS + 1, EVAL_INTERVAL))

AGENT_SPECS: dict[str, dict[str, str]] = {
    "ppo": {
        "agent_key": "PPO_classical",
        "label": "PPO",
        "config": "configs/benchmarks/ppo_classical_lunarlander.yaml",
    },
    "qrl": {
        "agent_key": "ppo_quantum_hybrid",
        "label": "QRL",
        "config": "configs/benchmarks/ppo_quantum_lunarlander.yaml",
    },
    "ppo_tiny": {
        "agent_key": "PPO_tiny_classical",
        "label": "PPO-tiny",
        "config": "ppo_classical_lunarlander_tinyparam.py",
    },
    "dqn": {
        "agent_key": "DQN_classical",
        "label": "DQN",
        "config": "configs/benchmarks/dqn_classical_lunarlander.yaml",
    },
}
AGENT_ORDER = ("ppo", "qrl", "ppo_tiny", "dqn")
AGENT_KEY_TO_FRIENDLY = {
    spec["agent_key"]: friendly for friendly, spec in AGENT_SPECS.items()
}

STANDARD_ROW_KEYS = {
    "metric_type",
    "agent",
    "seed",
    "env_id",
    "global_step",
    "training_timestep",
    "episode_reward",
    "episode_length",
    "success_rate",
}


@dataclass
class CandidateValidation:
    run_name: str
    path: Path
    agent_key: str = ""
    friendly_agent: str = ""
    seed: int | None = None
    configured_timesteps: int | None = None
    valid: bool = False
    selected: bool = False
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    max_timestep: int | None = None
    evaluation_steps: list[int] = field(default_factory=list)
    checkpoint_steps: list[int] = field(default_factory=list)
    train_rows: int = 0
    evaluation_rows: int = 0
    diagnostic_rows: int = 0

    @property
    def combo(self) -> tuple[str, int] | None:
        if self.friendly_agent and self.seed is not None:
            return self.friendly_agent, self.seed
        return None

    @property
    def reason(self) -> str:
        return "; ".join(self.failures) if self.failures else "valid"


@dataclass
class CohortValidation:
    candidates: list[CandidateValidation]
    selected: dict[tuple[str, int], CandidateValidation]
    ambiguities: dict[tuple[str, int], list[CandidateValidation]]
    missing: list[tuple[str, int]]

    @property
    def passed(self) -> bool:
        return not self.missing and len(self.selected) == len(AGENT_ORDER) * len(SEEDS)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.exists():
        return records, errors
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"result.json line {line_number} is invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"result.json line {line_number} is not a JSON object")
                continue
            records.append(record)
    return records, errors


def as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def record_timestep(record: dict[str, Any]) -> int | None:
    return as_int(record.get("training_timestep", record.get("global_step")))


def checkpoint_files(run_path: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    pattern = re.compile(r"_step(\d+)\.cleanqrl_model$")
    for path in sorted(run_path.glob("*.cleanqrl_model")):
        match = pattern.search(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    return checkpoints


def _add_config_failure(
    failures: list[str], config: dict[str, Any], key: str, expected: Any
) -> None:
    actual = config.get(key)
    if actual != expected:
        failures.append(f"config {key}={actual!r}, expected {expected!r}")


def _validate_evaluation_row(
    row: dict[str, Any], expected_agent: str, expected_seed: int, failures: list[str]
) -> None:
    step = record_timestep(row)
    missing = sorted((STANDARD_ROW_KEYS | {"eval_episodes"}) - set(row))
    if missing:
        failures.append(f"evaluation row at {step} missing keys {missing}")
    if row.get("agent") != expected_agent:
        failures.append(f"evaluation row at {step} has agent={row.get('agent')!r}")
    if as_int(row.get("seed")) != expected_seed:
        failures.append(f"evaluation row at {step} has seed={row.get('seed')!r}")
    if row.get("env_id") != "LunarLander-v3":
        failures.append(f"evaluation row at {step} has env_id={row.get('env_id')!r}")
    if as_int(row.get("global_step")) != step or as_int(row.get("training_timestep")) != step:
        failures.append(f"evaluation row at {step} has inconsistent timestep fields")
    if as_int(row.get("eval_episodes")) != EVAL_EPISODES:
        failures.append(
            f"evaluation row at {step} has eval_episodes={row.get('eval_episodes')!r}"
        )
    reward = as_float(row.get("episode_reward"))
    length = as_float(row.get("episode_length"))
    success_rate = as_float(row.get("success_rate"))
    if reward is None:
        failures.append(f"evaluation row at {step} lacks a finite mean reward")
    if length is None or length <= 0:
        failures.append(f"evaluation row at {step} lacks a positive mean episode length")
    if success_rate is None or not 0 <= success_rate <= 1:
        failures.append(f"evaluation row at {step} has invalid success_rate")
    elif not math.isclose(success_rate * EVAL_EPISODES, round(success_rate * EVAL_EPISODES), abs_tol=1e-7):
        failures.append(
            f"evaluation row at {step} success_rate is incompatible with 10 episodes"
        )
    explicit_successes = as_int(row.get("evaluation_successes"))
    if row.get("evaluation_successes") is not None:
        if explicit_successes is None or not 0 <= explicit_successes <= EVAL_EPISODES:
            failures.append(f"evaluation row at {step} has invalid evaluation_successes")
        elif success_rate is not None and explicit_successes != round(success_rate * EVAL_EPISODES):
            failures.append(
                f"evaluation row at {step} success count disagrees with success_rate"
            )

    raw_rewards = row.get("evaluation_episode_rewards")
    raw_lengths = row.get("evaluation_episode_lengths")
    if raw_rewards is not None:
        if not isinstance(raw_rewards, list) or len(raw_rewards) != EVAL_EPISODES:
            failures.append(f"evaluation row at {step} has invalid episode reward array")
        else:
            parsed_rewards = [as_float(value) for value in raw_rewards]
            if any(value is None for value in parsed_rewards):
                failures.append(f"evaluation row at {step} has non-finite episode rewards")
            elif reward is not None and not math.isclose(
                reward,
                sum(value for value in parsed_rewards if value is not None) / EVAL_EPISODES,
                rel_tol=1e-9,
                abs_tol=1e-7,
            ):
                failures.append(f"evaluation row at {step} mean disagrees with raw rewards")
            elif success_rate is not None:
                raw_successes = sum(
                    value is not None and value >= SUCCESS_THRESHOLD for value in parsed_rewards
                )
                if raw_successes != round(success_rate * EVAL_EPISODES):
                    failures.append(
                        f"evaluation row at {step} success rate disagrees with raw rewards"
                    )
    if raw_lengths is not None and (
        not isinstance(raw_lengths, list) or len(raw_lengths) != EVAL_EPISODES
    ):
        failures.append(f"evaluation row at {step} has invalid episode length array")


def validate_candidate(run_path: Path) -> CandidateValidation:
    run_path = run_path.resolve()
    validation = CandidateValidation(run_name=run_path.name, path=run_path)
    config_path = run_path / "config.yaml"
    result_path = run_path / "result.json"
    if not config_path.exists():
        validation.failures.append("config.yaml is missing")
        return validation
    try:
        config = load_yaml(config_path)
    except (OSError, yaml.YAMLError) as exc:
        validation.failures.append(f"config.yaml could not be read: {exc}")
        return validation

    validation.config = config
    validation.agent_key = str(config.get("agent", ""))
    validation.friendly_agent = AGENT_KEY_TO_FRIENDLY.get(validation.agent_key, "")
    validation.seed = as_int(config.get("seed"))
    validation.configured_timesteps = as_int(config.get("total_timesteps"))

    if config.get("env_id") != "LunarLander-v3":
        validation.failures.append(
            f"config env_id={config.get('env_id')!r}, expected 'LunarLander-v3'"
        )
        return validation
    if not validation.friendly_agent:
        validation.failures.append(f"agent {validation.agent_key!r} is outside the primary cohort")
    if validation.seed not in SEEDS:
        validation.failures.append(f"seed {validation.seed!r} is outside {list(SEEDS)}")
    if validation.configured_timesteps != TOTAL_TIMESTEPS:
        validation.failures.append(
            f"configured total_timesteps={validation.configured_timesteps!r}, expected {TOTAL_TIMESTEPS}"
        )

    # Avoid loading large historical outputs once their configuration already
    # proves that they cannot enter this dedicated cohort.
    if validation.failures:
        if not result_path.exists():
            validation.failures.append("result.json is missing")
        return validation

    _add_config_failure(validation.failures, config, "standardize_lunarlander", True)
    _add_config_failure(validation.failures, config, "observation_preprocessing", "none")
    _add_config_failure(validation.failures, config, "eval_interval", EVAL_INTERVAL)
    _add_config_failure(validation.failures, config, "checkpoint_interval", CHECKPOINT_INTERVAL)
    _add_config_failure(validation.failures, config, "eval_episodes", EVAL_EPISODES)
    _add_config_failure(validation.failures, config, "success_reward_threshold", SUCCESS_THRESHOLD)
    training_budget = config.get("training_budget_timesteps")
    if training_budget is not None and as_int(training_budget) != TOTAL_TIMESTEPS:
        validation.failures.append(
            f"config training_budget_timesteps={training_budget!r}, expected {TOTAL_TIMESTEPS}"
        )
    action_remapping = config.get("action_remapping", "none")
    if action_remapping not in (None, "none", "native"):
        validation.failures.append(f"action_remapping={action_remapping!r}, expected native/none")
    eval_exploration = config.get("evaluation_exploration", False)
    if eval_exploration not in (False, 0, "false", "False", None):
        validation.failures.append("evaluation exploration is not disabled")
    evaluation_policy = config.get("evaluation_policy", "deterministic_greedy")
    if evaluation_policy != "deterministic_greedy":
        validation.failures.append(
            f"evaluation_policy={evaluation_policy!r}, expected 'deterministic_greedy'"
        )

    if not result_path.exists():
        validation.failures.append("result.json is missing")
        return validation
    records, parse_errors = load_jsonl(result_path)
    validation.records = records
    validation.failures.extend(parse_errors)
    if not records:
        validation.failures.append("result.json has no metric records")
        return validation

    timesteps = [step for record in records if (step := record_timestep(record)) is not None]
    validation.max_timestep = max(timesteps) if timesteps else None
    if validation.max_timestep != TOTAL_TIMESTEPS:
        validation.failures.append(
            f"max logged training timestep={validation.max_timestep!r}, expected {TOTAL_TIMESTEPS}"
        )

    train_rows = [record for record in records if record.get("metric_type") == "train_episode"]
    eval_rows = [record for record in records if record.get("metric_type") == "evaluation"]
    diagnostic_rows = [
        record for record in records if record.get("metric_type") == "training_diagnostic"
    ]
    validation.train_rows = len(train_rows)
    validation.evaluation_rows = len(eval_rows)
    validation.diagnostic_rows = len(diagnostic_rows)
    if not train_rows:
        validation.failures.append("no train_episode rows found")
    if not diagnostic_rows:
        validation.failures.append("no training_diagnostic rows found")

    for index, row in enumerate(train_rows, start=1):
        missing = sorted(STANDARD_ROW_KEYS - set(row))
        if missing:
            validation.failures.append(f"train row {index} missing keys {missing}")
            break
        if as_int(row.get("global_step")) != as_int(row.get("training_timestep")):
            validation.failures.append(f"train row {index} has inconsistent timestep fields")
            break

    eval_counts: dict[int, int] = {}
    for row in eval_rows:
        step = record_timestep(row)
        if step is not None:
            eval_counts[step] = eval_counts.get(step, 0) + 1
        _validate_evaluation_row(row, validation.agent_key, int(validation.seed), validation.failures)
    validation.evaluation_steps = sorted(eval_counts)
    if validation.evaluation_steps != list(EXPECTED_STEPS):
        validation.failures.append(
            f"evaluation steps={validation.evaluation_steps}, expected {list(EXPECTED_STEPS)}"
        )
    duplicate_eval_steps = sorted(step for step, count in eval_counts.items() if count != 1)
    if duplicate_eval_steps:
        validation.failures.append(f"duplicate evaluation rows at steps {duplicate_eval_steps}")
    if len(eval_rows) != len(EXPECTED_STEPS):
        validation.failures.append(
            f"evaluation row count={len(eval_rows)}, expected {len(EXPECTED_STEPS)}"
        )

    final_diagnostics = [
        row for row in diagnostic_rows if record_timestep(row) == TOTAL_TIMESTEPS
    ]
    if not final_diagnostics:
        validation.failures.append("no training_diagnostic row at 900000")
    else:
        wall_values = [as_float(row.get("wall_clock_time", row.get("elapsed_time"))) for row in final_diagnostics]
        sps_values = [as_float(row.get("SPS")) for row in final_diagnostics]
        if not any(value is not None and value > 0 for value in wall_values):
            validation.failures.append("final diagnostic lacks positive wall-clock time")
        if not any(value is not None and value > 0 for value in sps_values):
            validation.failures.append("final diagnostic lacks positive SPS")

    checkpoints = checkpoint_files(run_path)
    validation.checkpoint_steps = sorted(step for step, _ in checkpoints)
    if validation.checkpoint_steps != list(EXPECTED_STEPS):
        validation.failures.append(
            f"checkpoint steps={validation.checkpoint_steps}, expected {list(EXPECTED_STEPS)}"
        )
    if len(checkpoints) != len(EXPECTED_STEPS):
        validation.failures.append(
            f"checkpoint file count={len(checkpoints)}, expected {len(EXPECTED_STEPS)}"
        )

    if validation.friendly_agent == "qrl":
        circuit_values = [
            as_float(row.get("circuit_evaluations"))
            for row in records
            if record_timestep(row) == TOTAL_TIMESTEPS
        ]
        if not any(value is not None and value > 0 for value in circuit_values):
            validation.failures.append("QRL circuit accounting is missing at 900000")

    validation.valid = not validation.failures
    return validation


def _newness_key(candidate: CandidateValidation) -> tuple[datetime, str]:
    try:
        timestamp = datetime.strptime(candidate.run_name[:19], "%Y-%m-%d--%H-%M-%S")
    except ValueError:
        timestamp = datetime.fromtimestamp((candidate.path / "config.yaml").stat().st_mtime)
    return timestamp, candidate.run_name


def discover_lunarlander_candidates(logs_dir: Path | str = DEFAULT_LOGS_DIR) -> list[Path]:
    logs_path = Path(logs_dir).resolve()
    if not logs_path.is_dir():
        return []
    candidates: list[Path] = []
    for run_path in sorted(path for path in logs_path.iterdir() if path.is_dir()):
        config_path = run_path / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config = load_yaml(config_path)
        except (OSError, yaml.YAMLError):
            candidates.append(run_path)
            continue
        if config.get("env_id") == "LunarLander-v3":
            candidates.append(run_path)
    return candidates


def validate_cohort(logs_dir: Path | str = DEFAULT_LOGS_DIR) -> CohortValidation:
    candidates = [validate_candidate(path) for path in discover_lunarlander_candidates(logs_dir)]
    selected: dict[tuple[str, int], CandidateValidation] = {}
    ambiguities: dict[tuple[str, int], list[CandidateValidation]] = {}
    for friendly in AGENT_ORDER:
        for seed in SEEDS:
            combo = (friendly, seed)
            valid = [candidate for candidate in candidates if candidate.valid and candidate.combo == combo]
            if not valid:
                continue
            valid.sort(key=_newness_key, reverse=True)
            selected[combo] = valid[0]
            valid[0].selected = True
            if len(valid) > 1:
                ambiguities[combo] = valid
                for older in valid[1:]:
                    older.warnings.append(
                        f"valid duplicate not selected; newer valid run {valid[0].run_name!r} selected"
                    )
    missing = [
        (friendly, seed)
        for friendly in AGENT_ORDER
        for seed in SEEDS
        if (friendly, seed) not in selected
    ]
    return CohortValidation(candidates, selected, ambiguities, missing)


def display_combo(combo: tuple[str, int]) -> str:
    friendly, seed = combo
    return f"{AGENT_SPECS[friendly]['label']} seed {seed}"


def candidate_by_combo(
    candidates: Iterable[CandidateValidation], friendly: str, seed: int
) -> list[CandidateValidation]:
    return [candidate for candidate in candidates if candidate.combo == (friendly, seed)]
