import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: PyYAML. Install it with:\n"
        "  python -m pip install PyYAML"
    ) from exc


EXPECTED_AGENTS = {
    "PPO_classical": "PPO classical",
    "PPO_tiny_classical": "PPO-tiny",
    "DQN_classical": "DQN classical",
    "DQN_quantum": "Quantum DQN",
    "ppo_quantum_hybrid": "Quantum PPO",
}

EXPECTED_ENV_ID = "LunarLander-v3"
EXPECTED_TOTAL_TIMESTEPS = 2_000_000
EXPECTED_SEED_SET = [0, 1, 2, 3, 4]
EXPECTED_EVAL_INTERVAL = 100_000
EXPECTED_CHECKPOINT_INTERVAL = 100_000
EXPECTED_EVAL_EPISODES = 10
EXPECTED_OBS_PREPROCESSING = "none"
EXPECTED_SUCCESS_THRESHOLD = 200.0
EXPECTED_EVAL_COUNT = EXPECTED_TOTAL_TIMESTEPS // EXPECTED_EVAL_INTERVAL
EXPECTED_CHECKPOINT_COUNT = EXPECTED_TOTAL_TIMESTEPS // EXPECTED_CHECKPOINT_INTERVAL

STANDARD_METRIC_KEYS = {
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
EVAL_EXTRA_KEYS = {"eval_episodes"}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "_parse_error": f"line {line_number}: {exc}",
                        "_raw": line[:200],
                    }
                )
    return records


def discover_runs(logs_dir):
    runs = []
    if not os.path.isdir(logs_dir):
        return runs
    for name in sorted(os.listdir(logs_dir)):
        run_dir = os.path.join(logs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        config_path = os.path.join(run_dir, "config.yaml")
        result_path = os.path.join(run_dir, "result.json")
        if not os.path.exists(config_path):
            continue
        config = load_yaml(config_path)
        if not config.get("standardize_lunarlander", False):
            continue
        if config.get("env_id") != EXPECTED_ENV_ID:
            continue
        runs.append(
            {
                "name": name,
                "path": run_dir,
                "config_path": config_path,
                "result_path": result_path,
                "config": config,
                "records": load_jsonl(result_path),
            }
        )
    return runs


def normalize_agent(agent):
    return str(agent)


def unique_values(runs, key):
    values = []
    for run in runs:
        value = run["config"].get(key)
        if value not in values:
            values.append(value)
    return values


def check_config(run):
    config = run["config"]
    checks = []
    expected_pairs = {
        "env_id": EXPECTED_ENV_ID,
        "total_timesteps": EXPECTED_TOTAL_TIMESTEPS,
        "seed_set": EXPECTED_SEED_SET,
        "eval_interval": EXPECTED_EVAL_INTERVAL,
        "checkpoint_interval": EXPECTED_CHECKPOINT_INTERVAL,
        "eval_episodes": EXPECTED_EVAL_EPISODES,
        "observation_preprocessing": EXPECTED_OBS_PREPROCESSING,
        "success_reward_threshold": EXPECTED_SUCCESS_THRESHOLD,
    }
    for key, expected in expected_pairs.items():
        actual = config.get(key)
        checks.append((key, actual == expected, actual, expected))
    return checks


def standard_records(run):
    return [
        record
        for record in run["records"]
        if record.get("metric_type") in {"train_episode", "evaluation"}
    ]


def check_logging_schema(run):
    failures = []
    train_count = 0
    eval_count = 0
    parse_errors = [record for record in run["records"] if "_parse_error" in record]
    for error in parse_errors:
        failures.append(error["_parse_error"])

    for idx, record in enumerate(standard_records(run), start=1):
        required = set(STANDARD_METRIC_KEYS)
        if record.get("metric_type") == "evaluation":
            required |= EVAL_EXTRA_KEYS
            eval_count += 1
        else:
            train_count += 1
        missing = sorted(required - set(record))
        if missing:
            failures.append(f"standard metric row {idx} missing {missing}")
        if record.get("global_step") != record.get("training_timestep"):
            failures.append(f"standard metric row {idx} has mismatched timestep fields")
    return failures, train_count, eval_count


def max_training_timestep(run):
    timesteps = [
        record.get("training_timestep", record.get("global_step"))
        for record in run["records"]
        if isinstance(record.get("training_timestep", record.get("global_step")), int)
    ]
    return max(timesteps) if timesteps else None


def checkpoint_steps(run):
    steps = []
    if not os.path.isdir(run["path"]):
        return steps
    for name in os.listdir(run["path"]):
        if not name.endswith(".cleanqrl_model"):
            continue
        marker = "_step"
        if marker not in name:
            continue
        step_text = name.split(marker, 1)[1].split(".", 1)[0]
        if step_text.isdigit():
            steps.append(int(step_text))
    return sorted(set(steps))


def evaluation_steps(run):
    return sorted(
        {
            int(record["training_timestep"])
            for record in run["records"]
            if record.get("metric_type") == "evaluation"
            and isinstance(record.get("training_timestep"), int)
        }
    )


def status_icon(ok):
    return "PASS" if ok else "FAIL"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_report(runs, logs_dir):
    by_agent = defaultdict(list)
    for run in runs:
        by_agent[normalize_agent(run["config"].get("agent"))].append(run)

    lines = [
        "# LunarLander Fairness Check Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Logs scanned: `{os.path.abspath(logs_dir)}`",
        f"Standardized runs found: {len(runs)}",
        "",
    ]

    missing_agents = sorted(set(EXPECTED_AGENTS) - set(by_agent))
    unexpected_agents = sorted(set(by_agent) - set(EXPECTED_AGENTS))
    observed_seed_set = sorted(
        {
            int(run["config"]["seed"])
            for run in runs
            if isinstance(run["config"].get("seed"), int)
        }
    )
    complete_seed_matrix = all(
        sorted(
            {
                int(run["config"]["seed"])
                for run in by_agent[agent]
                if isinstance(run["config"].get("seed"), int)
            }
        )
        == EXPECTED_SEED_SET
        for agent in EXPECTED_AGENTS
    )

    global_checks = [
        ("Expected agents present", not missing_agents, ", ".join(missing_agents) or "all present"),
        ("No unexpected standardized agents", not unexpected_agents, ", ".join(unexpected_agents) or "none"),
        ("Observed seed set", observed_seed_set == EXPECTED_SEED_SET, observed_seed_set),
        ("Every agent has every seed", complete_seed_matrix, "expected [0, 1, 2, 3, 4] per agent"),
        ("Same env_id", unique_values(runs, "env_id") == [EXPECTED_ENV_ID], unique_values(runs, "env_id")),
        (
            "Same total_timesteps",
            unique_values(runs, "total_timesteps") == [EXPECTED_TOTAL_TIMESTEPS],
            unique_values(runs, "total_timesteps"),
        ),
        (
            "Same eval protocol",
            unique_values(runs, "eval_interval") == [EXPECTED_EVAL_INTERVAL]
            and unique_values(runs, "eval_episodes") == [EXPECTED_EVAL_EPISODES],
            f"intervals={unique_values(runs, 'eval_interval')}, episodes={unique_values(runs, 'eval_episodes')}",
        ),
        (
            "Same checkpoint interval",
            unique_values(runs, "checkpoint_interval") == [EXPECTED_CHECKPOINT_INTERVAL],
            unique_values(runs, "checkpoint_interval"),
        ),
        (
            "Same observation preprocessing",
            unique_values(runs, "observation_preprocessing") == [EXPECTED_OBS_PREPROCESSING],
            unique_values(runs, "observation_preprocessing"),
        ),
    ]

    overall_pass = bool(runs) and all(ok for _, ok, _ in global_checks)
    lines.extend(
        [
            "## Overall Status",
            "",
            f"**{status_icon(overall_pass)}**",
            "",
            markdown_table(
                ["Check", "Status", "Details"],
                [(name, status_icon(ok), details) for name, ok, details in global_checks],
            ),
            "",
        ]
    )

    rows = []
    per_run_failures = {}
    for run in runs:
        config_checks = check_config(run)
        config_ok = all(ok for _, ok, _, _ in config_checks)
        schema_failures, train_count, eval_count = check_logging_schema(run)
        eval_steps = evaluation_steps(run)
        checkpoints = checkpoint_steps(run)
        max_step = max_training_timestep(run)
        completed_budget = max_step == EXPECTED_TOTAL_TIMESTEPS
        expected_steps = list(
            range(EXPECTED_EVAL_INTERVAL, EXPECTED_TOTAL_TIMESTEPS + 1, EXPECTED_EVAL_INTERVAL)
        )

        run_failures = []
        if not config_ok:
            run_failures.extend(
                [
                    f"{key}: actual {actual!r}, expected {expected!r}"
                    for key, ok, actual, expected in config_checks
                    if not ok
                ]
            )
        if schema_failures:
            run_failures.extend(schema_failures)
        if train_count == 0:
            run_failures.append("no train_episode standard metric rows found")
        if eval_count == 0:
            run_failures.append("no evaluation standard metric rows found")
        if not completed_budget:
            run_failures.append(
                f"max logged training_timestep is {max_step}, expected {EXPECTED_TOTAL_TIMESTEPS}"
            )
        if completed_budget and eval_count != EXPECTED_EVAL_COUNT:
            run_failures.append(
                f"evaluation row count is {eval_count}, expected {EXPECTED_EVAL_COUNT}"
            )
        if completed_budget and eval_steps != expected_steps:
            run_failures.append(f"evaluation steps are {eval_steps}, expected {expected_steps}")
        if completed_budget and len(checkpoints) != EXPECTED_CHECKPOINT_COUNT:
            run_failures.append(
                f"checkpoint count is {len(checkpoints)}, expected {EXPECTED_CHECKPOINT_COUNT}"
            )
        if completed_budget and checkpoints != expected_steps:
            run_failures.append(f"checkpoint steps are {checkpoints}, expected {expected_steps}")

        per_run_failures[run["name"]] = run_failures
        rows.append(
            [
                run["name"],
                run["config"].get("agent"),
                run["config"].get("seed"),
                status_icon(not run_failures),
                train_count,
                eval_count,
                max_step,
                ", ".join(str(step) for step in eval_steps[:5])
                + (" ..." if len(eval_steps) > 5 else ""),
                len(checkpoints),
            ]
        )

    lines.extend(
        [
            "## Run Inventory",
            "",
            markdown_table(
                [
                    "Run",
                    "Agent",
                    "Seed",
                    "Status",
                    "Train Rows",
                    "Eval Rows",
                    "Max Timestep",
                    "Eval Steps",
                    "Checkpoint Count",
                ],
                rows,
            )
            if rows
            else "No standardized LunarLander runs found.",
            "",
        ]
    )

    lines.extend(["## Agent Seed Coverage", ""])
    coverage_rows = []
    for agent, display_name in EXPECTED_AGENTS.items():
        seeds = sorted(
            {
                int(run["config"]["seed"])
                for run in by_agent.get(agent, [])
                if isinstance(run["config"].get("seed"), int)
            }
        )
        coverage_rows.append(
            [
                display_name,
                agent,
                seeds,
                status_icon(seeds == EXPECTED_SEED_SET),
            ]
        )
    lines.extend(
        [
            markdown_table(["Display Name", "Agent Key", "Observed Seeds", "Status"], coverage_rows),
            "",
        ]
    )

    failing_runs = {name: failures for name, failures in per_run_failures.items() if failures}
    lines.extend(["## Findings", ""])
    if not failing_runs and overall_pass:
        lines.append("All standardized LunarLander runs satisfy the fair-comparison checks.")
    else:
        if missing_agents:
            lines.append(f"- Missing expected agents: {', '.join(missing_agents)}")
        if observed_seed_set != EXPECTED_SEED_SET:
            lines.append(f"- Observed seed set is {observed_seed_set}, expected {EXPECTED_SEED_SET}.")
        for run_name, failures in failing_runs.items():
            lines.append(f"- `{run_name}`:")
            for failure in failures:
                lines.append(f"  - {failure}")
    lines.append("")

    lines.extend(
        [
            "## Comparison Rules Checked",
            "",
            "- All runs must use `LunarLander-v3`.",
            "- All runs must use `total_timesteps: 2000000`.",
            "- Every expected agent must have seeds `[0, 1, 2, 3, 4]`.",
            "- All runs must use `observation_preprocessing: none`.",
            "- All runs must use `eval_interval: 100000` and `eval_episodes: 10`.",
            "- All runs must use `checkpoint_interval: 100000`.",
            "- Standard train/eval rows must include `episode_reward`, `episode_length`, `success_rate`, and `training_timestep`.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check standardized LunarLander fair-comparison outputs."
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory containing run logs.")
    parser.add_argument(
        "--output",
        default="fairness_check_report.md",
        help="Markdown report output path.",
    )
    args = parser.parse_args()

    runs = discover_runs(args.logs_dir)
    report = build_report(runs, args.logs_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote fairness check report to {args.output}")


if __name__ == "__main__":
    main()
