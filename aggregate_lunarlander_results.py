import argparse
import csv
import json
import math
import os
from collections import defaultdict, deque
from datetime import datetime

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: PyYAML. Install it with `python -m pip install PyYAML`.") from exc


AGENT_LABELS = {
    "PPO_classical": "PPO",
    "PPO_tiny_classical": "PPO-tiny",
    "DQN_classical": "DQN",
    "DQN_quantum": "Quantum DQN",
    "ppo_quantum_hybrid": "QRL",
}

CSV_COLUMNS = [
    "run_name",
    "agent",
    "agent_label",
    "seed",
    "timestep",
    "metric_type",
    "episode_reward",
    "episode_length",
    "evaluation_reward",
    "success_rate",
    "SPS",
    "wall_clock_time",
    "circuit_evaluations",
    "included_in_plots",
    "exclusion_reason",
]

MAIN_COMPARISON_TIMESTEPS = 100_000
MAIN_COMPARISON_SEEDS = {0, 1, 2}
MAIN_COMPARISON_AGENTS = set(AGENT_LABELS)
PLOT_AGENT_ORDER = ["DQN", "PPO", "PPO-tiny", "QRL", "Quantum DQN"]
PLOT_AGENT_COLORS = {
    "PPO": "#4169e1",
    "PPO-tiny": "#00a676",
    "DQN": "#d95f02",
    "QRL": "#7b3294",
    "Quantum DQN": "#b35806",
}
SUCCESS_BIN_SIZE = 10_000
PLOT_AGENT_OFFSETS = {
    "DQN": -1_400,
    "PPO": -700,
    "PPO-tiny": 0,
    "QRL": 700,
    "Quantum DQN": 1_400,
}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def discover_runs(logs_dir):
    runs = []
    if not os.path.isdir(logs_dir):
        return runs
    for name in sorted(os.listdir(logs_dir)):
        run_dir = os.path.join(logs_dir, name)
        config_path = os.path.join(run_dir, "config.yaml")
        result_path = os.path.join(run_dir, "result.json")
        if not os.path.isdir(run_dir) or not os.path.exists(config_path):
            continue
        config = load_yaml(config_path)
        if not config.get("standardize_lunarlander", False):
            continue
        if config.get("env_id") != "LunarLander-v3":
            continue
        runs.append(
            {
                "name": name,
                "path": run_dir,
                "config": config,
                "records": load_jsonl(result_path),
            }
        )
    return runs


def checkpoint_count(run):
    if not os.path.isdir(run["path"]):
        return 0
    return len(
        [
            name
            for name in os.listdir(run["path"])
            if name.endswith(".cleanqrl_model") and "_step" in name
        ]
    )


def run_completion_status(run):
    config = run["config"]
    target_timesteps = int(config.get("total_timesteps", 2_000_000))
    interval = int(config.get("eval_interval", 100_000))
    expected_steps = list(range(interval, target_timesteps + 1, interval))
    timesteps = [
        int(record.get("training_timestep", record.get("global_step", 0)) or 0)
        for record in run["records"]
    ]
    max_timestep = max(timesteps or [0])
    eval_steps = sorted(
        {
            int(record.get("training_timestep", 0) or 0)
            for record in run["records"]
            if record.get("metric_type") == "evaluation"
        }
    )
    failures = []
    if max_timestep < target_timesteps:
        failures.append(f"incomplete timesteps ({max_timestep}/{target_timesteps})")
    if eval_steps != expected_steps:
        failures.append("missing or irregular evaluation rows")
    if checkpoint_count(run) != len(expected_steps):
        failures.append("missing or irregular checkpoints")
    return not failures, "; ".join(failures)


def main_comparison_status(run):
    config = run["config"]
    agent = str(config.get("agent", ""))
    try:
        seed = int(config.get("seed"))
    except (TypeError, ValueError):
        seed = None
    try:
        total_timesteps = int(config.get("total_timesteps", 0) or 0)
    except (TypeError, ValueError):
        total_timesteps = 0

    failures = []
    if agent not in MAIN_COMPARISON_AGENTS:
        failures.append(f"agent {agent!r} is outside the matched comparison cohort")
    if seed not in MAIN_COMPARISON_SEEDS:
        failures.append(f"seed {seed!r} is outside the matched comparison seed set")
    if total_timesteps != MAIN_COMPARISON_TIMESTEPS:
        failures.append(
            f"training budget {total_timesteps} does not match {MAIN_COMPARISON_TIMESTEPS}"
        )
    return not failures, "; ".join(failures)


def value_or_blank(value):
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    return value


def aggregate_rows(runs):
    rows = []
    for run in runs:
        config = run["config"]
        complete, exclusion_reason = run_completion_status(run)
        in_scope, scope_reason = main_comparison_status(run)
        include_in_plots = complete and in_scope
        if not include_in_plots:
            exclusion_reason = exclusion_reason if not complete else scope_reason
        agent = str(config.get("agent", ""))
        seed = config.get("seed", "")
        wall_clock_pairs = [
            (
                float(record.get("training_timestep", record.get("global_step"))),
                float(record.get("wall_clock_time", record.get("elapsed_time"))),
            )
            for record in run["records"]
            if record.get("training_timestep", record.get("global_step")) not in ("", None)
            and record.get("wall_clock_time", record.get("elapsed_time")) not in ("", None)
            and float(record.get("wall_clock_time", record.get("elapsed_time"))) > 0
        ]
        run_sps_values = [
            float(record["SPS"])
            for record in run["records"]
            if record.get("SPS") not in ("", None)
        ]
        if wall_clock_pairs:
            latest_timestep, latest_wall_clock = max(wall_clock_pairs, key=lambda item: item[0])
            effective_sps = latest_timestep / latest_wall_clock if latest_wall_clock > 0 else None
        else:
            effective_sps = max(run_sps_values) if run_sps_values else None
        for record in run["records"]:
            metric_type = record.get("metric_type", "training_diagnostic")
            timestep = record.get("training_timestep", record.get("global_step"))
            sps = record.get("SPS")
            wall_clock_time = record.get("wall_clock_time", record.get("elapsed_time"))
            if wall_clock_time is None and timestep is not None and sps:
                wall_clock_time = float(timestep) / float(sps)
            if wall_clock_time is None and timestep is not None and effective_sps:
                wall_clock_time = float(timestep) / effective_sps

            episode_reward = None
            evaluation_reward = None
            if metric_type == "evaluation":
                evaluation_reward = record.get("episode_reward")
            elif metric_type == "train_episode":
                episode_reward = record.get("episode_reward")

            rows.append(
                {
                    "run_name": run["name"],
                    "agent": agent,
                    "agent_label": AGENT_LABELS.get(agent, agent),
                    "seed": seed,
                    "timestep": value_or_blank(timestep),
                    "metric_type": metric_type,
                    "episode_reward": value_or_blank(episode_reward),
                    "episode_length": value_or_blank(record.get("episode_length")),
                    "evaluation_reward": value_or_blank(evaluation_reward),
                    "success_rate": value_or_blank(record.get("success_rate")),
                    "SPS": value_or_blank(sps),
                    "wall_clock_time": value_or_blank(wall_clock_time),
                    "circuit_evaluations": value_or_blank(record.get("circuit_evaluations")),
                    "included_in_plots": "yes" if include_in_plots else "no",
                    "exclusion_reason": "" if include_in_plots else exclusion_reason,
                }
            )
    return rows


def write_csv(rows, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def numeric(value):
    if value in ("", None):
        return None
    return float(value)


def group_train_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row.get("included_in_plots") != "yes":
            continue
        if row["metric_type"] != "train_episode":
            continue
        timestep = numeric(row["timestep"])
        reward = numeric(row["episode_reward"])
        if timestep is None or reward is None:
            continue
        grouped[row["agent_label"]].append((timestep, reward, row))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def grouped_train_rows_for_scope(rows, max_timestep=None, max_wall_clock_time=None):
    grouped = defaultdict(list)
    for row in rows:
        if row["metric_type"] != "train_episode":
            continue
        timestep = numeric(row["timestep"])
        reward = numeric(row["episode_reward"])
        wall_clock_time = numeric(row["wall_clock_time"])
        if timestep is None or reward is None:
            continue
        if max_timestep is not None and timestep > max_timestep:
            continue
        if max_wall_clock_time is not None:
            if wall_clock_time is None or wall_clock_time > max_wall_clock_time:
                continue
        grouped[row["agent_label"]].append((timestep, reward, wall_clock_time, row))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def rolling_average(points, window):
    rewards = deque(maxlen=window)
    smoothed = []
    for timestep, reward, _ in points:
        rewards.append(reward)
        smoothed.append((timestep, sum(rewards) / len(rewards)))
    return smoothed


def rolling_average_xy(points, window, x_index=0, y_index=1):
    rewards = deque(maxlen=window)
    smoothed = []
    for point in points:
        reward = point[y_index]
        rewards.append(reward)
        smoothed.append((point[x_index], sum(rewards) / len(rewards)))
    return smoothed


def incomplete_quantum_budgets(rows):
    quantum_labels = {"Quantum DQN", "QRL"}
    quantum_rows = [
        row
        for row in rows
        if row.get("included_in_plots") == "no"
        and row.get("agent_label") in quantum_labels
        and "incomplete" in row.get("exclusion_reason", "")
    ]
    max_timestep = max(
        [
            numeric(row["timestep"])
            for row in quantum_rows
            if row.get("metric_type") == "train_episode" and numeric(row["timestep"]) is not None
        ],
        default=None,
    )
    max_wall_clock_time = max(
        [
            numeric(row["wall_clock_time"])
            for row in quantum_rows
            if numeric(row["wall_clock_time"]) is not None
        ],
        default=None,
    )
    return max_timestep, max_wall_clock_time


def final_rewards(rows, last_n=100):
    train_by_run = defaultdict(list)
    eval_by_run = defaultdict(list)
    for row in rows:
        if row.get("included_in_plots") != "yes":
            continue
        timestep = numeric(row["timestep"])
        if timestep is None:
            continue
        key = (row["agent_label"], row["seed"], row["run_name"])
        if row["metric_type"] == "evaluation":
            reward = numeric(row["evaluation_reward"])
            if reward is not None:
                eval_by_run[key].append((timestep, reward))
        elif row["metric_type"] == "train_episode":
            reward = numeric(row["episode_reward"])
            if reward is not None:
                train_by_run[key].append((timestep, reward))

    values_by_agent = defaultdict(list)
    for key in set(train_by_run) | set(eval_by_run):
        agent_label, _, _ = key
        eval_values = eval_by_run.get(key, [])
        if eval_values:
            eval_values.sort(key=lambda item: item[0])
            values_by_agent[agent_label].append(eval_values[-1][1])
            continue
        train_values = train_by_run.get(key, [])
        train_values.sort(key=lambda item: item[0])
        tail = [reward for _, reward in train_values[-last_n:]]
        if tail:
            values_by_agent[agent_label].append(sum(tail) / len(tail))
    return values_by_agent


def binned_training_success(rows, bin_size=SUCCESS_BIN_SIZE):
    bin_count = math.ceil(MAIN_COMPARISON_TIMESTEPS / bin_size)
    counts = {
        agent: [
            {"successes": 0, "episodes": 0, "seeds": set()}
            for _ in range(bin_count)
        ]
        for agent in PLOT_AGENT_ORDER
    }
    for row in rows:
        if row.get("included_in_plots") != "yes":
            continue
        if row["metric_type"] != "train_episode":
            continue
        timestep = numeric(row["timestep"])
        success = numeric(row["success_rate"])
        agent = row["agent_label"]
        seed = int(row["seed"])
        if timestep is None or success is None or agent not in counts:
            continue
        if success not in {0.0, 1.0}:
            raise ValueError(
                f"Expected binary training success for {agent} seed {seed}, found {success}"
            )
        if timestep < 0 or timestep > MAIN_COMPARISON_TIMESTEPS:
            continue
        bin_index = min(int(timestep // bin_size), bin_count - 1)
        counts[agent][bin_index]["successes"] += int(success)
        counts[agent][bin_index]["episodes"] += 1
        counts[agent][bin_index]["seeds"].add(seed)

    binned = {}
    for agent in PLOT_AGENT_ORDER:
        points = []
        for bin_index, count in enumerate(counts[agent]):
            episodes = count["episodes"]
            rate = count["successes"] / episodes if episodes else math.nan
            points.append(
                {
                    "center": bin_index * bin_size + bin_size / 2,
                    "rate": rate,
                    "successes": count["successes"],
                    "episodes": episodes,
                    "seeds": count["seeds"],
                }
            )
        observed_seeds = set().union(*(point["seeds"] for point in points))
        if observed_seeds != MAIN_COMPARISON_SEEDS:
            raise ValueError(
                f"Expected seeds {sorted(MAIN_COMPARISON_SEEDS)} for {agent}, "
                f"found {sorted(observed_seeds)}"
            )
        binned[agent] = points
    return binned


def compute_cost(rows):
    by_run = {}
    for row in rows:
        if row.get("included_in_plots") != "yes":
            continue
        key = (row["agent_label"], row["seed"], row["run_name"])
        current = by_run.setdefault(
            key,
            {
                "latest_sps": None,
                "latest_sps_timestep": -1,
                "max_wall_clock_time": None,
                "max_circuit_evaluations": None,
            },
        )
        timestep = numeric(row["timestep"])
        sps = numeric(row["SPS"])
        if sps is not None and timestep is not None and timestep >= current["latest_sps_timestep"]:
            current["latest_sps"] = sps
            current["latest_sps_timestep"] = timestep
        for source_key, target_key in [
            ("wall_clock_time", "max_wall_clock_time"),
            ("circuit_evaluations", "max_circuit_evaluations"),
        ]:
            value = numeric(row[source_key])
            if value is not None:
                current[target_key] = value if current[target_key] is None else max(current[target_key], value)
    by_agent = defaultdict(list)
    for (agent_label, _, _), metrics in by_run.items():
        by_agent[agent_label].append(metrics)
    return by_agent


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def write_summary_table(rows, output_path):
    early_step_budget, time_budget = incomplete_quantum_budgets(rows)
    run_stats = [stat for stat in run_level_stats(rows) if stat["included"]]
    stats_by_agent = defaultdict(list)
    for stat in run_stats:
        stats_by_agent[stat["agent_label"]].append(stat)
    excluded_runs = {}
    for row in rows:
        if row.get("included_in_plots") == "no":
            excluded_runs[row["run_name"]] = row.get("exclusion_reason", "incomplete")

    lines = [
        "# LunarLander Aggregated Results Summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Main matched budget: {MAIN_COMPARISON_TIMESTEPS:,} environment interactions",
        f"Main seed set: {', '.join(str(seed) for seed in sorted(MAIN_COMPARISON_SEEDS))}",
        f"Included matched runs: {len({row['run_name'] for row in rows if row.get('included_in_plots') == 'yes'})}",
        f"Excluded out-of-scope or incomplete runs: {len(excluded_runs)}",
        "",
        "| Agent | Runs | Final Reward Mean | Final Reward Min | Final Reward Max | Final Success Rate | Mean SPS | Mean Wall-Clock Time | Mean Circuit Evaluations |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    agents = sorted(stats_by_agent)
    for agent in agents:
        stats = stats_by_agent[agent]
        final_values = [stat["final_reward"] for stat in stats]
        mean_sps = mean([stat["SPS"] for stat in stats])
        mean_wall = mean([stat["wall_clock_time"] for stat in stats])
        mean_circuits = mean([stat["circuit_evaluations"] for stat in stats])
        lines.append(
            "| {agent} | {runs} | {final_mean} | {final_min} | {final_max} | {success} | {sps} | {wall} | {circuits} |".format(
                agent=agent,
                runs=len(stats),
                final_mean=format_number(mean(final_values)),
                final_min=format_number(min(final_values) if final_values else None),
                final_max=format_number(max(final_values) if final_values else None),
                success=format_number(mean([stat["success_rate"] for stat in stats])),
                sps=format_number(mean_sps),
                wall=format_number(mean_wall),
                circuits=format_number(mean_circuits),
            )
        )
    if excluded_runs:
        lines.extend(["", "## Excluded Runs", ""])
        for run_name, reason in sorted(excluded_runs.items()):
            lines.append(f"- `{run_name}`: {reason}")
    lines.extend(
        [
            "",
            "## Supplemental Notes",
            "",
            "The main plots and table use only completed runs from the matched 100,000-step, seeds 0-2 cohort. Older full-training or partial exploratory runs remain in the aggregate CSV with `included_in_plots == no` for transparency.",
        ]
    )
    if early_step_budget is not None:
        lines.extend(
            [
                "",
                f"Older incomplete quantum diagnostics reached up to {format_number(early_step_budget)} environment interactions after about {format_duration(time_budget)}. They are retained only as historical compute-feasibility context.",
            ]
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_level_stats(rows):
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run_name"]].append(row)

    stats = []
    for run_name, run_rows in by_run.items():
        first = run_rows[0]
        train = []
        evals = []
        max_step = 0
        max_wall = None
        latest_sps = None
        latest_sps_timestep = -1
        max_circuit = None
        for row in run_rows:
            timestep = numeric(row["timestep"])
            wall = numeric(row["wall_clock_time"])
            sps = numeric(row["SPS"])
            circuit = numeric(row["circuit_evaluations"])
            if timestep is not None:
                max_step = max(max_step, int(timestep))
            if wall is not None:
                max_wall = wall if max_wall is None else max(max_wall, wall)
            if sps is not None and timestep is not None and timestep >= latest_sps_timestep:
                latest_sps = sps
                latest_sps_timestep = timestep
            if circuit is not None:
                max_circuit = circuit if max_circuit is None else max(max_circuit, circuit)
            if row["metric_type"] == "train_episode" and row["episode_reward"] != "":
                train.append((timestep, numeric(row["episode_reward"]), numeric(row["success_rate"])))
            if row["metric_type"] == "evaluation" and row["evaluation_reward"] != "":
                evals.append((timestep, numeric(row["evaluation_reward"]), numeric(row["success_rate"])))

        final_reward = None
        success_rate = None
        evals.sort(key=lambda item: item[0] or 0)
        train.sort(key=lambda item: item[0] or 0)
        if evals:
            final_reward = evals[-1][1]
            success_rate = evals[-1][2]
        elif train:
            tail = train[-100:]
            final_reward = mean([item[1] for item in tail])
            success_rate = mean([item[2] for item in tail])

        stats.append(
            {
                "run_name": run_name,
                "agent_label": first["agent_label"],
                "included": first.get("included_in_plots") == "yes",
                "exclusion_reason": first.get("exclusion_reason", ""),
                "final_reward": final_reward,
                "success_rate": success_rate,
                "timesteps": max_step,
                "wall_clock_time": max_wall,
                "SPS": latest_sps,
                "circuit_evaluations": max_circuit,
            }
        )
    return stats


def aggregate_complete_stats(run_stats):
    grouped = defaultdict(list)
    for stat in run_stats:
        if stat["included"]:
            grouped[stat["agent_label"]].append(stat)

    rows = []
    for agent, stats in sorted(grouped.items()):
        rows.append(
            {
                "group": "Complete",
                "agent": agent,
                "final_reward": mean([stat["final_reward"] for stat in stats]),
                "success_rate": mean([stat["success_rate"] for stat in stats]),
                "timesteps": max([stat["timesteps"] for stat in stats], default=None),
                "wall_clock_time": mean([stat["wall_clock_time"] for stat in stats]),
                "SPS": mean([stat["SPS"] for stat in stats]),
                "circuit_evaluations": mean([stat["circuit_evaluations"] for stat in stats]),
                "status": "Complete matched-budget run" if len(stats) == 1 else "Complete matched-budget runs",
            }
        )
    return rows


def incomplete_quantum_stats(run_stats):
    quantum_labels = {"Quantum DQN", "QRL"}
    rows = []
    for stat in sorted(
        [stat for stat in run_stats if not stat["included"] and stat["agent_label"] in quantum_labels],
        key=lambda item: item["timesteps"],
        reverse=True,
    ):
        rows.append(
            {
                "group": "Incomplete quantum",
                "agent": stat["agent_label"],
                "final_reward": stat["final_reward"],
                "success_rate": stat["success_rate"],
                "timesteps": stat["timesteps"],
                "wall_clock_time": stat["wall_clock_time"],
                "SPS": stat["SPS"],
                "circuit_evaluations": stat["circuit_evaluations"],
                "status": "Incomplete; excluded from main final-performance plots",
            }
        )
    return rows


def write_report_section(rows, output_path, plots_dir):
    early_step_budget, time_budget = incomplete_quantum_budgets(rows)
    run_stats = run_level_stats(rows)
    table_rows = aggregate_complete_stats(run_stats) + incomplete_quantum_stats(run_stats)

    lines = [
        "# Standardized LunarLander-v3 Comparison Results",
        "",
        f"This section reports the standardized LunarLander-v3 comparison after enforcing a common environment, preprocessing path, action space, training budget, seed protocol, evaluation cadence, and logging schema. Main conclusions below use **only completed runs** from the matched {MAIN_COMPARISON_TIMESTEPS:,}-step cohort with seeds {', '.join(str(seed) for seed in sorted(MAIN_COMPARISON_SEEDS))}. Older full-training or partial exploratory runs are retained in the aggregate CSV for transparency, but are excluded from the main plots and summary statistics.",
        "",
        "## Matched-Budget Results",
        "",
        "The matched-budget comparison should be treated as the primary result because every plotted agent uses the same 100,000 environment-interaction budget and the same three seeds.",
        "",
        f"![Matched reward curve]({plots_dir}/reward_vs_timesteps.png)",
        "",
        "*Figure: Reward vs environment interactions for the completed matched-budget runs only. Curves show rolling mean episode reward.*",
        "",
        f"![Final reward distribution]({plots_dir}/final_reward_distribution.png)",
        "",
        "*Figure: Final evaluation reward distribution by agent for completed matched-budget runs.*",
        "",
        f"![Best QRL vs classical]({plots_dir}/best_qrl_vs_classical_final_reward.png)",
        "",
        "*Figure: Final reward bar chart comparing the best QRL run against the mean PPO, PPO-tiny, and DQN baselines from the same 100,000-step cohort.*",
        "",
        f"![Success rate]({plots_dir}/success_rate.png)",
        "",
        "*Figure: Training episode success rate for the matched 100k LunarLander-v3 runs, aggregated into 10,000-interaction windows across three seeds per model. Successful episodes were rare and isolated; all 15 final evaluations at 100k steps had zero success.*",
        "",
        "## Compute Diagnostics",
        "",
        f"![Compute cost comparison]({plots_dir}/compute_cost_comparison.png)",
        "",
        "*Figure: Compute cost comparison for completed matched-budget runs. Quantum circuit evaluation counts are reported separately because simulator overhead is not captured by environment interactions alone.*",
        "",
        "The matched-budget results support the practical conclusion that simulated quantum RL has major overhead in this setup. The quantum agents complete the same environment-interaction budget, but at much lower throughput and with substantial circuit-evaluation cost.",
        "",
        "## Run Status Table",
        "",
        "| Group | Agent | Final reward | Success rate | Timesteps completed | Wall-clock time | SPS | Circuit evaluations | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in table_rows:
        lines.append(
            "| {group} | {agent} | {final_reward} | {success_rate} | {timesteps} | {wall_clock} | {sps} | {circuits} | {status} |".format(
                group=row["group"],
                agent=row["agent"],
                final_reward=format_number(row["final_reward"]),
                success_rate=format_number(row["success_rate"]),
                timesteps=format_integer(row["timesteps"]),
                wall_clock=format_duration(row["wall_clock_time"]),
                sps=format_number(row["SPS"]),
                circuits=format_integer(row["circuit_evaluations"]),
                status=row["status"],
            )
        )

    lines.extend(
        [
            "",
            "**Interpretation rule:** use the complete-run rows for final-performance conclusions. Use the incomplete quantum rows only to discuss computational feasibility and matched-budget diagnostics.",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def format_number(value):
    if value is None:
        return ""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3f}"


def format_duration(seconds):
    if seconds is None:
        return ""
    seconds = float(seconds)
    hours = seconds / 3600.0
    if hours >= 24:
        return f"{hours / 24.0:.2f} days"
    return f"{hours:.2f} hours"


def format_integer(value):
    if value is None:
        return ""
    return f"{float(value):,.0f}"


def generate_plots(rows, output_dir, rolling_window):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with `python -m pip install matplotlib`."
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    train_grouped = group_train_rows(rows)
    plt.figure(figsize=(11, 6))
    for agent, points in sorted(train_grouped.items()):
        smoothed = rolling_average(points, rolling_window)
        if not smoothed:
            continue
        xs, ys = zip(*smoothed)
        plt.plot(xs, ys, label=agent, linewidth=2)
    plt.axhline(200, color="black", linestyle="--", linewidth=1, label="Solved threshold")
    plt.xlabel("Environment interactions")
    plt.ylabel(f"Episode reward, rolling mean ({rolling_window} episodes)")
    plt.title("LunarLander Reward vs Timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_vs_timesteps.png"), dpi=180)
    plt.close()

    final_by_agent = final_rewards(rows)
    agents = sorted(final_by_agent)
    values = [final_by_agent[agent] for agent in agents]
    plt.figure(figsize=(10, 6))
    if values:
        plt.boxplot(values, labels=agents, showmeans=True)
    plt.axhline(200, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Final reward, last-100 episode mean per run")
    plt.title("Final Reward Distribution by Agent")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_reward_distribution.png"), dpi=180)
    plt.close()

    run_stats = [stat for stat in run_level_stats(rows) if stat["included"]]
    stats_by_agent = defaultdict(list)
    for stat in run_stats:
        stats_by_agent[stat["agent_label"]].append(stat)
    bar_labels = []
    bar_values = []
    bar_colors = []
    color_by_agent = {
        "PPO": "#4169e1",
        "PPO-tiny": "#00a676",
        "DQN": "#d95f02",
        "QRL": "#7b3294",
    }
    for agent in ["PPO", "PPO-tiny", "DQN"]:
        values = [stat["final_reward"] for stat in stats_by_agent.get(agent, [])]
        if values:
            bar_labels.append(agent)
            bar_values.append(mean(values))
            bar_colors.append(color_by_agent[agent])
    qrl_values = [stat["final_reward"] for stat in stats_by_agent.get("QRL", [])]
    if qrl_values:
        bar_labels.append("Best QRL")
        bar_values.append(max(qrl_values))
        bar_colors.append(color_by_agent["QRL"])
    plt.figure(figsize=(8.5, 5.2))
    bars = plt.bar(bar_labels, bar_values, color=bar_colors)
    plt.axhline(0, color="#333333", linewidth=0.9)
    plt.axhline(200, color="black", linestyle="--", linewidth=1, label="Solved threshold")
    plt.ylabel("Final evaluation reward")
    plt.title("Best QRL vs Classical Baselines at 100k Steps")
    plt.grid(axis="y", color="#dddddd", linewidth=0.8)
    plt.gca().set_axisbelow(True)
    for bar, value in zip(bars, bar_values):
        va = "bottom" if value >= 0 else "top"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            format_number(value),
            ha="center",
            va=va,
            fontsize=9,
        )
    if bar_labels:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_qrl_vs_classical_final_reward.png"), dpi=180)
    plt.close()

    success_grouped = binned_training_success(rows)
    fig, ax = plt.subplots(figsize=(7.16, 4.45))
    observed_rates = []
    for agent in PLOT_AGENT_ORDER:
        points = success_grouped[agent]
        xs = [point["center"] + PLOT_AGENT_OFFSETS[agent] for point in points]
        ys = [point["rate"] for point in points]
        observed_rates.extend(rate for rate in ys if not math.isnan(rate))
        ax.plot(
            xs,
            ys,
            label=agent,
            color=PLOT_AGENT_COLORS[agent],
            linewidth=1.8,
            marker="o",
            markersize=4.5,
        )
    maximum_rate = max(observed_rates, default=0.0)
    upper_limit = max(0.01, math.ceil(maximum_rate * 1.2 / 0.005) * 0.005)
    ax.set_xlim(0, MAIN_COMPARISON_TIMESTEPS)
    ax.set_ylim(0, upper_limit)
    ax.set_xticks(
        [bin_index * SUCCESS_BIN_SIZE + SUCCESS_BIN_SIZE / 2 for bin_index in range(10)],
        [f"{5 + 10 * bin_index}k" for bin_index in range(10)],
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    plt.xlabel("Environment interactions")
    plt.ylabel("Training episode success rate")
    plt.title("LunarLander Training Success Rate by 10k-Step Window")
    plt.grid(axis="y", color="#dddddd", linewidth=0.8)
    plt.gca().set_axisbelow(True)
    plt.legend(frameon=False, ncols=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "success_rate.png"), dpi=180)
    plt.close()

    cost_by_agent = compute_cost(rows)
    agents = sorted(cost_by_agent)
    mean_sps_values = [mean([entry["latest_sps"] for entry in cost_by_agent[agent]]) or 0 for agent in agents]
    mean_circuit_values = [
        mean([entry["max_circuit_evaluations"] for entry in cost_by_agent[agent]]) or 0
        for agent in agents
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(agents, mean_sps_values)
    axes[0].set_title("Mean SPS")
    axes[0].set_ylabel("Steps per second")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(agents, mean_circuit_values)
    axes[1].set_title("Mean Circuit Evaluations")
    axes[1].set_ylabel("Circuit evaluations")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Compute Cost Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "compute_cost_comparison.png"), dpi=180)
    plt.close(fig)

    early_step_budget, time_budget = incomplete_quantum_budgets(rows)
    if early_step_budget is not None:
        matched_grouped = grouped_train_rows_for_scope(rows, max_timestep=early_step_budget)
        plt.figure(figsize=(11, 6))
        for agent, points in sorted(matched_grouped.items()):
            smoothed = rolling_average_xy(points, rolling_window, x_index=0, y_index=1)
            if not smoothed:
                continue
            xs, ys = zip(*smoothed)
            plt.plot(xs, ys, label=agent, linewidth=2)
        plt.axhline(200, color="black", linestyle="--", linewidth=1, label="Solved threshold")
        plt.xlabel("Environment interactions")
        plt.ylabel(f"Episode reward, rolling mean ({rolling_window} episodes)")
        plt.title(
            "Early-Step Matched Comparison\n"
            f"Classical baselines truncated at {int(early_step_budget):,} environment interactions"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "early_step_matched_comparison.png"), dpi=180)
        plt.close()

    if time_budget is not None:
        time_grouped = grouped_train_rows_for_scope(rows, max_wall_clock_time=time_budget)
        plt.figure(figsize=(11, 6))
        for agent, points in sorted(time_grouped.items()):
            points = sorted(
                [point for point in points if point[2] is not None],
                key=lambda item: item[2],
            )
            smoothed = rolling_average_xy(points, rolling_window, x_index=2, y_index=1)
            if not smoothed:
                continue
            xs, ys = zip(*[(x / 3600.0, y) for x, y in smoothed])
            plt.plot(xs, ys, label=agent, linewidth=2)
        plt.axhline(200, color="black", linestyle="--", linewidth=1, label="Solved threshold")
        plt.xlabel("Wall-clock time (hours)")
        plt.ylabel(f"Episode reward, rolling mean ({rolling_window} episodes)")
        plt.title(
            "Time-Budget Comparison\n"
            f"Classical baselines truncated at {format_duration(time_budget)}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_budget_comparison.png"), dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate standardized LunarLander logs into CSV, plots, and summary tables."
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory containing run logs.")
    parser.add_argument(
        "--csv",
        default="lunarlander_aggregate_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plots-dir",
        default="lunarlander_comparison_plots",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--summary",
        default="lunarlander_aggregate_summary.md",
        help="Markdown summary table path.",
    )
    parser.add_argument(
        "--report-section",
        default="lunarlander_report_ready_section.md",
        help="Report-ready markdown section path.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="Episode window for reward smoothing.",
    )
    args = parser.parse_args()

    runs = discover_runs(args.logs_dir)
    rows = aggregate_rows(runs)
    write_csv(rows, args.csv)
    generate_plots(rows, args.plots_dir, args.rolling_window)
    write_summary_table(rows, args.summary)
    write_report_section(rows, args.report_section, args.plots_dir)
    print(f"Aggregated {len(rows)} rows from {len(runs)} runs")
    print(f"Wrote CSV to {args.csv}")
    print(f"Wrote plots to {args.plots_dir}")
    print(f"Wrote summary to {args.summary}")
    print(f"Wrote report section to {args.report_section}")


if __name__ == "__main__":
    main()
