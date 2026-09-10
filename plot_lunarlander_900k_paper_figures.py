"""Generate every primary and supplemental matched-900k paper figure from data."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from aggregate_lunarlander_900k import student_t_ci, wilson_interval
from lunarlander_900k_common import (
    AGENT_ORDER as FRIENDLY_ORDER,
    AGENT_SPECS,
    EXPECTED_STEPS,
    SEEDS,
    SUCCESS_THRESHOLD,
    TOTAL_TIMESTEPS,
    display_combo,
    validate_cohort,
)
from lunarlander_900k_plot_style import (
    AGENT_COLORS,
    AGENT_MARKERS,
    AGENT_ORDER,
    DOUBLE_COLUMN_WIDTH,
    SINGLE_COLUMN_WIDTH,
    apply_paper_style,
    save_figure_png_pdf,
    style_axis,
)


ASSET_DIR = Path("lunarlander_matched_900k_assets")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required generated dataset is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Missing/non-numeric {context}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {context}: {value!r}")
    return parsed


def load_validated_runs(logs_dir: Path) -> dict[tuple[str, int], Any]:
    cohort = validate_cohort(logs_dir)
    if not cohort.passed:
        raise RuntimeError(
            "Validator has not passed; missing "
            + ", ".join(display_combo(combo) for combo in cohort.missing)
        )
    if cohort.ambiguities:
        ambiguous = ", ".join(display_combo(combo) for combo in cohort.ambiguities)
        raise RuntimeError(
            "Multiple completely valid runs exist for these agent/seed pairs: "
            + ambiguous
            + ". Resolve or archive the duplicate before plotting."
        )
    return cohort.selected


def load_evaluation_data(asset_dir: Path) -> list[dict[str, str]]:
    return _read_csv(asset_dir / "matched_900k_evaluation_checkpoints.csv")


def compute_seed_checkpoint_means(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {agent: [] for agent in AGENT_ORDER}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        agent = row.get("agent", "")
        if agent not in grouped:
            raise ValueError(f"Unexpected agent in evaluation dataset: {agent!r}")
        step = int(row["training_timestep"])
        key = (agent, step)
        if key in seen:
            raise ValueError(f"Duplicate evaluation summary row for {agent} at {step}")
        seen.add(key)
        seed_values = [
            _number(row[f"seed_{seed}_evaluation_reward"], f"{agent} seed {seed} at {step}")
            for seed in SEEDS
        ]
        center = statistics.mean(seed_values)
        std = statistics.stdev(seed_values)
        ci_low, ci_high = student_t_ci(seed_values)
        if ci_low is None or ci_high is None:
            raise AssertionError("Three-seed Student-t interval was not produced")
        if not math.isclose(center, _number(row["mean_evaluation_reward"], "stored mean"), abs_tol=1e-8):
            raise ValueError(f"Stored mean disagrees with raw seeds for {agent} at {step}")
        grouped[agent].append(
            {
                "step": step,
                "seed_values": seed_values,
                "mean": center,
                "std": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "success_rate": 100 * _number(row["pooled_success_rate"], "success rate"),
                "wilson_low": 100 * _number(row["wilson95_lower"], "Wilson lower"),
                "wilson_high": 100 * _number(row["wilson95_upper"], "Wilson upper"),
            }
        )
    expected = {(agent, step) for agent in AGENT_ORDER for step in EXPECTED_STEPS}
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"Evaluation checkpoint coverage mismatch; missing={missing}, extra={extra}")
    for agent in grouped:
        grouped[agent].sort(key=lambda row: row["step"])
    return grouped


def _format_k(value: float, _: int) -> str:
    return f"{int(value / 1000)}k"


def _add_solved_threshold(ax: Any, label: bool = True) -> Any:
    return ax.axhline(
        SUCCESS_THRESHOLD,
        color="#333333",
        linestyle="--",
        linewidth=1.0,
        label="Solved threshold" if label else None,
        zorder=1,
    )


def _set_step_axis(ax: Any) -> None:
    ax.set_xlim(EXPECTED_STEPS[0], EXPECTED_STEPS[-1])
    ax.set_xticks(EXPECTED_STEPS)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_k))
    ax.set_xlabel("Environment interactions")


def plot_evaluation_reward(
    checkpoint_data: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.5), constrained_layout=True)
    for agent in AGENT_ORDER:
        rows = checkpoint_data[agent]
        steps = [row["step"] for row in rows]
        for seed_index, seed in enumerate(SEEDS):
            ax.plot(
                steps,
                [row["seed_values"][seed_index] for row in rows],
                color=AGENT_COLORS[agent],
                linewidth=0.75,
                alpha=0.2,
                marker=AGENT_MARKERS[agent],
                markersize=2.2,
                zorder=2,
            )
        ax.fill_between(
            steps,
            [row["ci_low"] for row in rows],
            [row["ci_high"] for row in rows],
            color=AGENT_COLORS[agent],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            steps,
            [row["mean"] for row in rows],
            label=agent,
            color=AGENT_COLORS[agent],
            marker=AGENT_MARKERS[agent],
            markersize=4.5,
            linewidth=1.9,
            zorder=3,
        )
    _add_solved_threshold(ax)
    _set_step_axis(ax)
    ax.set_ylabel("Evaluation reward")
    style_axis(ax)
    ax.legend(frameon=False, ncols=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save_figure_png_pdf(fig, output_dir / "01_evaluation_reward_vs_steps_900k")
    plt.close(fig)


def plot_final_evaluation(
    checkpoint_data: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.3), constrained_layout=True)
    offsets = (-0.09, 0.0, 0.09)
    for x, agent in enumerate(AGENT_ORDER):
        final = checkpoint_data[agent][-1]
        for offset, value in zip(offsets, final["seed_values"]):
            ax.scatter(
                x + offset,
                value,
                s=27,
                color=AGENT_COLORS[agent],
                marker=AGENT_MARKERS[agent],
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
        ax.errorbar(
            x,
            final["mean"],
            yerr=[[final["mean"] - final["ci_low"]], [final["ci_high"] - final["mean"]]],
            color=AGENT_COLORS[agent],
            ecolor=AGENT_COLORS[agent],
            marker="P",
            markeredgecolor="#222222",
            markeredgewidth=0.7,
            markersize=8,
            capsize=4,
            elinewidth=1.2,
            zorder=5,
        )
    _add_solved_threshold(ax, label=False)
    ax.set_xticks(range(len(AGENT_ORDER)), AGENT_ORDER)
    ax.set_ylabel("Final evaluation reward")
    ax.margins(x=0.12)
    style_axis(ax)
    handles = [
        Line2D([], [], linestyle="none", marker="o", color="#555555", label="Seed values"),
        Line2D([], [], linestyle="none", marker="P", markeredgecolor="#222222", color="#777777", markersize=8, label="3-seed mean"),
        Line2D([], [], color="#555555", marker="_", markersize=10, label="95% t CI"),
        Line2D([], [], color="#333333", linestyle="--", label="Solved threshold"),
    ]
    ax.legend(handles=handles, frameon=False, ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save_figure_png_pdf(fig, output_dir / "02_final_evaluation_900k_all_seeds")
    plt.close(fig)


def plot_evaluation_success(
    checkpoint_data: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.35), constrained_layout=True)
    maximum = 0.0
    for agent in AGENT_ORDER:
        rows = checkpoint_data[agent]
        steps = [row["step"] for row in rows]
        rates = [row["success_rate"] for row in rows]
        maximum = max(maximum, *rates)
        ax.fill_between(
            steps,
            [row["wilson_low"] for row in rows],
            [row["wilson_high"] for row in rows],
            color=AGENT_COLORS[agent],
            alpha=0.11,
            linewidth=0,
        )
        ax.plot(
            steps,
            rates,
            color=AGENT_COLORS[agent],
            marker=AGENT_MARKERS[agent],
            markersize=4.5,
            label=agent,
        )
    _set_step_axis(ax)
    ax.set_ylim(0, 100 if maximum >= 80 else max(10, math.ceil(maximum * 1.2 / 10) * 10))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Evaluation success rate")
    style_axis(ax)
    ax.legend(frameon=False, ncols=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save_figure_png_pdf(fig, output_dir / "03_evaluation_success_rate_900k")
    plt.close(fig)


def _model_rows(asset_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(asset_dir / "matched_900k_model_summary.csv")
    if len(rows) != len(AGENT_ORDER) or {row["agent"] for row in rows} != set(AGENT_ORDER):
        raise ValueError("Model summary must contain exactly the four primary agents")
    for row in rows:
        if int(row["seed_count"]) != 3 or row["status"] != "COMPLETE":
            raise ValueError(f"Incomplete model summary for {row['agent']}")
    return {row["agent"]: row for row in rows}


def _parameter_rows(asset_dir: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(asset_dir / "matched_900k_parameter_summary.csv")
    return {row["agent"]: row for row in rows}


def plot_performance_vs_parameters(
    model_rows: dict[str, dict[str, str]],
    parameter_rows: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.7), constrained_layout=True)
    counts = []
    offsets = {"PPO": (4, -12), "QRL": (4, 6), "PPO-tiny": (4, 6), "DQN": (-30, 6)}
    for agent in AGENT_ORDER:
        params = _number(parameter_rows[agent]["total_trainable_parameters"], "parameters")
        row = model_rows[agent]
        reward = _number(row["mean_final_evaluation_reward"], "mean final reward")
        low = _number(row["t95_ci_lower"], "CI lower")
        high = _number(row["t95_ci_upper"], "CI upper")
        counts.append(params)
        ax.errorbar(
            params,
            reward,
            yerr=[[reward - low], [high - reward]],
            fmt=AGENT_MARKERS[agent],
            color=AGENT_COLORS[agent],
            capsize=3,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
        ax.annotate(
            agent,
            (params, reward),
            xytext=offsets[agent],
            textcoords="offset points",
            fontsize=8,
        )
    if max(counts) / min(counts) >= 100:
        ax.set_xscale("log")
    _add_solved_threshold(ax, label=False)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Final evaluation reward")
    style_axis(ax)
    save_figure_png_pdf(fig, output_dir / "04_performance_vs_parameters_900k")
    plt.close(fig)


def _compute_rows(asset_dir: Path) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, dict[str, str]]]:
    rows = _read_csv(asset_dir / "matched_900k_compute_summary.csv")
    seed_rows: dict[tuple[str, int], dict[str, str]] = {}
    aggregate_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["row_type"] == "seed":
            key = (row["agent"], int(row["seed"]))
            if key in seed_rows:
                raise ValueError(f"Duplicate compute row: {key}")
            seed_rows[key] = row
        elif row["row_type"] == "aggregate":
            aggregate_rows[row["agent"]] = row
    expected = {(agent, seed) for agent in AGENT_ORDER for seed in SEEDS}
    if set(seed_rows) != expected:
        raise ValueError("Compute dataset does not contain exactly one row per agent/seed")
    return seed_rows, aggregate_rows


def plot_compute_cost(
    seed_rows: dict[tuple[str, int], dict[str, str]],
    aggregate_rows: dict[str, dict[str, str]],
    parameter_rows: dict[str, dict[str, str]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN_WIDTH, 3.25), constrained_layout=True)
    offsets = (-0.09, 0.0, 0.09)
    all_hours = []
    for x, agent in enumerate(AGENT_ORDER):
        values = [
            _number(seed_rows[(agent, seed)]["wall_clock_hours"], f"{agent} wall-clock")
            for seed in SEEDS
        ]
        all_hours.extend(values)
        for offset, value in zip(offsets, values):
            axes[0].scatter(
                x + offset,
                value,
                color=AGENT_COLORS[agent],
                marker=AGENT_MARKERS[agent],
                s=25,
                zorder=3,
            )
        axes[0].scatter(
            x,
            statistics.mean(values),
            marker="P",
            s=62,
            color=AGENT_COLORS[agent],
            edgecolor="#222222",
            linewidth=0.6,
            zorder=4,
        )
        cumulative = _number(
            aggregate_rows[agent]["wall_clock_cumulative_hours"], f"{agent} cumulative time"
        )
        axes[0].annotate(
            f"Σ {cumulative:.1f} h",
            (x, max(values)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    if max(all_hours) / min(all_hours) >= 20:
        axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(AGENT_ORDER)), AGENT_ORDER)
    axes[0].set_ylabel("Training time (hours)")
    style_axis(axes[0])

    qrl_values = [
        _number(seed_rows[("QRL", seed)]["circuit_evaluations"], f"QRL seed {seed} circuits")
        for seed in SEEDS
    ]
    for offset, value in zip(offsets, qrl_values):
        axes[1].scatter(
            offset,
            value,
            color=AGENT_COLORS["QRL"],
            marker=AGENT_MARKERS["QRL"],
            s=28,
            zorder=3,
        )
    axes[1].scatter(
        0,
        statistics.mean(qrl_values),
        marker="P",
        s=65,
        color=AGENT_COLORS["QRL"],
        edgecolor="#222222",
        linewidth=0.6,
        zorder=4,
    )
    cumulative_circuits = _number(
        aggregate_rows["QRL"]["circuit_evaluations_cumulative"], "QRL cumulative circuits"
    )
    qrl_params = parameter_rows["QRL"]
    axes[1].set_xticks([0], ["QRL"])
    axes[1].set_xlim(-0.55, 0.55)
    axes[1].set_ylabel("Circuit evaluations")
    axes[1].text(
        0.03,
        0.97,
        f"Cumulative: {cumulative_circuits:,.0f}\n"
        f"{qrl_params['qubits']} qubits; {qrl_params['variational_layers']} layers; "
        f"{qrl_params['quantum_circuit_parameters']} quantum parameters\n"
        "Classical models: no quantum-circuit evaluations.",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    style_axis(axes[1])
    save_figure_png_pdf(fig, output_dir / "05_compute_cost_900k")
    plt.close(fig)


def plot_training_reward(asset_dir: Path, output_dir: Path) -> None:
    rows = _read_csv(asset_dir / "matched_900k_all_metrics.csv")
    rewards: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row["metric_type"] != "train_episode":
            continue
        key = (row["agent"], int(row["seed"]))
        rewards[key].append(
            (int(row["training_timestep"]), _number(row["episode_reward"], "training reward"))
        )
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.5), constrained_layout=True)
    for agent in AGENT_ORDER:
        seed_series = []
        for seed in SEEDS:
            points = sorted(rewards[(agent, seed)])
            checkpoint_values = []
            for step in EXPECTED_STEPS:
                eligible = [reward for timestep, reward in points if timestep <= step]
                if not eligible:
                    raise ValueError(f"No training episodes for {agent} seed {seed} by {step}")
                checkpoint_values.append(statistics.mean(eligible[-100:]))
            seed_series.append(checkpoint_values)
            ax.plot(
                EXPECTED_STEPS,
                checkpoint_values,
                color=AGENT_COLORS[agent],
                linewidth=0.75,
                alpha=0.22,
                marker=AGENT_MARKERS[agent],
                markersize=2.2,
            )
        means = [statistics.mean(values) for values in zip(*seed_series)]
        ax.plot(
            EXPECTED_STEPS,
            means,
            color=AGENT_COLORS[agent],
            marker=AGENT_MARKERS[agent],
            markersize=4,
            linewidth=1.9,
            label=agent,
        )
    _add_solved_threshold(ax)
    _set_step_axis(ax)
    ax.set_ylabel("Training reward (last 100 episodes)")
    style_axis(ax)
    ax.legend(frameon=False, ncols=5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save_figure_png_pdf(fig, output_dir / "S1_training_reward_vs_steps_900k")
    plt.close(fig)


def plot_best_qrl_vs_classical(
    checkpoint_data: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    labels = list(AGENT_ORDER)
    values = []
    for agent in labels:
        final = checkpoint_data[agent][-1]
        values.append(max(final["seed_values"]) if agent == "QRL" else final["mean"])
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.1), constrained_layout=True)
    for x, (agent, value) in enumerate(zip(labels, values)):
        ax.scatter(
            x,
            value,
            s=58,
            marker=AGENT_MARKERS[agent],
            color=AGENT_COLORS[agent],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    _add_solved_threshold(ax, label=False)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Final evaluation reward")
    ax.text(
        0.5,
        0.98,
        "QRL = best seed; classical = 3-seed means",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    style_axis(ax)
    save_figure_png_pdf(fig, output_dir / "S2_best_qrl_vs_classical_900k")
    plt.close(fig)


def plot_parameter_bar(parameter_rows: dict[str, dict[str, str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.1), constrained_layout=True)
    values = [
        _number(parameter_rows[agent]["total_trainable_parameters"], "parameter count")
        for agent in AGENT_ORDER
    ]
    bars = ax.bar(
        AGENT_ORDER,
        values,
        color=[AGENT_COLORS[agent] for agent in AGENT_ORDER],
        width=0.62,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Trainable parameters")
    style_axis(ax)
    save_figure_png_pdf(fig, output_dir / "S3_parameter_count_bar_900k")
    plt.close(fig)


def plot_throughput(
    seed_rows: dict[tuple[str, int], dict[str, str]], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COLUMN_WIDTH, 3.1), constrained_layout=True)
    offsets = (-0.09, 0.0, 0.09)
    all_values = []
    for x, agent in enumerate(AGENT_ORDER):
        values = [_number(seed_rows[(agent, seed)]["SPS"], f"{agent} SPS") for seed in SEEDS]
        all_values.extend(values)
        for offset, value in zip(offsets, values):
            ax.scatter(
                x + offset,
                value,
                color=AGENT_COLORS[agent],
                marker=AGENT_MARKERS[agent],
                s=25,
                zorder=3,
            )
        ax.scatter(
            x,
            statistics.mean(values),
            marker="P",
            s=62,
            color=AGENT_COLORS[agent],
            edgecolor="#222222",
            linewidth=0.6,
            zorder=4,
        )
    if max(all_values) / min(all_values) >= 20:
        ax.set_yscale("log")
    ax.set_xticks(range(len(AGENT_ORDER)), AGENT_ORDER)
    ax.set_ylabel("Steps per second")
    style_axis(ax)
    save_figure_png_pdf(fig, output_dir / "S4_throughput_sps_900k")
    plt.close(fig)


def write_captions(output_dir: Path) -> None:
    content = """# Matched 900k figure captions

## Primary

1. **Deterministic evaluation performance over training.** Lines show the mean and 95% Student-t confidence interval across three seeds; faint paths show individual seeds. Each checkpoint comprises 10 deterministic evaluation episodes per seed. No smoothing or interpolation is applied.
2. **Final deterministic evaluation at 900k interactions.** Raw values for seeds 0, 1, and 2 are shown with the three-seed mean and 95% Student-t confidence interval (`df=2`).
3. **Deterministic evaluation success over training.** Success is pooled over 30 evaluation episodes per agent and checkpoint (10 episodes × three seeds); shading shows Wilson 95% binomial intervals.
4. **Performance versus trainable parameter count.** Counts are obtained by programmatically instantiating the exact trained architectures; vertical intervals are 95% Student-t intervals across three seeds.
5. **Matched-budget computational cost.** Panel A shows per-seed and mean wall-clock training time, with cumulative three-seed time annotated. Panel B reports QRL circuit evaluations; classical models perform no quantum-circuit evaluations.

## Supplemental

- S1: Training reward, explicitly separated from deterministic evaluation reward; points are last-100 episode means at the nine reporting checkpoints.
- S2: Best QRL seed vs classical three-seed means. This selected-seed diagnostic is not used for the primary conclusion.
- S3: Exact instantiated trainable parameter counts.
- S4: Per-seed and mean throughput; supplemental because all agents receive the same interaction budget.
"""
    (output_dir / "lunarlander_900k_figure_captions.md").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR / "figures")
    args = parser.parse_args()

    load_validated_runs(args.logs_dir)
    seed_report = _read_csv(args.asset_dir / "matched_900k_seed_report.csv")
    keys = [(row["friendly_agent"], int(row["seed"])) for row in seed_report]
    expected_keys = [(friendly, seed) for friendly in FRIENDLY_ORDER for seed in SEEDS]
    if sorted(keys) != sorted(expected_keys) or len(keys) != len(set(keys)):
        raise RuntimeError("Seed report has missing or duplicate agent/seed rows")
    for row in seed_report:
        if row["status"] != "PASS" or int(row["total_timesteps"]) != TOTAL_TIMESTEPS:
            raise RuntimeError(f"Non-900k or incomplete primary row: {row}")

    apply_paper_style()
    checkpoint_data = compute_seed_checkpoint_means(load_evaluation_data(args.asset_dir))
    model_rows = _model_rows(args.asset_dir)
    parameter_rows = _parameter_rows(args.asset_dir)
    seed_compute, aggregate_compute = _compute_rows(args.asset_dir)

    plot_evaluation_reward(checkpoint_data, args.output_dir)
    plot_final_evaluation(checkpoint_data, args.output_dir)
    plot_evaluation_success(checkpoint_data, args.output_dir)
    plot_performance_vs_parameters(model_rows, parameter_rows, args.output_dir)
    plot_compute_cost(seed_compute, aggregate_compute, parameter_rows, args.output_dir)
    plot_training_reward(args.asset_dir, args.output_dir)
    plot_best_qrl_vs_classical(checkpoint_data, args.output_dir)
    plot_parameter_bar(parameter_rows, args.output_dir)
    plot_throughput(seed_compute, args.output_dir)
    write_captions(args.output_dir)
    print(f"Wrote all primary and supplemental figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FIGURE GENERATION REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1)
