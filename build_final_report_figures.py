"""Create the curated final figure set for the LunarLander report."""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("final_report_figures")
AGGREGATE_CSV = Path("lunarlander_aggregate_results.csv")
COMPUTE_CSV = Path("compute_cost_comparison.csv")
PARAM_CSV = Path("parameter_count_comparison.csv")
IBM_MD = Path("results/ibm_qppo_short_trained_inference_results.md")
INDEX_MD = OUT_DIR / "final_report_figures_index.md"

COLORS = {
    "PPO": "#4169e1",
    "PPO-tiny": "#00a676",
    "DQN": "#d95f02",
    "QPPO/QRL short-trained": "#7b3294",
    "IBM QPPO hardware inference": "#595959",
}


def as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def format_value(value: object, digits: int = 1) -> str:
    numeric = as_float(value)
    if numeric is None:
        return ""
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,}"
    return f"{numeric:,.{digits}f}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)


def save_fig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def aggregate_classical_evaluations() -> dict[str, list[tuple[float, float]]]:
    records = read_csv(AGGREGATE_CSV)
    series: dict[str, list[tuple[float, float]]] = {}
    for row in records:
        if row.get("included_in_plots") != "yes":
            continue
        if row.get("metric_type") != "evaluation":
            continue
        agent = row.get("agent_label", "")
        if agent not in {"PPO", "PPO-tiny", "DQN"}:
            continue
        timestep = as_float(row.get("timestep"))
        reward = as_float(row.get("evaluation_reward") or row.get("episode_reward"))
        if timestep is None or reward is None:
            continue
        series.setdefault(agent, []).append((timestep, reward))
    for values in series.values():
        values.sort()
    return series


def qppo_short_evaluations() -> list[tuple[float, float]]:
    points = []
    for result_path in sorted(Path("logs").glob("*qppo_short_trained_lunarlander*/result.json")):
        for record in read_jsonl(result_path):
            if record.get("metric_type") != "evaluation":
                continue
            timestep = as_float(record.get("training_timestep") or record.get("global_step"))
            reward = as_float(record.get("episode_reward"))
            if timestep is not None and reward is not None:
                points.append((timestep, reward))
    points.sort()
    return points


def plot_reward_comparison() -> tuple[str, str]:
    filename = "fig_reward_curves_classical_full_and_qppo_short.png"
    path = OUT_DIR / filename
    classical = aggregate_classical_evaluations()
    qppo = qppo_short_evaluations()

    fig, (ax_full, ax_short) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.35, 1]})
    for agent in ["PPO", "PPO-tiny", "DQN"]:
        points = classical.get(agent, [])
        if not points:
            continue
        xs, ys = zip(*points)
        ax_full.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.2, label=agent, color=COLORS[agent])
    ax_full.set_title("Full-training classical evaluation")
    ax_full.set_xlabel("Environment steps")
    ax_full.set_ylabel("Evaluation reward")
    ax_full.grid(color="#dddddd", linewidth=0.8)
    ax_full.legend(frameon=False)
    ax_full.set_xlim(left=0, right=2_000_000)

    if qppo:
        xs, ys = zip(*qppo)
        ax_short.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, color=COLORS["QPPO/QRL short-trained"])
    ax_short.axvline(25_000, color="#444444", linestyle="--", linewidth=1)
    ax_short.set_title("Short-trained QPPO feasibility only")
    ax_short.set_xlabel("Environment steps")
    ax_short.set_ylabel("Evaluation reward")
    ax_short.grid(color="#dddddd", linewidth=0.8)
    ax_short.set_xlim(left=0, right=25_000)
    ax_short.text(24_500, ax_short.get_ylim()[0], "25k-step limit", ha="right", va="bottom", fontsize=8)

    fig.suptitle("Reward Curves: Completed Classical Training vs Short QPPO Feasibility", y=1.04)
    save_fig(path)
    caption = (
        "Use this as the main reward-trajectory figure. The left panel contains only completed "
        "2M-step classical runs; the right panel isolates the 25k-step QPPO simulator feasibility run so it is not mistaken for a full fair benchmark."
    )
    return filename, caption


def plot_final_reward_comparison() -> tuple[str, str]:
    filename = "fig_final_reward_full_classical_with_qppo_feasibility.png"
    path = OUT_DIR / filename
    rows = read_csv(COMPUTE_CSV)
    selected = [r for r in rows if r["agent"] in {"PPO", "PPO-tiny", "DQN", "QPPO/QRL short-trained"}]
    labels = [r["agent"].replace(" short-trained", "\nshort-trained") for r in selected]
    values = [as_float(r.get("final_reward")) or 0 for r in selected]
    colors = [COLORS[r["agent"]] for r in selected]
    hatches = ["" if r["category"] == "Full-training classical" else "//" for r in selected]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=colors)
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_value(value, 1), ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_title("Final Reward: Full Classical Results vs Short QPPO Feasibility")
    ax.set_ylabel("Final evaluation reward")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    save_fig(path)
    caption = (
        "Use this for final reward comparison with care: PPO, PPO-tiny, and DQN are completed 2M-step classical runs, while QPPO is a separate 25k-step feasibility result marked with hatching."
    )
    return filename, caption


def plot_parameter_count() -> tuple[str, str]:
    filename = "fig_parameter_count_total_trainable.png"
    path = OUT_DIR / filename
    rows = read_csv(PARAM_CSV)
    labels = [r["agent"].replace(" short-trained", "\nshort-trained") for r in rows]
    values = [as_float(r["total_trainable_params"]) or 0 for r in rows]
    colors = [COLORS.get(r["agent"], "#666666") for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_value(value, 0), ha="center", va="bottom", fontsize=9)
    ax.set_title("Trainable Parameters by LunarLander Agent")
    ax.set_ylabel("Trainable parameters")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    save_fig(path)
    caption = (
        "Use this to discuss parameter efficiency. QPPO has fewer trainable parameters than full PPO but more than PPO-tiny, and parameter count should not be interpreted as performance."
    )
    return filename, caption


def plot_compute_metric(metric: str, filename: str, title: str, ylabel: str, log_scale: bool = False) -> tuple[str, str]:
    path = OUT_DIR / filename
    rows = read_csv(COMPUTE_CSV)
    labels = [r["agent"].replace(" short-trained", "\nshort-trained").replace(" hardware inference", "\nhardware inference") for r in rows]
    values = [as_float(r.get(metric)) or 0 for r in rows]
    colors = [COLORS.get(r["agent"], "#666666") for r in rows]
    hatches = [
        "" if r["category"] == "Full-training classical" else ("//" if "Short-trained" in r["category"] else "xx")
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    bars = ax.bar(labels, values, color=colors)
    for bar, hatch, value in zip(bars, hatches, values):
        bar.set_hatch(hatch)
        ax.text(bar.get_x() + bar.get_width() / 2, value, format_value(value, 1) if value else "0", ha="center", va="bottom", fontsize=8)
    if log_scale:
        ax.set_yscale("symlog", linthresh=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)
    save_fig(path)

    captions = {
        "SPS": "Use this to compare environment-training throughput. IBM inference is shown as zero because it has no environment-training SPS.",
        "wall_clock_time_seconds": "Use this to compare measured wall-clock cost by category. The IBM bar is total script/job time, not the approximately 2-second dashboard QPU execution time.",
        "circuit_evaluations": "Use this to show quantum execution overhead. Classical baselines have zero circuit evaluations, QPPO reports simulated circuit evaluations during short training, and IBM reports 5 circuits x 100 shots.",
    }
    return filename, captions[metric]


def parse_action_from_vector(text: str) -> tuple[int, str]:
    values = ast.literal_eval(text.strip("`"))
    action = max(range(len(values)), key=lambda i: values[i])
    names = ["noop", "fire_left", "fire_main", "fire_right"]
    return action, f"{names[action]} ({action})"


def ibm_table_rows() -> list[list[str]]:
    text = IBM_MD.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| State") or line.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        state = cells[0]
        sim_action = parse_action_from_vector(cells[1])[1]
        ibm_action = parse_action_from_vector(cells[2])[1]
        agreement = "Yes" if cells[4] == "True" else "No"
        rows.append([state, sim_action, ibm_action, agreement])
    return rows


def plot_ibm_agreement_table() -> tuple[str, str]:
    filename = "fig_ibm_qppo_inference_agreement_table.png"
    path = OUT_DIR / filename
    rows = ibm_table_rows()
    agreement_count = sum(1 for row in rows if row[-1] == "Yes")
    rate = agreement_count / len(rows) if rows else 0

    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    ax.axis("off")
    ax.set_title("IBM QPPO Hardware Inference Agreement: Inference Only", pad=14)
    table = ax.table(
        cellText=rows,
        colLabels=["State", "Simulator action", "IBM action", "Agreement"],
        loc="center",
        cellLoc="center",
        colWidths=[0.12, 0.31, 0.31, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        if row == 0:
            cell.set_facecolor("#efefef")
            cell.set_text_props(weight="bold")
        elif col == 3:
            cell.set_facecolor("#dff3e6" if cell.get_text().get_text() == "Yes" else "#f7dddd")
    ax.text(
        0.5,
        0.05,
        f"Agreement rate: {rate:.3f}; backend ibm_kingston; 5 fixed states; 100 shots; dashboard QPU execution approx. 2 seconds.",
        ha="center",
        va="center",
        fontsize=9,
        transform=ax.transAxes,
    )
    save_fig(path)
    caption = (
        "Use this only for the IBM inference feasibility section. It shows 4/5 action agreement between simulator and hardware on fixed states, not training reward or quantum advantage."
    )
    return filename, caption


def write_index(entries: list[tuple[str, str]]) -> None:
    lines = [
        "# Final Report Figures Index",
        "",
        "This folder contains the curated figures recommended for the final report. QPPO/QRL figures are labeled as short-trained feasibility unless they refer only to parameter count; IBM figures are inference-only hardware feasibility.",
        "",
    ]
    for i, (filename, caption) in enumerate(entries, start=1):
        lines.extend([f"## Figure {i}: `{filename}`", "", caption, ""])
    INDEX_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_out_dir()
    entries = [
        plot_reward_comparison(),
        plot_final_reward_comparison(),
        plot_parameter_count(),
        plot_compute_metric(
            "SPS",
            "fig_compute_sps_full_classical_vs_qppo_short.png",
            "SPS: Full Classical Training vs Short QPPO Feasibility",
            "Steps per second",
        ),
        plot_compute_metric(
            "wall_clock_time_seconds",
            "fig_compute_wall_clock_by_result_category.png",
            "Wall-Clock Cost by Result Category",
            "Seconds",
        ),
        plot_compute_metric(
            "circuit_evaluations",
            "fig_compute_circuit_evaluations_and_shots.png",
            "Circuit Evaluations: Simulated QPPO and IBM Inference",
            "Circuit evaluations / shots",
            log_scale=True,
        ),
        plot_ibm_agreement_table(),
    ]
    write_index(entries)
    shutil.copy2(COMPUTE_CSV, OUT_DIR / "compute_cost_comparison.csv")
    shutil.copy2(PARAM_CSV, OUT_DIR / "parameter_count_comparison.csv")
    print(f"Wrote {len(entries)} figures to {OUT_DIR}")
    print(f"Wrote {INDEX_MD}")


if __name__ == "__main__":
    main()
