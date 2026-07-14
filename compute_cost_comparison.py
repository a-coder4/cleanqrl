"""Build compute-cost comparison artifacts for matched LunarLander experiments.

Inputs:
- lunarlander_aggregate_results.csv for completed matched-budget runs.

Outputs:
- compute_cost_comparison.csv
- compute_cost_comparison.md
- compute_cost_sps.png
- compute_cost_wall_clock_time.png
- compute_cost_circuit_evaluations.png
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AGGREGATE_CSV = Path("lunarlander_aggregate_results.csv")
IBM_RESULTS_MD = Path("results/ibm_qppo_short_trained_inference_results.md")
OUTPUT_CSV = Path("compute_cost_comparison.csv")
OUTPUT_MD = Path("compute_cost_comparison.md")
SPS_PLOT = Path("compute_cost_sps.png")
WALL_CLOCK_PLOT = Path("compute_cost_wall_clock_time.png")
CIRCUIT_PLOT = Path("compute_cost_circuit_evaluations.png")
IBM_QPU_EXECUTION_TIME_SECONDS = 2.0
AGENT_ORDER = ["PPO", "PPO-tiny", "DQN", "Quantum DQN", "QRL"]
QUANTUM_AGENTS = {"Quantum DQN", "QRL"}
COLORS = {
    "PPO": "#4169e1",
    "PPO-tiny": "#00a676",
    "DQN": "#d95f02",
    "Quantum DQN": "#b35806",
    "QRL": "#7b3294",
}
IBM_HARDWARE_SPECS = {
    "Backend": "ibm_kingston",
    "Processor type": "Heron r2",
    "Region": "Washington DC us-east",
    "Qubits": "156",
    "Couplers": "176",
    "Median 2Q error": "2.01E-3",
    "Layered 2Q error": "3.42E-3",
    "CLOPS": "340K",
    "Median readout error": "8.91E-3",
    "Median T1": "256.51 us",
    "Median T2": "132.57 us",
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


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return statistics.mean(clean)


def latest_by_timestep(records: list[dict[str, object]], metric_type: str) -> dict[str, object] | None:
    candidates = [r for r in records if r.get("metric_type") == metric_type]
    if not candidates:
        return None
    return max(candidates, key=lambda r: as_float(r.get("timestep") or r.get("training_timestep") or r.get("global_step")) or -1)


def last_nonempty(records: list[dict[str, object]], key: str) -> float | None:
    value = None
    for record in records:
        parsed = as_float(record.get(key))
        if parsed is not None:
            value = parsed
    return value


def max_nonempty(records: list[dict[str, object]], key: str) -> float | None:
    values = [as_float(record.get(key)) for record in records]
    return max((value for value in values if value is not None), default=None)


def format_number(value: object, digits: int = 3) -> str:
    numeric = as_float(value)
    if numeric is None:
        return ""
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,}"
    return f"{numeric:,.{digits}f}"


def load_matched_training_rows() -> list[dict[str, object]]:
    if not AGGREGATE_CSV.exists():
        return []

    with AGGREGATE_CSV.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    included = [r for r in records if r.get("included_in_plots") == "yes"]
    by_run: dict[str, list[dict[str, object]]] = {}
    for record in included:
        by_run.setdefault(record["run_name"], []).append(record)

    run_summaries: list[dict[str, object]] = []
    for run_name, run_records in by_run.items():
        label = str(run_records[0].get("agent_label") or run_records[0].get("agent") or "")
        if label not in set(AGENT_ORDER):
            continue

        final_eval = latest_by_timestep(run_records, "evaluation")
        max_step = max(as_float(r.get("timestep")) or 0 for r in run_records)

        reward = None
        success = None
        if final_eval:
            reward = as_float(final_eval.get("evaluation_reward")) or as_float(final_eval.get("episode_reward"))
            success = as_float(final_eval.get("success_rate"))

        run_summaries.append(
            {
                "agent": label,
                "run_name": run_name,
                "environment_steps_completed": max_step,
                "wall_clock_time_seconds": max_nonempty(run_records, "wall_clock_time"),
                "qpu_execution_time_seconds": None,
                "SPS": last_nonempty(run_records, "SPS"),
                "circuit_evaluations": last_nonempty(run_records, "circuit_evaluations"),
                "final_reward": reward,
                "success_rate": success,
            }
        )

    rows = []
    for agent in AGENT_ORDER:
        runs = [r for r in run_summaries if r["agent"] == agent]
        if not runs:
            continue
        category = "Matched 100k quantum" if agent in QUANTUM_AGENTS else "Matched 100k classical"
        rows.append(
            {
                "agent": agent,
                "category": category,
                "runs": len(runs),
                "environment_steps_completed": mean(as_float(r["environment_steps_completed"]) for r in runs),
                "wall_clock_time_seconds": mean(as_float(r["wall_clock_time_seconds"]) for r in runs),
                "qpu_execution_time_seconds": None,
                "SPS": mean(as_float(r["SPS"]) for r in runs),
                "circuit_evaluations": mean(as_float(r["circuit_evaluations"]) for r in runs)
                if agent in QUANTUM_AGENTS
                else 0,
                "final_reward": mean(as_float(r["final_reward"]) for r in runs),
                "success_rate": mean(as_float(r["success_rate"]) for r in runs),
                "status": "complete matched-budget result",
                "notes": "Included by matched 100k fairness checker and aggregate plotting rules.",
            }
        )
    return rows


def load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_short_qppo_rows() -> list[dict[str, object]]:
    rows = []
    for result_path in sorted(Path("logs").glob("*qppo_short_trained_lunarlander*/result.json")):
        records = load_jsonl(result_path)
        if not records:
            continue
        latest_diag = latest_by_timestep(records, "training_diagnostic")
        final_eval = latest_by_timestep(records, "evaluation")
        max_step = max(as_float(r.get("training_timestep") or r.get("global_step")) or 0 for r in records)
        rows.append(
            {
                "agent": "QPPO/QRL short-trained",
                "category": "Short-trained simulator feasibility",
                "runs": 1,
                "environment_steps_completed": max_step,
                "wall_clock_time_seconds": as_float(latest_diag.get("wall_clock_time") if latest_diag else None),
                "qpu_execution_time_seconds": None,
                "SPS": as_float(latest_diag.get("SPS") if latest_diag else None),
                "circuit_evaluations": as_float(latest_diag.get("circuit_evaluations") if latest_diag else None),
                "final_reward": as_float(final_eval.get("episode_reward") if final_eval else None),
                "success_rate": as_float(final_eval.get("success_rate") if final_eval else None),
                "status": "short-trained only; not comparable as full-training final performance",
                "notes": f"Source: {result_path.as_posix()}",
            }
        )
    return rows


def load_ibm_inference_row() -> list[dict[str, object]]:
    if not IBM_RESULTS_MD.exists():
        return []
    text = IBM_RESULTS_MD.read_text(encoding="utf-8")
    table_rows = [
        line
        for line in text.splitlines()
        if line.startswith("| ") and not line.startswith("| State") and not line.startswith("| ---")
    ]
    state_count = len(table_rows)
    backend = ""
    shots = None
    job_id = ""
    circuit_time = None
    if table_rows:
        cells = [cell.strip().strip("`") for cell in table_rows[0].strip("|").split("|")]
        backend = cells[5]
        shots = as_float(cells[6])
        job_id = cells[7]
        circuit_time = as_float(cells[8])

    agreement_match = re.search(r"Action agreement rate:\s+\*\*([0-9.]+)\*\*", text)
    agreement = as_float(agreement_match.group(1)) if agreement_match else None

    return [
        {
            "agent": "IBM QPPO hardware inference",
            "category": "IBM inference-only feasibility",
            "runs": 1,
            "environment_steps_completed": 0,
            "wall_clock_time_seconds": circuit_time,
            "qpu_execution_time_seconds": IBM_QPU_EXECUTION_TIME_SECONDS,
            "SPS": None,
            "circuit_evaluations": state_count * shots if shots is not None else None,
            "final_reward": None,
            "success_rate": agreement,
            "status": "inference-only; no training and no environment rollout reward",
            "notes": f"{state_count} fixed states, {int(shots or 0)} shots, backend {backend}, job {job_id}; wall_clock_time_seconds is total script/job time from output file; qpu_execution_time_seconds is dashboard-reported hardware execution time; success_rate column stores action agreement for this row.",
        }
    ]


def build_rows() -> list[dict[str, object]]:
    rows = []
    rows.extend(load_matched_training_rows())
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "agent",
        "category",
        "runs",
        "environment_steps_completed",
        "wall_clock_time_seconds",
        "qpu_execution_time_seconds",
        "SPS",
        "circuit_evaluations",
        "final_reward",
        "success_rate",
        "status",
        "notes",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bar_plot(rows: list[dict[str, object]], metric: str, output_path: Path, title: str, ylabel: str, log_scale: bool = False) -> None:
    labels = [str(r["agent"]).replace(" short-trained", "") for r in rows]
    values = [as_float(r.get(metric)) or 0 for r in rows]
    colors = [COLORS.get(str(r["agent"]), "#666666") for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("symlog", linthresh=1)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=18)

    for bar, value in zip(bars, values):
        label = format_number(value, digits=1) if value else "0"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_plots(rows: list[dict[str, object]]) -> None:
    write_bar_plot(rows, "SPS", SPS_PLOT, "LunarLander Compute Throughput", "Steps per second")
    write_bar_plot(rows, "wall_clock_time_seconds", WALL_CLOCK_PLOT, "LunarLander Wall-Clock Cost", "Seconds")
    write_bar_plot(
        rows,
        "circuit_evaluations",
        CIRCUIT_PLOT,
        "Quantum Circuit Evaluation Cost",
        "Circuit evaluations / shots",
        log_scale=True,
    )


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "Agent",
        "Category",
        "Steps",
        "Wall-clock (s)",
        "QPU exec. (s)",
        "SPS",
        "Circuit evals",
        "Final reward",
        "Success rate",
        "Status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| {agent} | {category} | {steps} | {wall} | {qpu} | {sps} | {circuits} | {reward} | {success} | {status} |".format(
                agent=row["agent"],
                category=row["category"],
                steps=format_number(row.get("environment_steps_completed"), digits=0),
                wall=format_number(row.get("wall_clock_time_seconds"), digits=1),
                qpu=format_number(row.get("qpu_execution_time_seconds"), digits=1),
                sps=format_number(row.get("SPS"), digits=1),
                circuits=format_number(row.get("circuit_evaluations"), digits=0),
                reward=format_number(row.get("final_reward"), digits=3),
                success=format_number(row.get("success_rate"), digits=3),
                status=row["status"],
            )
        )
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, object]]) -> None:
    paragraph = (
        "The compute-cost results use the completed matched 100,000-step "
        "LunarLander cohort. This keeps environment interactions, seed protocol, "
        "and evaluation cadence aligned across PPO, PPO-tiny, DQN, Quantum DQN, "
        "and QRL. The quantum agents complete the same nominal training budget, "
        "but their wall-clock cost is much higher because simulated circuit "
        "evaluation dominates throughput."
    )
    ibm_paragraph = (
        "The IBM hardware run separates end-to-end script/job wall-clock time "
        "from actual QPU execution time. The output file reports about 1,618 "
        "seconds for the job/script path, while the IBM dashboard showed about "
        "2 seconds of QPU execution on `ibm_kingston`. This means the QPU can "
        "execute the small inference circuit quickly once the job reaches "
        "hardware, but practical usability is still shaped by queue time, "
        "runtime overhead, shot noise, hardware noise, limited free QPU access, "
        "and the fact that training was still performed in simulation. The "
        "2-second QPU runtime should not be treated as full training time or as "
        "evidence of quantum advantage."
    )
    specs_table = "\n".join(
        ["| Hardware detail | Value |", "| --- | --- |"]
        + [f"| {key} | {value} |" for key, value in IBM_HARDWARE_SPECS.items()]
    )

    content = f"""# LunarLander Compute-Cost Comparison

{markdown_table(rows)}

## Plots

![SPS comparison]({SPS_PLOT.as_posix()})

**Caption.** Steps per second for completed matched-budget LunarLander runs.

![Wall-clock comparison]({WALL_CLOCK_PLOT.as_posix()})

**Caption.** Wall-clock seconds for each completed matched-budget LunarLander run category.

![Circuit evaluation comparison]({CIRCUIT_PLOT.as_posix()})

**Caption.** Circuit evaluation cost. Classical models have zero circuit evaluations; quantum rows report simulated circuit evaluations during matched-budget training.

## Report-Ready Interpretation

{paragraph}

## IBM QPU Hardware Details

{specs_table}

{ibm_paragraph}

## Notes

- Rows come from `lunarlander_aggregate_results.csv` with `included_in_plots == yes`.
- The IBM inference artifact remains separate because it is inference-only, not a matched training run.
"""
    OUTPUT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_plots(rows)
    write_markdown(rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {SPS_PLOT}")
    print(f"Wrote {WALL_CLOCK_PLOT}")
    print(f"Wrote {CIRCUIT_PLOT}")


if __name__ == "__main__":
    main()
