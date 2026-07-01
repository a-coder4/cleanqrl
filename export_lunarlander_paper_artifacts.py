"""Export paper-ready LunarLander artifacts without raw logs/checkpoints."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


EXPORT_ROOT = Path("paper_export_lunarlander")

SUMMARY_FILES = [
    ("lunarlander_aggregate_summary.md", "summaries/lunarlander_aggregate_summary.md", "Aggregated LunarLander result summary for completed and excluded runs."),
    ("fairness_check_report.md", "summaries/fairness_check_report.md", "Fairness audit documenting standardized environment, budget, seeds, evaluation, and exclusions."),
    ("lunarlander_report_ready_section.md", "summaries/lunarlander_report_ready_section.md", "Report-ready discussion of completed results, matched-budget diagnostics, and quantum infeasibility."),
    ("parameter_count_comparison.csv", "summaries/parameter_count_comparison.csv", "Parameter-count table for PPO, PPO-tiny, DQN, and short-trained QPPO/QRL."),
    ("parameter_count_comparison.md", "summaries/parameter_count_comparison.md", "Report-ready parameter-count explanation and table."),
    ("compute_cost_comparison.csv", "summaries/compute_cost_comparison.csv", "Compute-cost table separating full classical training, short QPPO feasibility, and IBM inference-only results."),
    ("compute_cost_comparison.md", "summaries/compute_cost_comparison.md", "Report-ready compute-cost explanation including IBM QPU hardware details."),
    ("results/ibm_qppo_short_trained_inference_results.md", "summaries/ibm_qppo_short_trained_inference_results.md", "IBM QPU inference table for five fixed LunarLander states."),
    ("ibm_qppo_short_trained_inference_results.csv", "summaries/ibm_qppo_short_trained_inference_results.csv", "CSV version of IBM QPU inference results."),
]

FIGURE_FILES = [
    ("final_report_figures/fig_reward_curves_classical_full_and_qppo_short.png", "figures/fig_reward_curves_classical_full_and_qppo_short.png", "Main reward-curve figure: completed 2M-step classical runs separated from 25k-step QPPO feasibility."),
    ("final_report_figures/fig_final_reward_full_classical_with_qppo_feasibility.png", "figures/fig_final_reward_full_classical_with_qppo_feasibility.png", "Final reward comparison with QPPO explicitly marked as short-trained feasibility only."),
    ("final_report_figures/fig_parameter_count_total_trainable.png", "figures/fig_parameter_count_total_trainable.png", "Total trainable parameter comparison."),
    ("final_report_figures/fig_compute_sps_full_classical_vs_qppo_short.png", "figures/fig_compute_sps_full_classical_vs_qppo_short.png", "SPS comparison for full classical training versus short QPPO feasibility."),
    ("final_report_figures/fig_compute_wall_clock_by_result_category.png", "figures/fig_compute_wall_clock_by_result_category.png", "Wall-clock compute comparison by result category."),
    ("final_report_figures/fig_compute_circuit_evaluations_and_shots.png", "figures/fig_compute_circuit_evaluations_and_shots.png", "Circuit-evaluation and shot-cost comparison."),
    ("final_report_figures/fig_ibm_qppo_inference_agreement_table.png", "figures/fig_ibm_qppo_inference_agreement_table.png", "IBM hardware action-agreement figure/table."),
    ("final_report_figures/final_report_figures_index.md", "figures/final_report_figures_index.md", "Figure index with report captions and usage notes."),
]

VISUALS_ROOT = Path("lunarlander_simulation_visuals")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def copy_file(src: str, dst: str, purpose: str, exported: list[tuple[str, str]]) -> None:
    src_path = Path(src)
    if not src_path.exists():
        return
    dst_path = EXPORT_ROOT / dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)
    exported.append((dst_path.relative_to(EXPORT_ROOT).as_posix(), purpose))


def copy_visuals(exported: list[tuple[str, str]]) -> None:
    if not VISUALS_ROOT.exists():
        return
    dst_root = EXPORT_ROOT / "rollout_visuals"
    dst_root.mkdir(parents=True, exist_ok=True)
    for src_path in sorted(VISUALS_ROOT.rglob("*")):
        if src_path.is_dir():
            continue
        rel = src_path.relative_to(VISUALS_ROOT)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        if src_path.name == "lunarlander_simulation_visuals_index.md":
            purpose = "Index for qualitative LunarLander rollout videos and screenshots."
        elif src_path.suffix.lower() in {".mp4", ".gif"}:
            purpose = "Qualitative deterministic rollout video; illustrative only, not quantitative evidence."
        elif src_path.suffix.lower() == ".png":
            purpose = "Qualitative rollout still image from a key episode moment."
        elif src_path.name == "rollout_manifest.json":
            purpose = "Machine-readable rollout metadata with returns, status, checkpoint paths, and captions."
        else:
            purpose = "Qualitative rollout visual artifact."
        exported.append((dst_path.relative_to(EXPORT_ROOT).as_posix(), purpose))


def metric_lookup(rows: list[dict[str, str]], agent: str) -> dict[str, str]:
    for row in rows:
        if row.get("agent") == agent:
            return row
    return {}


def fmt(value: str, digits: int = 1) -> str:
    if value is None or value == "":
        return ""
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,}"
    return f"{numeric:,.{digits}f}"


def write_index(exported: list[tuple[str, str]]) -> None:
    compute = read_csv(EXPORT_ROOT / "summaries/compute_cost_comparison.csv")
    ppo = metric_lookup(compute, "PPO")
    dqn = metric_lookup(compute, "DQN")
    qppo = metric_lookup(compute, "QPPO/QRL short-trained")
    ibm = metric_lookup(compute, "IBM QPPO hardware inference")

    lines = [
        "# LunarLander Paper Artifact Export Index",
        "",
        "This folder is intended to be copied directly into an Overleaf project. All file paths below are relative to `paper_export_lunarlander/`.",
        "",
        "## Key Metrics To Use In The Paper",
        "",
        f"- Classical PPO completed the full {fmt(ppo.get('environment_steps_completed', ''), 0)}-step LunarLander benchmark. Final reward: {fmt(ppo.get('final_reward', ''), 3)}; success rate: {fmt(ppo.get('success_rate', ''), 3)}; wall-clock: {fmt(ppo.get('wall_clock_time_seconds', ''), 1)} s; SPS: {fmt(ppo.get('SPS', ''), 1)}.",
        f"- Classical DQN completed the full {fmt(dqn.get('environment_steps_completed', ''), 0)}-step LunarLander benchmark. Final reward: {fmt(dqn.get('final_reward', ''), 3)}; success rate: {fmt(dqn.get('success_rate', ''), 3)}; wall-clock: {fmt(dqn.get('wall_clock_time_seconds', ''), 1)} s; SPS: {fmt(dqn.get('SPS', ''), 1)}.",
        f"- Short-trained QPPO/QRL is feasibility-only, not a full 2M-step fair benchmark. It completed {fmt(qppo.get('environment_steps_completed', ''), 0)} steps, final reward {fmt(qppo.get('final_reward', ''), 3)}, success rate {fmt(qppo.get('success_rate', ''), 3)}, wall-clock {fmt(qppo.get('wall_clock_time_seconds', ''), 1)} s, SPS {fmt(qppo.get('SPS', ''), 1)}, and {fmt(qppo.get('circuit_evaluations', ''), 0)} simulated circuit evaluations.",
        f"- IBM inference was hardware-only: 5 fixed LunarLander states, 100 shots each, action agreement {fmt(ibm.get('success_rate', ''), 3)}, total job/script time about {fmt(ibm.get('wall_clock_time_seconds', ''), 1)} s, and dashboard QPU execution time about {fmt(ibm.get('qpu_execution_time_seconds', ''), 1)} s.",
        "",
        "## Exported Files",
        "",
        "| Relative path | Supports in paper |",
        "| --- | --- |",
    ]
    for rel_path, purpose in sorted(exported):
        lines.append(f"| `{rel_path}` | {purpose} |")
    lines.extend(
        [
            "",
            "## Usage Notes",
            "",
            "- The files in `figures/` are the preferred paper figures and already use captions/titles that separate full-training classical results, short-trained QPPO feasibility, and IBM inference-only results.",
            "- The files in `rollout_visuals/` are qualitative examples only and should not be used as quantitative performance evidence.",
            "- This export intentionally excludes raw checkpoints, TensorBoard logs, large aggregate row-level logs, and training scripts.",
        ]
    )
    (EXPORT_ROOT / "lunarlander_artifacts_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    EXPORT_ROOT.mkdir(exist_ok=True)
    exported: list[tuple[str, str]] = []
    for src, dst, purpose in SUMMARY_FILES + FIGURE_FILES:
        copy_file(src, dst, purpose, exported)
    copy_visuals(exported)
    write_index(exported)
    print(f"Exported {len(exported)} files to {EXPORT_ROOT}")
    print(f"Wrote {EXPORT_ROOT / 'lunarlander_artifacts_index.md'}")


if __name__ == "__main__":
    main()
