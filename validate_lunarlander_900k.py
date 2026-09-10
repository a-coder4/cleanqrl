"""Strict validator for the dedicated matched LunarLander-v3 900k cohort."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from lunarlander_900k_common import (
    AGENT_ORDER,
    AGENT_SPECS,
    CHECKPOINT_INTERVAL,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    EXPECTED_STEPS,
    SEEDS,
    SUCCESS_THRESHOLD,
    TOTAL_TIMESTEPS,
    CohortValidation,
    display_combo,
    validate_cohort,
)


DEFAULT_MARKDOWN = Path("lunarlander_900k_validation_report.md")
DEFAULT_CSV = Path("lunarlander_900k_validation_report.csv")


def _cell(value: object) -> str:
    return "" if value is None else str(value)


def write_csv_report(cohort: CohortValidation, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "run_path",
        "agent",
        "agent_key",
        "seed",
        "configured_timesteps",
        "max_timestep",
        "evaluation_steps",
        "checkpoint_steps",
        "train_rows",
        "evaluation_rows",
        "diagnostic_rows",
        "valid",
        "selected",
        "status",
        "failure_reasons",
        "warnings",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in cohort.candidates:
            status = "SELECTED" if candidate.selected else ("VALID_NOT_SELECTED" if candidate.valid else "FAIL")
            writer.writerow(
                {
                    "run_name": candidate.run_name,
                    "run_path": str(candidate.path),
                    "agent": AGENT_SPECS.get(candidate.friendly_agent, {}).get("label", ""),
                    "agent_key": candidate.agent_key,
                    "seed": _cell(candidate.seed),
                    "configured_timesteps": _cell(candidate.configured_timesteps),
                    "max_timestep": _cell(candidate.max_timestep),
                    "evaluation_steps": ";".join(map(str, candidate.evaluation_steps)),
                    "checkpoint_steps": ";".join(map(str, candidate.checkpoint_steps)),
                    "train_rows": candidate.train_rows,
                    "evaluation_rows": candidate.evaluation_rows,
                    "diagnostic_rows": candidate.diagnostic_rows,
                    "valid": "yes" if candidate.valid else "no",
                    "selected": "yes" if candidate.selected else "no",
                    "status": status,
                    "failure_reasons": candidate.reason if not candidate.valid else "",
                    "warnings": "; ".join(candidate.warnings),
                }
            )


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_markdown_report(cohort: CohortValidation, output_path: Path, logs_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows: list[list[object]] = []
    for friendly in AGENT_ORDER:
        for seed in SEEDS:
            candidate = cohort.selected.get((friendly, seed))
            selected_rows.append(
                [
                    AGENT_SPECS[friendly]["label"],
                    seed,
                    "PASS" if candidate else "MISSING",
                    f"`{candidate.run_name}`" if candidate else "",
                ]
            )

    inventory_rows = []
    for candidate in cohort.candidates:
        status = "SELECTED" if candidate.selected else ("VALID, NOT SELECTED" if candidate.valid else "FAIL")
        reason = "; ".join(candidate.warnings) if candidate.valid else candidate.reason
        inventory_rows.append(
            [
                f"`{candidate.run_name}`",
                AGENT_SPECS.get(candidate.friendly_agent, {}).get("label", candidate.agent_key),
                _cell(candidate.seed),
                _cell(candidate.configured_timesteps),
                _cell(candidate.max_timestep),
                status,
                reason,
            ]
        )

    lines = [
        "# Matched LunarLander-v3 900k Validation Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Logs scanned: `{logs_dir.resolve()}`",
        "",
        "## Overall status",
        "",
        f"**{'PASS' if cohort.passed else 'FAIL'}** — {len(cohort.selected)}/12 required runs selected.",
        "",
        "The experiment is complete only when this report says PASS and all twelve agent/seed rows below pass.",
        "",
        "## Required cohort",
        "",
        _markdown_table(["Agent", "Seed", "Status", "Selected run"], selected_rows),
        "",
        "## Missing runs",
        "",
    ]
    if cohort.missing:
        lines.extend(f"- {display_combo(combo)}" for combo in cohort.missing)
    else:
        lines.append("None.")

    lines.extend(["", "## Duplicate resolution", ""])
    lines.append(
        "Selection rule: for each agent/seed, select the newest completely valid dedicated 900k run, "
        "using the leading run timestamp and then the run name as a deterministic tie-breaker. "
        "A valid older duplicate remains reported but is not selected."
    )
    lines.append("")
    if cohort.ambiguities:
        for combo, candidates in cohort.ambiguities.items():
            lines.append(
                f"- {display_combo(combo)}: selected `{candidates[0].run_name}` from "
                + ", ".join(f"`{candidate.run_name}`" for candidate in candidates)
                + "."
            )
    else:
        lines.append("No agent/seed has multiple completely valid candidates.")

    lines.extend(
        [
            "",
            "## Candidate inventory",
            "",
            _markdown_table(
                ["Run", "Agent", "Seed", "Configured", "Max logged", "Status", "Reason"],
                inventory_rows,
            )
            if inventory_rows
            else "No LunarLander candidates with `config.yaml` were found.",
            "",
            "## Protocol enforced",
            "",
            f"- Environment: `LunarLander-v3`; native discrete actions; no action remapping.",
            f"- Configured and completed interactions: exactly {TOTAL_TIMESTEPS:,}.",
            f"- Agents: {', '.join(AGENT_SPECS[name]['label'] for name in AGENT_ORDER)}.",
            f"- Seeds: {list(SEEDS)} for every agent.",
            "- Observation preprocessing: `none`.",
            f"- Deterministic greedy evaluation with exploration disabled: {EVAL_EPISODES} episodes at "
            + ", ".join(f"{step // 1000}k" for step in EXPECTED_STEPS)
            + ".",
            f"- Checkpoints every {CHECKPOINT_INTERVAL:,} interactions through {TOTAL_TIMESTEPS:,}.",
            f"- Success threshold: episode reward >= {SUCCESS_THRESHOLD:g}.",
            "- Standard train/evaluation metric schemas, final positive wall-clock/SPS diagnostics, and QRL circuit accounting.",
            "- Runs configured for any horizon other than 900k are excluded, even if they contain a 900k checkpoint.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write reports but return success while the required cohort is incomplete.",
    )
    args = parser.parse_args()

    cohort = validate_cohort(args.logs_dir)
    write_markdown_report(cohort, args.markdown, args.logs_dir)
    write_csv_report(cohort, args.csv)
    print(f"Wrote {args.markdown}")
    print(f"Wrote {args.csv}")
    print(f"Selected {len(cohort.selected)}/12 required runs")
    if cohort.missing:
        print("Missing: " + ", ".join(display_combo(combo) for combo in cohort.missing))
    if cohort.ambiguities:
        print(f"Ambiguous valid combinations resolved deterministically: {len(cohort.ambiguities)}")
    if not cohort.passed and not args.allow_incomplete:
        print("VALIDATION FAILED: the matched 900k experiment is incomplete.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
