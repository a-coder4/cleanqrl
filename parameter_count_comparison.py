"""Generate LunarLander agent parameter-count comparison artifacts.

The counts here are derived from the model definitions used by the project:

- cleanqrl/ppo_classical.py::PPOAgentClassical
- ppo_classical_lunarlander_tinyparam.py::TinyClassicalAgent
- cleanqrl/dqn_classical.py::DQNAgentClassical
- cleanqrl/ppo_quantum_hybrid.py::Agent / QuantumLayer

For DQN, the trainable total counts the optimized online Q-network. The target
network has the same shape but is a synchronization copy, not a separately
optimized model.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBS_DIM = 8
ACTION_DIM = 4

CSV_PATH = Path("parameter_count_comparison.csv")
MD_PATH = Path("parameter_count_comparison.md")
PLOT_PATH = Path("parameter_count_total_trainable_params.png")


def linear_params(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def build_rows() -> list[dict[str, object]]:
    ppo_actor = (
        linear_params(OBS_DIM, 64)
        + linear_params(64, 64)
        + linear_params(64, ACTION_DIM)
    )
    ppo_critic = (
        linear_params(OBS_DIM, 64)
        + linear_params(64, 64)
        + linear_params(64, 1)
    )

    ppo_tiny_actor = linear_params(OBS_DIM, 32) + linear_params(32, ACTION_DIM)
    ppo_tiny_critic = linear_params(OBS_DIM, 32) + linear_params(32, 1)

    dqn_q_network = (
        linear_params(OBS_DIM, 120)
        + linear_params(120, 84)
        + linear_params(84, ACTION_DIM)
    )

    qppo_encoder = linear_params(OBS_DIM, 64) + linear_params(64, 4)
    qppo_quantum_weights = 2 * 4 * 3
    qppo_actor_scale = 1
    qppo_actor = qppo_encoder + qppo_quantum_weights + qppo_actor_scale
    qppo_critic = (
        linear_params(OBS_DIM, 64)
        + linear_params(64, 64)
        + linear_params(64, 1)
    )

    rows = [
        {
            "agent": "PPO",
            "policy_actor_params": ppo_actor,
            "critic_value_params": ppo_critic,
            "q_network_params": "",
            "quantum_circuit_params": 0,
            "total_trainable_params": ppo_actor + ppo_critic,
            "source": "cleanqrl/ppo_classical.py::PPOAgentClassical",
            "notes": "Actor and critic are separate 8-64-64-output MLPs.",
        },
        {
            "agent": "PPO-tiny",
            "policy_actor_params": ppo_tiny_actor,
            "critic_value_params": ppo_tiny_critic,
            "q_network_params": "",
            "quantum_circuit_params": 0,
            "total_trainable_params": ppo_tiny_actor + ppo_tiny_critic,
            "source": "ppo_classical_lunarlander_tinyparam.py::TinyClassicalAgent",
            "notes": "Tiny actor and critic each use one hidden layer with 32 units.",
        },
        {
            "agent": "DQN",
            "policy_actor_params": dqn_q_network,
            "critic_value_params": 0,
            "q_network_params": dqn_q_network,
            "quantum_circuit_params": 0,
            "total_trainable_params": dqn_q_network,
            "source": "cleanqrl/dqn_classical.py::DQNAgentClassical",
            "notes": "Counts the optimized online Q-network only; target network is a synchronized copy.",
        },
        {
            "agent": "QPPO/QRL short-trained",
            "policy_actor_params": qppo_actor,
            "critic_value_params": qppo_critic,
            "q_network_params": "",
            "quantum_circuit_params": qppo_quantum_weights,
            "total_trainable_params": qppo_actor + qppo_critic,
            "source": "cleanqrl/ppo_quantum_hybrid.py::Agent; configs/benchmarks/ppo_quantum_lunarlander_short_train.yaml",
            "notes": "Hybrid actor includes an 8-64-4 classical encoder, 24 variational circuit weights, and one actor scale parameter.",
        },
    ]
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "agent",
        "policy_actor_params",
        "critic_value_params",
        "q_network_params",
        "quantum_circuit_params",
        "total_trainable_params",
        "source",
        "notes",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "Agent",
        "Policy/actor params",
        "Critic/value params",
        "Q-network params",
        "Quantum circuit params",
        "Total trainable params",
        "Status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        status = "Short-trained feasibility model" if "QPPO" in str(row["agent"]) else "Classical baseline"
        lines.append(
            "| {agent} | {actor:,} | {critic:,} | {q_network} | {quantum:,} | {total:,} | {status} |".format(
                agent=row["agent"],
                actor=int(row["policy_actor_params"]),
                critic=int(row["critic_value_params"]),
                q_network=(
                    f"{int(row['q_network_params']):,}"
                    if row["q_network_params"] != ""
                    else ""
                ),
                quantum=int(row["quantum_circuit_params"]),
                total=int(row["total_trainable_params"]),
                status=status,
            )
        )
    return "\n".join(lines)


def write_markdown(rows: list[dict[str, object]]) -> None:
    totals = {str(row["agent"]): int(row["total_trainable_params"]) for row in rows}
    qppo = totals["QPPO/QRL short-trained"]
    ppo = totals["PPO"]
    ppo_tiny = totals["PPO-tiny"]

    paragraph = (
        "The short-trained QPPO/QRL model uses 5,662 trainable parameters, "
        f"which is {qppo / ppo:.1%} of full PPO's 9,797 parameters. In that "
        "narrow parameter-count sense, QPPO is more parameter-efficient than "
        "the full PPO baseline. However, QPPO is not smaller than PPO-tiny: "
        f"it has {qppo / ppo_tiny:.1f}x as many trainable parameters as the "
        "741-parameter PPO-tiny baseline. Only 24 QPPO parameters are quantum "
        "circuit weights; most trainable parameters are still in the classical "
        "encoder and critic. These counts should not be interpreted as evidence "
        "of better performance from fewer parameters, because the short-trained "
        "QPPO hardware result is an inference-only feasibility demonstration "
        "and the matched training results do not establish quantum advantage."
    )

    content = f"""# LunarLander Parameter Count Comparison

{markdown_table(rows)}

![Total trainable parameter comparison]({PLOT_PATH.as_posix()})

**Figure caption.** Total trainable parameter count for the LunarLander PPO, PPO-tiny, DQN, and short-trained QPPO/QRL agents. DQN counts the optimized online Q-network only; the target network is not counted as a separately trained model.

## Report-Ready Interpretation

{paragraph}

## Counting Notes

- Full PPO uses separate actor and critic MLPs with two 64-unit hidden layers.
- PPO-tiny uses separate actor and critic MLPs with one 32-unit hidden layer.
- DQN uses one optimized online Q-network with 8-120-84-4 dimensions.
- QPPO/QRL short-trained uses a classical 8-64-4 encoder, a 4-qubit 2-layer variational actor circuit with 24 trainable circuit weights, one actor scaling parameter, and a classical 8-64-64-1 critic.
"""
    MD_PATH.write_text(content, encoding="utf-8")


def write_plot(rows: list[dict[str, object]]) -> None:
    labels = [str(row["agent"]).replace(" short-trained", "") for row in rows]
    totals = [int(row["total_trainable_params"]) for row in rows]
    colors = ["#4169e1", "#00a676", "#d95f02", "#7b3294"]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, totals, color=colors)
    ax.set_title("LunarLander Total Trainable Parameters")
    ax.set_ylabel("Trainable parameters")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_plot(rows)
    write_markdown(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
