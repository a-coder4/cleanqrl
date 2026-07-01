"""Generate qualitative LunarLander rollout videos and screenshots.

Run this script from an environment that has the CleanQRL runtime installed
(`gymnasium[box2d]`, `pygame`, `torch`, `imageio`, and optionally `pennylane`
for QPPO/QRL).

The outputs are qualitative examples only. They should not be used as proof of
performance; use reward curves, final reward tables, success rates, and compute
cost comparisons for quantitative claims.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


REQUIRED = ["gymnasium", "pygame", "torch", "imageio"]
OPTIONAL_QPPO = ["pennylane"]
ACTION_NAMES = ["noop", "fire_left", "fire_main", "fire_right"]
SUCCESS_REWARD_THRESHOLD = 200.0
DEFAULT_EPISODES = 3
OUT_DIR = Path("lunarlander_simulation_visuals")
INDEX_PATH = OUT_DIR / "lunarlander_simulation_visuals_index.md"


@dataclass
class ModelSpec:
    name: str
    slug: str
    kind: str
    checkpoint: Path
    include: bool = True


def dependency_status() -> dict[str, bool]:
    modules = REQUIRED + OPTIONAL_QPPO + ["pygame", "imageio_ffmpeg"]
    return {module: importlib.util.find_spec(module) is not None for module in modules}


def require_dependencies(include_qppo: bool) -> None:
    status = dependency_status()
    missing = [module for module in REQUIRED if not status[module]]
    if include_qppo:
        missing.extend(module for module in OPTIONAL_QPPO if not status[module])
    if missing:
        details = ", ".join(missing)
        raise SystemExit(
            "Cannot generate LunarLander visuals because this Python environment "
            f"is missing: {details}\n\n"
            "Activate the CleanQRL environment first, then rerun:\n"
            "  python generate_lunarlander_simulation_visuals.py --episodes 3\n\n"
            "If LunarLander rendering is missing, install the project extras or:\n"
            "  python -m pip install \"gymnasium[box2d]\" pygame imageio imageio-ffmpeg"
        )


def latest_existing(paths: list[str]) -> Path | None:
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def default_model_specs(include_qppo: bool) -> list[ModelSpec]:
    ppo = latest_existing(
        [
            "logs/2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0/2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0_step2000000.cleanqrl_model",
            "logs/2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0/2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0.cleanqrl_model",
            "logs/2026-01-24--13-32-47_ppo_lunarlander_stable_B/2026-01-24--13-32-47_ppo_lunarlander_stable_B.cleanqrl_model",
        ]
    )
    dqn = latest_existing(
        [
            "logs/2026-06-29--15-26-31_dqn_lunarlander_classical_seed0/2026-06-29--15-26-31_dqn_lunarlander_classical_seed0_step2000000.cleanqrl_model",
            "logs/2026-06-29--15-26-31_dqn_lunarlander_classical_seed0/2026-06-29--15-26-31_dqn_lunarlander_classical_seed0.cleanqrl_model",
            "logs/2026-01-22--02-39-44_dqn_lunarlander_classical/2026-01-22--02-39-44_dqn_lunarlander_classical.cleanqrl_model",
        ]
    )
    qppo = latest_existing(
        [
            "logs/2026-07-01--00-55-29_qppo_short_trained_lunarlander_seed0/2026-07-01--00-55-29_qppo_short_trained_lunarlander_seed0_step25000.cleanqrl_model",
            "logs/2026-07-01--00-55-29_qppo_short_trained_lunarlander_seed0/2026-07-01--00-55-29_qppo_short_trained_lunarlander_seed0.cleanqrl_model",
        ]
    )

    specs = []
    if ppo:
        specs.append(ModelSpec("PPO classical", "ppo_classical", "ppo", ppo))
    if dqn:
        specs.append(ModelSpec("DQN classical", "dqn_classical", "dqn", dqn))
    if include_qppo and qppo:
        specs.append(ModelSpec("QPPO/QRL short-trained", "qppo_qrl_short_trained", "qppo", qppo))
    return specs


def build_ppo_agent(torch):
    import torch.nn as nn

    class PPOAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.critic = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
            self.actor = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

        def act(self, obs):
            with torch.no_grad():
                logits = self.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                return int(torch.argmax(logits, dim=1).item())

    return PPOAgent()


def build_dqn_agent(torch):
    import torch.nn as nn

    class DQNAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(8, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 4))

        def act(self, obs):
            with torch.no_grad():
                q_values = self.network(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                return int(torch.argmax(q_values, dim=1).item())

    return DQNAgent()


def build_qppo_agent(torch):
    import pennylane as qml
    import torch.nn as nn

    n_qubits = 4
    n_layers = 2
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

    class QuantumLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(n_layers, n_qubits, 3), requires_grad=True)

        def forward(self, x):
            results = []
            for i in range(x.shape[0]):
                results.append(torch.stack(quantum_circuit(x[i], self.weights)))
            return torch.stack(results)

    class QPPOAgent(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 4), nn.Tanh())
            self.actor_scale = nn.Parameter(torch.ones(1), requires_grad=True)
            self.quantum_layer = QuantumLayer()
            self.critic = nn.Sequential(nn.Linear(8, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

        def act(self, obs):
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                features = self.network(x) * np.pi
                logits = self.quantum_layer(features) * (1.0 + self.actor_scale)
                return int(torch.argmax(logits, dim=1).item())

    return QPPOAgent()


def load_agent(spec: ModelSpec):
    import torch

    builders = {
        "ppo": build_ppo_agent,
        "dqn": build_dqn_agent,
        "qppo": build_qppo_agent,
    }
    agent = builders[spec.kind](torch)
    state = torch.load(spec.checkpoint, map_location="cpu")
    agent.load_state_dict(state)
    agent.eval()
    return agent


def make_env():
    import gymnasium as gym

    return gym.make("LunarLander-v3", render_mode="rgb_array")


def save_video(frames: list[np.ndarray], path_base: Path, fps: int) -> Path:
    import imageio.v2 as imageio

    mp4_path = path_base.with_suffix(".mp4")
    gif_path = path_base.with_suffix(".gif")
    try:
        writer = imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return mp4_path
    except Exception:
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path


def save_png(frame: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio

    imageio.imwrite(path, frame)


def classify_episode(total_reward: float, terminated: bool, truncated: bool) -> str:
    if total_reward >= SUCCESS_REWARD_THRESHOLD:
        return "success"
    if terminated:
        return "failure_or_crash"
    if truncated:
        return "truncated_final_state"
    return "final_state"


def pick_keyframes(frames: list[np.ndarray], observations: list[np.ndarray], status: str) -> list[tuple[str, int]]:
    if not frames:
        return []
    n = len(frames)
    picks = [("start", 0), ("descent", max(0, n // 4)), ("near_landing", max(0, int(n * 0.75)))]
    contact_index = None
    for i, obs in enumerate(observations):
        if len(obs) >= 8 and obs[6] > 0 and obs[7] > 0:
            contact_index = min(i, n - 1)
            break
    if contact_index is not None:
        picks.append(("touchdown_or_ground_contact", contact_index))
    final_label = "touchdown_final" if status == "success" else ("crash_or_failure_final" if status == "failure_or_crash" else "final_state")
    picks.append((final_label, n - 1))

    unique = []
    seen = set()
    for label, idx in picks:
        key = (label, idx)
        if key not in seen:
            unique.append((label, idx))
            seen.add(key)
    return unique


def run_episode(spec: ModelSpec, agent, episode_idx: int, seed: int, fps: int, max_steps: int) -> dict[str, object]:
    env = make_env()
    obs, _ = env.reset(seed=seed)
    frames = []
    observations = []
    actions = []
    total_reward = 0.0
    terminated = False
    truncated = False

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)
        observations.append(np.asarray(obs).copy())
        action = agent.act(obs)
        actions.append(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            frames.append(env.render())
            observations.append(np.asarray(obs).copy())
            break
    env.close()

    model_dir = OUT_DIR / spec.slug
    model_dir.mkdir(parents=True, exist_ok=True)
    status = classify_episode(total_reward, terminated, truncated)
    base = model_dir / f"episode_{episode_idx:02d}"
    video_path = save_video(frames, base, fps=fps)
    screenshot_paths = []
    for label, frame_idx in pick_keyframes(frames, observations, status):
        screenshot_path = model_dir / f"episode_{episode_idx:02d}_{label}.png"
        save_png(frames[frame_idx], screenshot_path)
        screenshot_paths.append(screenshot_path)

    return {
        "model": spec.name,
        "slug": spec.slug,
        "checkpoint": str(spec.checkpoint),
        "episode": episode_idx,
        "seed": seed,
        "return": total_reward,
        "steps": len(actions),
        "status": status,
        "video": str(video_path),
        "screenshots": [str(path) for path in screenshot_paths],
        "caption": (
            f"Qualitative deterministic rollout for {spec.name}, episode {episode_idx}, "
            f"return {total_reward:.3f}, status {status}. This visual is illustrative only; "
            "quantitative conclusions should use reward curves, final reward tables, success rates, and compute-cost comparisons."
        ),
    }


def write_index(records: list[dict[str, object]], skipped: list[str]) -> None:
    lines = [
        "# LunarLander Simulation Visuals Index",
        "",
        "These visuals are qualitative rollout examples only. They are not proof of performance; the quantitative evidence remains the reward curves, final reward tables, success rates, and compute-cost comparisons.",
        "",
    ]
    if skipped:
        lines.extend(["## Skipped Models", ""])
        lines.extend(f"- {item}" for item in skipped)
        lines.append("")

    for record in records:
        lines.extend(
            [
                f"## {record['model']} - Episode {record['episode']}",
                "",
                f"- Model: {record['model']}",
                f"- Checkpoint: `{record['checkpoint']}`",
                f"- Episode return: {float(record['return']):.3f}",
                f"- Episode steps: {record['steps']}",
                f"- Success/failure: {record['status']}",
                f"- Video: `{Path(str(record['video'])).as_posix()}`",
                "- Screenshots:",
            ]
        )
        lines.extend(f"  - `{Path(path).as_posix()}`" for path in record["screenshots"])
        lines.extend(["", f"Caption: {record['caption']}", ""])

    INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(records: list[dict[str, object]]) -> None:
    (OUT_DIR / "rollout_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative LunarLander rollout visuals.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Deterministic evaluation episodes per model.")
    parser.add_argument("--seed", type=int, default=9000, help="Base seed for rollout episodes.")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--skip-qppo", action="store_true", help="Skip QPPO/QRL rollout generation.")
    parser.add_argument("--clean", action="store_true", help="Remove existing visuals before regenerating.")
    args = parser.parse_args()

    include_qppo = not args.skip_qppo
    require_dependencies(include_qppo)

    if args.clean and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(exist_ok=True)

    specs = default_model_specs(include_qppo)
    skipped = []
    if include_qppo and not any(spec.kind == "qppo" for spec in specs):
        skipped.append("QPPO/QRL: no usable short-trained checkpoint found.")
    if not specs:
        raise SystemExit("No usable LunarLander checkpoints were found.")

    records = []
    for spec in specs:
        try:
            agent = load_agent(spec)
        except Exception as exc:
            skipped.append(f"{spec.name}: checkpoint could not be loaded from {spec.checkpoint}: {exc}")
            continue
        for episode_idx in range(args.episodes):
            record = run_episode(spec, agent, episode_idx, args.seed + episode_idx, args.fps, args.max_steps)
            records.append(record)
            print(f"{spec.name} episode {episode_idx}: return={record['return']:.3f} status={record['status']}")

    write_index(records, skipped)
    write_manifest(records)
    print(f"Wrote visuals to {OUT_DIR}")
    print(f"Wrote index to {INDEX_PATH}")


if __name__ == "__main__":
    main()
