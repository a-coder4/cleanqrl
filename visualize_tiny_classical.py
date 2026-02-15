import os
import glob
import time
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo


# --- Tiny Agent (copied minimal definition) ---
class TinyClassicalAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        action_dim = int(getattr(envs.single_action_space, "n", 4))

        hidden_size = 32
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def find_latest_tiny_run(logs_root="logs"):
    patterns = os.path.join(logs_root, "*ppo_tiny_classical*")
    candidates = glob.glob(patterns)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def clear_video_folder(path):
    os.makedirs(path, exist_ok=True)
    for f in glob.glob(os.path.join(path, "*.mp4")):
        try:
            os.remove(f)
        except Exception:
            pass


def main(run_name: str = None, episodes: int = 8, deterministic: bool = True):
    # locate run
    if run_name is None:
        run_path = find_latest_tiny_run()
        if run_path is None:
            print("No ppo_tiny_classical run found in logs/")
            return
        run_name = os.path.basename(run_path)
    else:
        run_path = os.path.join("logs", run_name)

    model_path = os.path.join(run_path, "tiny_model.pth")
    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Using run: {run_name}")

    # prepare agent
    dummy_env = gym.vector.SyncVectorEnv([lambda: gym.make("LunarLander-v3")])
    device = torch.device("cpu")
    agent = TinyClassicalAgent(dummy_env).to(device)
    state = torch.load(model_path, map_location=device)
    try:
        agent.load_state_dict(state)
    except Exception as e:
        print("Failed loading state_dict:", e)
        return
    agent.eval()

    # prepare video folder
    video_folder = os.path.join("runs", run_name, "videos_tiny")
    clear_video_folder(video_folder)

    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)

    ep_to_file = {}
    print(f"Recording {episodes} episodes into {video_folder}")

    for ep in range(episodes):
        before = set(glob.glob(os.path.join(video_folder, "*.mp4")))
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if deterministic:
                    logits = agent.actor(obs_tensor)
                    action = int(torch.argmax(logits, dim=1).item())
                else:
                    action_tensor, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = int(action_tensor.item())

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

        # detect created file(s)
        after = set(glob.glob(os.path.join(video_folder, "*.mp4")))
        new = after - before
        if new:
            newest = max(new, key=os.path.getmtime)
            ep_to_file[ep] = (newest, total_reward)
            print(f"Episode {ep} reward={total_reward:.2f} -> {os.path.basename(newest)}")
        else:
            print(f"Episode {ep} reward={total_reward:.2f} -> no video file found")

    env.close()

    if not ep_to_file:
        print("No videos were produced.")
        return

    # pick best by reward
    best_ep, (best_file, best_reward) = max(ep_to_file.items(), key=lambda kv: kv[1][1])
    print("-" * 30)
    print(f"Best episode: {best_ep} reward={best_reward:.2f}")
    print(f"Video file: {best_file}")
    print("-" * 30)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None, help="run folder name under logs/")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--deterministic", action="store_true", help="use argmax instead of sampling")
    args = p.parse_args()
    main(run_name=args.run, episodes=args.episodes, deterministic=args.deterministic)
