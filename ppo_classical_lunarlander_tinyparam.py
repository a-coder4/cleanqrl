import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time
import random
import os
import json
import argparse
import yaml
from datetime import datetime
from collections import deque

# --- CONFIGURATION ---
CONFIG = {
    "env_id": "LunarLander-v3",
    "total_timesteps": 2000000,
    "learning_rate": 0.0005,
    "num_envs": 4,
    "num_steps": 1024,
    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 32,
    "update_epochs": 10,
    "norm_adv": True,
    "clip_coef": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "batch_size": 4096,
    "minibatch_size": 128,
    "trial_name": "ppo_tiny_classical" # Added for config saving
}

# --- TINY AGENT (~840 Params) ---
class TinyClassicalAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # Retrieve shape and force type check to pass
        obs_shape = envs.single_observation_space.shape
        action_space = envs.single_action_space
        
        # Safe calculations for dimensions
        assert obs_shape is not None, "Observation space cannot be None"
        obs_dim = np.prod(obs_shape)
        action_dim = getattr(action_space, "n", 4) 
        
        self.hidden_size = 32

        self.actor = nn.Sequential(
            nn.Linear(int(obs_dim), self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, int(action_dim)),
        )

        self.critic = nn.Sequential(
            nn.Linear(int(obs_dim), self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="seed of the experiment")
    args = parser.parse_args()

    # --- PATH SETUP (MATCHING main.py) ---
    repo_root = os.path.dirname(os.path.abspath(__file__))
    logs_root = os.path.join(repo_root, "logs")
    
    # Create unique timestamped name
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    trial_name = f"{timestamp}_{CONFIG['trial_name']}_seed{args.seed}"
    
    # Final path: ./logs/2026-02-12--..._ppo_tiny_classical_seed1
    log_path = os.path.join(logs_root, trial_name)
    os.makedirs(log_path, exist_ok=True)
    
    print(f"Logging to: {log_path}")

    # Save Config (Crucial for plotting scripts to recognize this folder)
    with open(os.path.join(log_path, "config.yaml"), "w") as f:
        yaml.dump(CONFIG, f)
    
    writer = SummaryWriter(log_path)
    # Prepare result.json for logging metrics (one JSON object per line)
    json_file_path = os.path.join(log_path, "result.json")
    # create an empty file (consistent with other scripts)
    with open(json_file_path, "w") as _:
        pass
    
    # Logging helpers and counters
    print_interval = 10
    episode_returns = deque(maxlen=print_interval)
    global_episodes = 0

    def log_metrics(config, metrics, report_path=None):
        # Minimal compatibility with other scripts' behavior
        try:
            import wandb
        except Exception:
            wandb = None

        try:
            import ray
        except Exception:
            ray = None

        if isinstance(config, dict) and config.get("wandb", False) and wandb is not None:
            try:
                wandb.log(metrics)
            except Exception:
                pass

        if ray is not None and ray.is_initialized():
            try:
                ray.train.report(metrics=metrics)
            except Exception:
                pass
        else:
            if report_path is not None:
                with open(os.path.join(str(report_path), "result.json"), "a") as f:
                    json.dump(metrics, f)
                    f.write("\n")
            else:
                # fallback: append to json_file_path if provided
                with open(json_file_path, "a") as f:
                    json.dump(metrics, f)
                    f.write("\n")
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Env Setup: use a factory that adds RecordEpisodeStatistics so episode info is available
    def make_env_fn(env_id):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk

    envs = gym.vector.SyncVectorEnv([make_env_fn(CONFIG["env_id"]) for _ in range(CONFIG["num_envs"])])
    
    obs_shape = envs.single_observation_space.shape
    assert obs_shape is not None, "Env shape is None"
    
    # Agent Setup
    device = torch.device("cpu")
    agent = TinyClassicalAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG["learning_rate"], eps=1e-5)

    print(f"Starting Tiny Classical Training (Seed {args.seed}) | ~840 params")
    
    # Initialize storage
    obs = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"]) + obs_shape).to(device) # type: ignore
    actions = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"]) + envs.single_action_space.shape).to(device) # type: ignore
    logprobs = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)
    rewards = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)
    dones = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)
    values = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(CONFIG["num_envs"]).to(device)
    num_updates = CONFIG["total_timesteps"] // CONFIG["batch_size"]

    # Open result.json in append mode
    json_file_path = os.path.join(log_path, "result.json")

    for update in range(1, num_updates + 1):
        if CONFIG["anneal_lr"]:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * CONFIG["learning_rate"]

        for step in range(CONFIG["num_steps"]):
            global_step += 1 * CONFIG["num_envs"]
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Episode reporting: handle vectorized env infos (RecordEpisodeStatistics)
            if isinstance(infos, dict) and "_episode" in infos:
                for idx, finished in enumerate(infos["_episode"]):
                    if finished:
                        global_episodes += 1
                        ep_r = infos["episode"]["r"].tolist()[idx]
                        ep_l = infos["episode"]["l"].tolist()[idx]
                        episode_returns.append(ep_r)

                        # TensorBoard
                        writer.add_scalar("charts/episodic_return", float(ep_r), global_step)

                        # JSON logging (one JSON object per line)
                        metrics = {
                            "global_step": int(global_step),
                            "episode_reward": float(ep_r),
                            "charts/episodic_return": float(ep_r),
                            "episode_length": int(ep_l),
                        }
                        log_metrics(CONFIG, metrics, log_path)

                # print progress periodically when not running under Ray
                try:
                    import ray as _ray
                    ray_inited = _ray.is_initialized()
                except Exception:
                    ray_inited = False

                if global_episodes % print_interval == 0 and not ray_inited:
                    print("Global step:", global_step, " Mean return:", np.mean(episode_returns))

        # Advantage Calculation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(CONFIG["num_steps"])):
                if t == CONFIG["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + CONFIG["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + CONFIG["gamma"] * CONFIG["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + obs_shape) # type: ignore
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape) # type: ignore
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize
        b_inds = np.arange(CONFIG["batch_size"])
        for epoch in range(CONFIG["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, CONFIG["batch_size"], CONFIG["minibatch_size"]):
                end = start + CONFIG["minibatch_size"]
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if CONFIG["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CONFIG["clip_coef"], 1 + CONFIG["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - CONFIG["ent_coef"] * entropy_loss + CONFIG["vf_coef"] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), CONFIG["max_grad_norm"])
                optimizer.step()
            
            if CONFIG["target_kl"] is not None and approx_kl > CONFIG["target_kl"]:
                break

    # Save
    torch.save(agent.state_dict(), f"{log_path}/tiny_model.pth")
    envs.close()
    writer.close()
    print(f"Tiny Classical Run Complete. Results saved to {log_path}")