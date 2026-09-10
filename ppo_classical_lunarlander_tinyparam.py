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
from cleanqrl.experiment import (
    diagnostic_metrics,
    episode_metrics,
    evaluate_greedy_policy,
    log_metrics,
    make_env,
    maybe_save_checkpoint,
    standardize_lunarlander_config,
)

# --- CONFIGURATION ---
CONFIG = {
    "env_id": "LunarLander-v3",
    "total_timesteps": 2000000,
    "learning_rate": 0.0005,
    "num_envs": 4,
    "num_steps": 1000,
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
    "batch_size": 4000,
    "minibatch_size": 125,
    "trial_name": "ppo_tiny_classical",
    "agent": "PPO_tiny_classical",
    "wandb": False,
    "save_model": True,
    "standardize_lunarlander": True,
    "observation_preprocessing": "none",
    "eval_interval": 100000,
    "checkpoint_interval": 100000,
    "eval_episodes": 10,
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
    parser.add_argument("--path", default=None, help="Optional explicit log directory.")
    parser.add_argument("--trial-name", default=None, help="Optional explicit trial name.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional generated YAML configuration to load before CLI overrides.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Optional training budget override.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Optional evaluation interval override.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Optional checkpoint interval override.",
    )
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as config_file:
            CONFIG.update(yaml.safe_load(config_file) or {})
    CONFIG["seed"] = args.seed
    if args.total_timesteps is not None:
        CONFIG["training_budget_timesteps"] = args.total_timesteps
        CONFIG["total_timesteps"] = args.total_timesteps
    if args.eval_interval is not None:
        CONFIG["eval_interval"] = args.eval_interval
    if args.checkpoint_interval is not None:
        CONFIG["checkpoint_interval"] = args.checkpoint_interval
    standardize_lunarlander_config(CONFIG)

    # --- PATH SETUP (MATCHING main.py) ---
    repo_root = os.path.dirname(os.path.abspath(__file__))
    logs_root = os.path.join(repo_root, "logs")
    
    if args.path is not None:
        log_path = os.path.abspath(args.path)
        trial_name = args.trial_name or os.path.basename(log_path)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        trial_name = args.trial_name or f"{timestamp}_{CONFIG['trial_name']}_seed{args.seed}"
        log_path = os.path.join(logs_root, trial_name)
    CONFIG["path"] = log_path
    CONFIG["trial_name"] = trial_name
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

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = gym.vector.SyncVectorEnv(
        [make_env(CONFIG["env_id"], CONFIG) for _ in range(CONFIG["num_envs"])]
    )
    
    obs_shape = envs.single_observation_space.shape
    assert obs_shape is not None, "Env shape is None"
    
    # Agent Setup
    device = torch.device("cpu")
    agent = TinyClassicalAgent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG["learning_rate"], eps=1e-5)

    print(f"Starting Tiny Classical Training (Seed {args.seed}) | ~840 params")
    start_time = time.time()
    
    # Initialize storage
    obs = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"]) + obs_shape).to(device) # type: ignore
    actions = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"]) + envs.single_action_space.shape).to(device) # type: ignore
    logprobs = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)
    rewards = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)
    dones = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)
    values = torch.zeros((CONFIG["num_steps"], CONFIG["num_envs"])).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset(seed=args.seed)[0]).to(device)
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
                        metrics = episode_metrics(CONFIG, ep_r, ep_l, global_step)
                        metrics["charts/episodic_return"] = float(ep_r)
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
        log_metrics(
            CONFIG,
            diagnostic_metrics(
                CONFIG,
                global_step,
                start_time,
                learning_rate=optimizer.param_groups[0]["lr"],
                value_loss=v_loss.item(),
                policy_loss=pg_loss.item(),
                entropy=entropy_loss.item(),
                approx_kl=approx_kl.item(),
                SPS=int(global_step / (time.time() - start_time)),
            ),
            log_path,
        )
        evaluate_greedy_policy(
            CONFIG,
            lambda eval_obs: torch.argmax(agent.actor(eval_obs), dim=1).cpu().numpy(),
            device,
            global_step,
            log_path,
        )
        maybe_save_checkpoint(CONFIG, agent, log_path, trial_name, global_step)

    # Save
    torch.save(agent.state_dict(), f"{log_path}/tiny_model.pth")
    envs.close()
    writer.close()
    print(f"Tiny Classical Run Complete. Results saved to {log_path}")
