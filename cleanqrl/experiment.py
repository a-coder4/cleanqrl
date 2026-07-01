import json
import os
import time
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import ray
import torch
import wandb


LUNARLANDER_ENV_ID = "LunarLander-v3"
LUNARLANDER_TOTAL_TIMESTEPS = 2_000_000
LUNARLANDER_SEEDS = [0, 1, 2, 3, 4]
LUNARLANDER_SUCCESS_REWARD = 200.0
DEFAULT_EVAL_INTERVAL = 100_000
DEFAULT_CHECKPOINT_INTERVAL = 100_000
DEFAULT_EVAL_EPISODES = 10


class ArctanObservationWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.arctan(obs)


def standardize_lunarlander_config(config: dict) -> dict:
    if config.get("standardize_lunarlander", False):
        config["env_id"] = LUNARLANDER_ENV_ID
        config["total_timesteps"] = int(
            config.get("training_budget_timesteps", LUNARLANDER_TOTAL_TIMESTEPS)
        )
        config.setdefault("seed_set", LUNARLANDER_SEEDS)
        config.setdefault("observation_preprocessing", "none")
        config.setdefault("eval_interval", DEFAULT_EVAL_INTERVAL)
        config.setdefault("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)
        config.setdefault("eval_episodes", DEFAULT_EVAL_EPISODES)
        config.setdefault("success_reward_threshold", LUNARLANDER_SUCCESS_REWARD)
    return config


def make_env(
    env_id: str,
    config: dict,
    extra_wrapper: Optional[Callable[[gym.Env], gym.Env]] = None,
):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        preprocessing = config.get("observation_preprocessing", "none")
        if preprocessing in (None, "none"):
            pass
        elif preprocessing == "arctan":
            env = ArctanObservationWrapper(env)
        else:
            raise ValueError(f"Unknown observation_preprocessing={preprocessing!r}")

        if extra_wrapper is not None:
            env = extra_wrapper(env)
        return env

    return thunk


def log_metrics(config, metrics, report_path=None):
    if config.get("wandb", False):
        wandb.log(metrics)
    if ray.is_initialized():
        ray.train.report(metrics=metrics)
    else:
        if report_path is None:
            raise ValueError("report_path is None, cannot write result.json")
        with open(os.path.join(str(report_path), "result.json"), "a") as f:
            json.dump(metrics, f)
            f.write("\n")


def episode_metrics(config: dict, episode_reward, episode_length, global_step: int) -> dict:
    reward = float(episode_reward)
    length = int(episode_length)
    success = reward >= float(
        config.get("success_reward_threshold", LUNARLANDER_SUCCESS_REWARD)
    )
    return {
        "metric_type": "train_episode",
        "agent": config.get("agent"),
        "seed": config.get("seed"),
        "env_id": config.get("env_id"),
        "global_step": int(global_step),
        "training_timestep": int(global_step),
        "episode_reward": reward,
        "episode_length": length,
        "success_rate": float(success),
    }


def eval_metrics(
    config: dict,
    global_step: int,
    rewards: list[float],
    lengths: list[int],
) -> dict:
    threshold = float(config.get("success_reward_threshold", LUNARLANDER_SUCCESS_REWARD))
    return {
        "metric_type": "evaluation",
        "agent": config.get("agent"),
        "seed": config.get("seed"),
        "env_id": config.get("env_id"),
        "global_step": int(global_step),
        "training_timestep": int(global_step),
        "episode_reward": float(np.mean(rewards)),
        "episode_length": float(np.mean(lengths)),
        "success_rate": float(np.mean([reward >= threshold for reward in rewards])),
        "eval_episodes": len(rewards),
    }


def diagnostic_metrics(config: dict, global_step: int, start_time: float, **metrics) -> dict:
    payload = {
        "metric_type": "training_diagnostic",
        "agent": config.get("agent"),
        "seed": config.get("seed"),
        "env_id": config.get("env_id"),
        "global_step": int(global_step),
        "training_timestep": int(global_step),
        "wall_clock_time": float(time.time() - start_time),
    }
    payload.update(metrics)
    return payload


def maybe_save_checkpoint(config, model, report_path, name, global_step):
    interval = int(config.get("checkpoint_interval", 0) or 0)
    if not config.get("save_model", True) or interval <= 0:
        return
    last_checkpoint_step = int(config.get("_last_checkpoint_step", 0) or 0)
    next_checkpoint_step = ((last_checkpoint_step // interval) + 1) * interval
    while global_step >= next_checkpoint_step:
        checkpoint_path = os.path.join(
            str(report_path), f"{name}_step{int(next_checkpoint_step)}.cleanqrl_model"
        )
        torch.save(model.state_dict(), checkpoint_path)
        config["_last_checkpoint_step"] = int(next_checkpoint_step)
        next_checkpoint_step += interval


def evaluate_greedy_policy(
    config: dict,
    action_fn: Callable[[torch.Tensor], np.ndarray],
    device: torch.device,
    global_step: int,
    report_path,
):
    interval = int(config.get("eval_interval", 0) or 0)
    last_eval_step = int(config.get("_last_eval_step", 0) or 0)
    if interval <= 0 or global_step <= 0:
        return
    next_eval_step = ((last_eval_step // interval) + 1) * interval
    while global_step >= next_eval_step:
        env = make_env(config["env_id"], config)()
        rewards = []
        lengths = []
        base_seed = int(config["seed"]) if config.get("seed") is not None else 0
        for episode_idx in range(int(config.get("eval_episodes", DEFAULT_EVAL_EPISODES))):
            obs, _ = env.reset(seed=base_seed + 10_000 + episode_idx)
            done = False
            total_reward = 0.0
            length = 0
            while not done:
                with torch.no_grad():
                    action = action_fn(torch.Tensor(np.asarray(obs)[None, ...]).to(device))
                obs, reward, terminated, truncated, _ = env.step(int(action[0]))
                done = terminated or truncated
                total_reward += float(reward)
                length += 1
            rewards.append(total_reward)
            lengths.append(length)
        env.close()
        log_metrics(
            config,
            eval_metrics(config, next_eval_step, rewards, lengths),
            report_path,
        )
        config["_last_eval_step"] = int(next_eval_step)
        next_eval_step += interval
