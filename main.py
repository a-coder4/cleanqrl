import argparse
import os
import shutil
from datetime import datetime

import yaml

from cleanqrl_utils.plotting import plot_single_run
from cleanqrl_utils.train import train_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a CleanQRL training config")
    parser.add_argument(
        "--config",
        default="configs/benchmarks/reinforce_quantum_cartpole.yaml",
        help="Path to a YAML config (relative or absolute).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot results after training (requires result.json).",
    )
    args = parser.parse_args()

    config_path = args.config

    # Load the config file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Based on the current time, create a unique name for the experiment
    config["trial_name"] = (
        datetime.now().strftime("%Y-%m-%d--%H-%M-%S") + "_" + config["trial_name"]
    )

    # Always save logs under this repo's ./logs folder
    repo_root = os.path.dirname(os.path.abspath(__file__))
    logs_root = os.path.join(repo_root, "logs")
    config["path"] = os.path.join(logs_root, config["trial_name"])

    # Create the directory and save a copy of the config file so
    # that the experiment can be replicated
    os.makedirs(os.path.dirname(config["path"] + "/"), exist_ok=True)
    shutil.copy(config_path, os.path.join(config["path"], "config.yaml"))

    # Start the agent training
    train_agent(config)

    if args.plot:
        plot_single_run(config["path"])
