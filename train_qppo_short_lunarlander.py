import os
import shutil
from datetime import datetime

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: PyYAML. Install with `python -m pip install PyYAML`.") from exc

from cleanqrl_utils.train import train_agent


CONFIG_PATH = "configs/benchmarks/ppo_quantum_lunarlander_short_train.yaml"


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(repo_root, CONFIG_PATH)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    config["trial_name"] = f"{timestamp}_{config['trial_name']}_seed{config['seed']}"
    config["path"] = os.path.join(repo_root, "logs", config["trial_name"])

    os.makedirs(config["path"], exist_ok=True)
    shutil.copy(config_path, os.path.join(config["path"], "source_config.yaml"))
    with open(os.path.join(config["path"], "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    train_agent(config)
    print(f"Short-trained QPPO checkpoint/logs written to {config['path']}")


if __name__ == "__main__":
    main()
