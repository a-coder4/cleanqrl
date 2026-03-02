import os
import sys

# This is important for the import of the cleanqrl package. Do not delete this line
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
sys.path.append(os.path.join(repo_path, "cleanqrl"))

from cleanqrl.dqn_classical import dqn_classical
from cleanqrl.dqn_quantum import dqn_quantum
from cleanqrl.ppo_classical import ppo_classical
from cleanqrl.ppo_quantum import ppo_quantum
from cleanqrl.ppo_quantum_hybrid import ppo_quantum_hybrid
from cleanqrl.reinforce_classical import reinforce_classical
from cleanqrl.reinforce_quantum import reinforce_quantum

agent_switch = {
    "ppo_classical": ppo_classical,
    "ppo_quantum": ppo_quantum,
    "ppo_quantum_hybrid": ppo_quantum_hybrid,
    "dqn_classical": dqn_classical,
    "dqn_quantum": dqn_quantum,
    "reinforce_classical": reinforce_classical,
    "reinforce_quantum": reinforce_quantum,
}

def train_agent(config):
    try:
        agent_type = config["agent"].lower()
        if agent_type not in agent_switch:
            raise KeyError(
                f"Agent type '{agent_type}' not found in agent_switch dictionary"
            )
        agent_switch[agent_type](config)
    except KeyError as e:
        raise e
