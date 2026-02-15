import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import pennylane as qml
from gymnasium.wrappers import RecordVideo

# --- 1. Define Quantum Circuit & Agent (Copied here to be standalone) ---
n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Encoding
    for q in range(n_qubits):
        qml.RX(inputs[:, q], wires=q)
    # Variational Layer
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    
    def forward(self, x):
        return self.q_layer(x)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # 1. Classical Feature Extractor (8 -> 4)
        self.network = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 4), 
            nn.Tanh(),
        )
        # 2. Quantum Actor Head
        self.actor_scale = nn.Parameter(torch.ones(1) * 1.0)
        self.quantum_layer = QuantumLayer(n_qubits=4, n_layers=2)
        
        # 3. Classical Critic
        self.critic = nn.Sequential(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        value = self.critic(x)
        features = self.network(x)
        features_scaled = features * np.pi 
        q_out = self.quantum_layer(features_scaled)
        logits = q_out * (1.0 + self.actor_scale)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value

# --- 2. Visualization Function (Updated) ---
def enjoy():
    # ---------------------------------------------------------
    # STEP 1: Update this to the folder name of your BEST run
    # (Look for the run with the highest 'max_reward' in your logs)
    # ---------------------------------------------------------
    run_name = "2026-01-31--00-06-31_qppo_hybrid_v1" 
    
    # Path to the saved model
    # Check if your file ends in .pth or .cleanqrl_model and update below:
    model_path = f"logs/{run_name}/{run_name}.cleanqrl_model"
    video_folder = f"runs/{run_name}/videos"

    print(f"Loading model from: {model_path}")
    
    # Setup Environment
    dummy_env = gym.vector.SyncVectorEnv([lambda: gym.make("LunarLander-v3")])
    
    # Setup Video Recording (Record ALL episodes)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder, episode_trigger=lambda x: True)

    # Load Model
    device = torch.device("cpu")
    agent = Agent(dummy_env).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Fix for potential key mismatch in older saves
        if "quantum_layer.weights" in state_dict and "quantum_layer.q_layer.weights" not in state_dict:
            state_dict["quantum_layer.q_layer.weights"] = state_dict.pop("quantum_layer.weights")
        
        agent.load_state_dict(state_dict)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Could not find model at {model_path}")
        return

    agent.eval()
    
    # Run Multiple Episodes to find the best one
    num_episodes = 10
    best_reward = -float('inf')
    best_episode_idx = -1
    
    print(f"Recording {num_episodes} episodes to find the perfect run...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                features = agent.network(obs_tensor)
                features_scaled = features * np.pi
                q_out = agent.quantum_layer(features_scaled)
                logits = q_out * (1.0 + agent.actor_scale)
                action = torch.argmax(logits, dim=1).item()
                
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            
        print(f"Episode {episode}: Reward = {total_reward:.2f}")
        
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_idx = episode

    env.close()
    
    print("-" * 30)
    print(f"🏆 BEST RUN: Episode {best_episode_idx}")
    print(f"💎 SCORE:    {best_reward:.2f}")
    print(f"📹 VIDEO FILE: {video_folder}/rl-video-episode-{best_episode_idx}.mp4")
    print("-" * 30)

if __name__ == "__main__":
    enjoy()