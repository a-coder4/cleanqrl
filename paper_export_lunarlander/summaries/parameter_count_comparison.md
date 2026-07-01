# LunarLander Parameter Count Comparison

| Agent | Policy/actor params | Critic/value params | Q-network params | Quantum circuit params | Total trainable params | Status |
| --- | --- | --- | --- | --- | --- | --- |
| PPO | 4,996 | 4,801 |  | 0 | 9,797 | Classical baseline |
| PPO-tiny | 420 | 321 |  | 0 | 741 | Classical baseline |
| DQN | 11,584 | 0 | 11,584 | 0 | 11,584 | Classical baseline |
| QPPO/QRL short-trained | 861 | 4,801 |  | 24 | 5,662 | Short-trained feasibility model |

![Total trainable parameter comparison](parameter_count_total_trainable_params.png)

**Figure caption.** Total trainable parameter count for the LunarLander PPO, PPO-tiny, DQN, and short-trained QPPO/QRL agents. DQN counts the optimized online Q-network only; the target network is not counted as a separately trained model.

## Report-Ready Interpretation

The short-trained QPPO/QRL model uses 5,662 trainable parameters, which is 57.8% of full PPO's 9,797 parameters. In that narrow parameter-count sense, QPPO is more parameter-efficient than the full PPO baseline. However, QPPO is not smaller than PPO-tiny: it has 7.6x as many trainable parameters as the 741-parameter PPO-tiny baseline. Only 24 QPPO parameters are quantum circuit weights; most trainable parameters are still in the classical encoder and critic. These counts should not be interpreted as evidence of better performance from fewer parameters, because the short-trained QPPO hardware result is an inference-only feasibility demonstration and the matched training results do not establish quantum advantage.

## Counting Notes

- Full PPO uses separate actor and critic MLPs with two 64-unit hidden layers.
- PPO-tiny uses separate actor and critic MLPs with one 32-unit hidden layer.
- DQN uses one optimized online Q-network with 8-120-84-4 dimensions.
- QPPO/QRL short-trained uses a classical 8-64-4 encoder, a 4-qubit 2-layer variational actor circuit with 24 trainable circuit weights, one actor scaling parameter, and a classical 8-64-64-1 critic.
