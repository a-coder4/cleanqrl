# LunarLander Fairness Check Report

Generated: 2026-06-30 12:35:40
Logs scanned: `C:\Users\Aadesh\cleanqrl\logs`
Standardized runs found: 7

## Overall Status

**FAIL**

| Check | Status | Details |
| --- | --- | --- |
| Expected agents present | FAIL | ppo_quantum_hybrid |
| No unexpected standardized agents | PASS | none |
| Observed seed set | FAIL | [0] |
| Every agent has every seed | FAIL | expected [0, 1, 2, 3, 4] per agent |
| Same env_id | PASS | ['LunarLander-v3'] |
| Same total_timesteps | PASS | [2000000] |
| Same eval protocol | PASS | intervals=[100000], episodes=[10] |
| Same checkpoint interval | PASS | [100000] |
| Same observation preprocessing | PASS | ['none'] |

## Run Inventory

| Run | Agent | Seed | Status | Train Rows | Eval Rows | Max Timestep | Eval Steps | Checkpoint Count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-06-29--13-49-18_ppo_lunarlander_stable_B_seed0 | PPO_classical | 0 | FAIL | 8259 | 10 | 2000000 | 200000, 400000, 600000, 800000, 1000000 ... | 10 |
| 2026-06-29--13-55-15_dqn_lunarlander_classical_seed0 | DQN_classical | 0 | PASS | 4917 | 20 | 2000000 | 100000, 200000, 300000, 400000, 500000 ... | 20 |
| 2026-06-29--14-06-06_dqn_lunarlander_quantum_seed0 | DQN_quantum | 0 | FAIL | 219 | 0 | 21587 |  | 0 |
| 2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0 | PPO_classical | 0 | PASS | 8259 | 20 | 2000000 | 100000, 200000, 300000, 400000, 500000 ... | 20 |
| 2026-06-29--15-17-34_ppo_tiny_classical_seed0 | PPO_tiny_classical | 0 | PASS | 3849 | 20 | 2000000 | 100000, 200000, 300000, 400000, 500000 ... | 20 |
| 2026-06-29--15-26-31_dqn_lunarlander_classical_seed0 | DQN_classical | 0 | PASS | 4917 | 20 | 2000000 | 100000, 200000, 300000, 400000, 500000 ... | 20 |
| 2026-06-29--15-39-18_dqn_lunarlander_quantum_seed0 | DQN_quantum | 0 | FAIL | 2548 | 2 | 267400 | 100000, 200000 | 2 |

## Agent Seed Coverage

| Display Name | Agent Key | Observed Seeds | Status |
| --- | --- | --- | --- |
| PPO classical | PPO_classical | [0] | FAIL |
| PPO-tiny | PPO_tiny_classical | [0] | FAIL |
| DQN classical | DQN_classical | [0] | FAIL |
| Quantum DQN | DQN_quantum | [0] | FAIL |
| Quantum PPO | ppo_quantum_hybrid | [] | FAIL |

## Findings

- Missing expected agents: ppo_quantum_hybrid
- Observed seed set is [0], expected [0, 1, 2, 3, 4].
- `2026-06-29--13-49-18_ppo_lunarlander_stable_B_seed0`:
  - evaluation row count is 10, expected 20
  - evaluation steps are [200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000], expected [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000]
  - checkpoint count is 10, expected 20
  - checkpoint steps are [200000, 400000, 600000, 800000, 1000000, 1200000, 1400000, 1600000, 1800000, 2000000], expected [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000]
- `2026-06-29--14-06-06_dqn_lunarlander_quantum_seed0`:
  - no evaluation standard metric rows found
  - max logged training_timestep is 21587, expected 2000000
- `2026-06-29--15-39-18_dqn_lunarlander_quantum_seed0`:
  - max logged training_timestep is 267400, expected 2000000

## Comparison Rules Checked

- All runs must use `LunarLander-v3`.
- All runs must use `total_timesteps: 2000000`.
- Every expected agent must have seeds `[0, 1, 2, 3, 4]`.
- All runs must use `observation_preprocessing: none`.
- All runs must use `eval_interval: 100000` and `eval_episodes: 10`.
- All runs must use `checkpoint_interval: 100000`.
- Standard train/eval rows must include `episode_reward`, `episode_length`, `success_rate`, and `training_timestep`.
