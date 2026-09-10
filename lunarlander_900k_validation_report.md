# Matched LunarLander-v3 900k Validation Report

Generated: 2026-09-09T22:45:40-07:00
Logs scanned: `C:\Users\Aadesh\cleanqrl\logs`

## Overall status

**FAIL** — 3/12 required runs selected.

The experiment is complete only when this report says PASS and all twelve agent/seed rows below pass.

## Required cohort

| Agent | Seed | Status | Selected run |
| --- | --- | --- | --- |
| PPO | 0 | PASS | `2026-07-10--14-57-31_ppo_lunarlander_stable_B_900k_seed0` |
| PPO | 1 | MISSING |  |
| PPO | 2 | MISSING |  |
| QRL | 0 | MISSING |  |
| QRL | 1 | MISSING |  |
| QRL | 2 | MISSING |  |
| PPO-tiny | 0 | PASS | `2026-07-10--15-00-19_ppo_tiny_classical_900k_seed0` |
| PPO-tiny | 1 | MISSING |  |
| PPO-tiny | 2 | MISSING |  |
| DQN | 0 | PASS | `2026-07-10--15-04-22_dqn_lunarlander_classical_900k_seed0` |
| DQN | 1 | MISSING |  |
| DQN | 2 | MISSING |  |

## Missing runs

- PPO seed 1
- PPO seed 2
- QRL seed 0
- QRL seed 1
- QRL seed 2
- PPO-tiny seed 1
- PPO-tiny seed 2
- DQN seed 1
- DQN seed 2

## Duplicate resolution

Selection rule: for each agent/seed, select the newest completely valid dedicated 900k run, using the leading run timestamp and then the run name as a deterministic tie-breaker. A valid older duplicate remains reported but is not selected.

No agent/seed has multiple completely valid candidates.

## Candidate inventory

| Run | Agent | Seed | Configured | Max logged | Status | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| `2026-01-22--02-39-44_dqn_lunarlander_classical` | DQN | 42 | 500000 |  | FAIL | seed 42 is outside [0, 1, 2]; configured total_timesteps=500000, expected 900000 |
| `2026-01-24--13-32-47_ppo_lunarlander_stable_B` | PPO | 42 | 2000000 |  | FAIL | seed 42 is outside [0, 1, 2]; configured total_timesteps=2000000, expected 900000 |
| `2026-01-31--00-06-31_qppo_hybrid_v1` | QRL |  | 1000000 |  | FAIL | seed None is outside [0, 1, 2]; configured total_timesteps=1000000, expected 900000 |
| `2026-02-10--00-34-25_qppo_hybrid_scaling_v3` | ppo_quantum_hybrid_scaled |  | 1500000 |  | FAIL | agent 'ppo_quantum_hybrid_scaled' is outside the primary cohort; seed None is outside [0, 1, 2]; configured total_timesteps=1500000, expected 900000 |
| `2026-02-10--03-06-36_qppo_hybrid_configC` | QRL |  | 3000000 |  | FAIL | seed None is outside [0, 1, 2]; configured total_timesteps=3000000, expected 900000 |
| `2026-02-10--06-56-04_qppo_hybrid_configC` | QRL | 2 | 3000000 |  | FAIL | configured total_timesteps=3000000, expected 900000 |
| `2026-02-10--06-56-36_qppo_hybrid_configC` | QRL | 3 | 3000000 |  | FAIL | seed 3 is outside [0, 1, 2]; configured total_timesteps=3000000, expected 900000 |
| `2026-02-15--03-02-17_ppo_tiny_classical_seed42` |  |  | 2000000 |  | FAIL | agent '' is outside the primary cohort; seed None is outside [0, 1, 2]; configured total_timesteps=2000000, expected 900000 |
| `2026-02-23--17-45-53_qppo_hybrid_configC_best` | QRL | 2 | 1500000 |  | FAIL | configured total_timesteps=1500000, expected 900000 |
| `2026-06-29--13-49-18_ppo_lunarlander_stable_B_seed0` | PPO | 0 | 2000000 |  | FAIL | configured total_timesteps=2000000, expected 900000 |
| `2026-06-29--13-55-15_dqn_lunarlander_classical_seed0` | DQN | 0 | 2000000 |  | FAIL | configured total_timesteps=2000000, expected 900000 |
| `2026-06-29--14-06-06_dqn_lunarlander_quantum_seed0` | DQN_quantum | 0 | 2000000 |  | FAIL | agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=2000000, expected 900000 |
| `2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0` | PPO | 0 | 2000000 |  | FAIL | configured total_timesteps=2000000, expected 900000 |
| `2026-06-29--15-17-34_ppo_tiny_classical_seed0` | PPO-tiny | 0 | 2000000 |  | FAIL | configured total_timesteps=2000000, expected 900000 |
| `2026-06-29--15-26-31_dqn_lunarlander_classical_seed0` | DQN | 0 | 2000000 |  | FAIL | configured total_timesteps=2000000, expected 900000 |
| `2026-06-29--15-39-18_dqn_lunarlander_quantum_seed0` | DQN_quantum | 0 | 2000000 |  | FAIL | agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=2000000, expected 900000 |
| `2026-07-01--00-55-29_qppo_short_trained_lunarlander_seed0` | QRL | 0 | 25000 |  | FAIL | configured total_timesteps=25000, expected 900000 |
| `2026-07-10--14-42-41_ppo_lunarlander_stable_B_900k_seed0` | PPO | 0 | 2000000 |  | FAIL | configured total_timesteps=2000000, expected 900000 |
| `2026-07-10--14-51-39_ppo_lunarlander_stable_B_900k_seed0` | PPO | 0 | 900000 | 896000 | FAIL | max logged training timestep=896000, expected 900000; evaluation steps=[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000], expected [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000]; evaluation row count=8, expected 9; no training_diagnostic row at 900000; checkpoint steps=[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000], expected [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000]; checkpoint file count=8, expected 9 |
| `2026-07-10--14-57-31_ppo_lunarlander_stable_B_900k_seed0` | PPO | 0 | 900000 | 900000 | SELECTED |  |
| `2026-07-10--15-00-19_ppo_tiny_classical_900k_seed0` | PPO-tiny | 0 | 900000 | 900000 | SELECTED |  |
| `2026-07-10--15-04-22_dqn_lunarlander_classical_900k_seed0` | DQN | 0 | 900000 | 900000 | SELECTED |  |
| `2026-07-11--23-30-04_ppo_lunarlander_stable_B_seed0` | PPO | 0 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-11--23-30-35_ppo_tiny_classical_seed0` | PPO-tiny | 0 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-11--23-31-04_dqn_lunarlander_classical_seed0` | DQN | 0 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-11--23-31-41_dqn_lunarlander_quantum_seed0` | DQN_quantum | 0 | 100000 |  | FAIL | agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=100000, expected 900000 |
| `2026-07-12--06-37-43_qppo_hybrid_configC_seed0` | QRL | 0 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--09-28-53_ppo_lunarlander_stable_B_seed1` | PPO | 1 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--09-29-19_ppo_tiny_classical_seed1` | PPO-tiny | 1 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--09-29-49_dqn_lunarlander_classical_seed1` | DQN | 1 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--09-30-25_dqn_lunarlander_quantum_seed1` | DQN_quantum | 1 | 100000 |  | FAIL | agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=100000, expected 900000 |
| `2026-07-12--16-41-12_qppo_hybrid_configC_seed1` | QRL | 1 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--19-30-57_ppo_lunarlander_stable_B_seed2` | PPO | 2 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--19-31-23_ppo_tiny_classical_seed2` | PPO-tiny | 2 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--19-31-52_dqn_lunarlander_classical_seed2` | DQN | 2 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |
| `2026-07-12--19-32-29_dqn_lunarlander_quantum_seed2` | DQN_quantum | 2 | 100000 |  | FAIL | agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=100000, expected 900000 |
| `2026-07-13--02-36-19_qppo_hybrid_configC_seed2` | QRL | 2 | 100000 |  | FAIL | configured total_timesteps=100000, expected 900000 |

## Protocol enforced

- Environment: `LunarLander-v3`; native discrete actions; no action remapping.
- Configured and completed interactions: exactly 900,000.
- Agents: PPO, QRL, PPO-tiny, DQN.
- Seeds: [0, 1, 2] for every agent.
- Observation preprocessing: `none`.
- Deterministic greedy evaluation with exploration disabled: 10 episodes at 100k, 200k, 300k, 400k, 500k, 600k, 700k, 800k, 900k.
- Checkpoints every 100,000 interactions through 900,000.
- Success threshold: episode reward >= 200.
- Standard train/evaluation metric schemas, final positive wall-clock/SPS diagnostics, and QRL circuit accounting.
- Runs configured for any horizon other than 900k are excluded, even if they contain a 900k checkpoint.
