# LunarLander Aggregated Results Summary

Generated: 2026-07-13 21:19:00

Main matched budget: 100,000 environment interactions
Main seed set: 0, 1, 2
Included matched runs: 15
Excluded out-of-scope or incomplete runs: 12

| Agent | Runs | Final Reward Mean | Final Reward Min | Final Reward Max | Final Success Rate | Mean SPS | Mean Wall-Clock Time | Mean Circuit Evaluations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | 3 | -156.345 | -196.591 | -96.334 | 0.000 | 2,968 | 33.684 |  |
| PPO | 3 | -233.738 | -338.476 | -166.736 | 0.000 | 4,319 | 23.171 |  |
| PPO-tiny | 3 | -1,188 | -1,844 | -532.882 | 0.000 | 4,078 | 24.522 |  |
| QRL | 3 | -815.517 | -1,079 | -357.771 | 0.000 | 9.333 | 10,114 | 1,100,000 |
| Quantum DQN | 3 | -217.484 | -254.953 | -148.642 | 0.000 | 3.000 | 25,599 | 358,654,043 |

## Excluded Runs

- `2026-06-29--13-49-18_ppo_lunarlander_stable_B_seed0`: missing or irregular evaluation rows; missing or irregular checkpoints
- `2026-06-29--13-55-15_dqn_lunarlander_classical_seed0`: training budget 2000000 does not match 100000
- `2026-06-29--14-06-06_dqn_lunarlander_quantum_seed0`: incomplete timesteps (21587/2000000); missing or irregular evaluation rows; missing or irregular checkpoints
- `2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0`: training budget 2000000 does not match 100000
- `2026-06-29--15-17-34_ppo_tiny_classical_seed0`: training budget 2000000 does not match 100000
- `2026-06-29--15-26-31_dqn_lunarlander_classical_seed0`: training budget 2000000 does not match 100000
- `2026-06-29--15-39-18_dqn_lunarlander_quantum_seed0`: incomplete timesteps (447700/2000000); missing or irregular evaluation rows; missing or irregular checkpoints
- `2026-07-10--14-42-41_ppo_lunarlander_stable_B_900k_seed0`: training budget 2000000 does not match 100000
- `2026-07-10--14-51-39_ppo_lunarlander_stable_B_900k_seed0`: incomplete timesteps (896000/900000); missing or irregular evaluation rows; missing or irregular checkpoints
- `2026-07-10--14-57-31_ppo_lunarlander_stable_B_900k_seed0`: training budget 900000 does not match 100000
- `2026-07-10--15-00-19_ppo_tiny_classical_900k_seed0`: training budget 900000 does not match 100000
- `2026-07-10--15-04-22_dqn_lunarlander_classical_900k_seed0`: training budget 900000 does not match 100000

## Supplemental Notes

The main plots and table use only completed runs from the matched 100,000-step, seeds 0-2 cohort. Older full-training or partial exploratory runs remain in the aggregate CSV with `included_in_plots == no` for transparency.

Older incomplete quantum diagnostics reached up to 447,690 environment interactions after about 1.49 days. They are retained only as historical compute-feasibility context.
