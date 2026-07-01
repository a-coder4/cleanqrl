# LunarLander Aggregated Results Summary

Generated: 2026-06-30 22:11:37

Included complete runs: 4
Excluded incomplete runs: 3
Early-step matched budget: 385,538 environment interactions
Time-budget comparison: 1.27 days

| Agent | Runs | Final Reward Mean | Final Reward Min | Final Reward Max | Final Success Rate | Mean SPS | Mean Wall-Clock Time | Mean Circuit Evaluations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | 2 | 253.602 | 253.602 | 253.602 | 1.000 | 15,427 | 436.665 |  |
| PPO | 1 | 260.208 | 260.208 | 260.208 | 1.000 | 6,580 | 362.527 |  |
| PPO-tiny | 1 | 119.244 | 119.244 | 119.244 | 0.500 | 3,801 | 531.906 |  |

## Excluded Runs

- `2026-06-29--13-49-18_ppo_lunarlander_stable_B_seed0`: missing or irregular evaluation rows; missing or irregular checkpoints
- `2026-06-29--14-06-06_dqn_lunarlander_quantum_seed0`: incomplete timesteps (21587/2000000); missing or irregular evaluation rows; missing or irregular checkpoints
- `2026-06-29--15-39-18_dqn_lunarlander_quantum_seed0`: incomplete timesteps (385538/2000000); missing or irregular evaluation rows; missing or irregular checkpoints

## Supplemental Matched-Budget Comparisons

The early-step matched comparison and time-budget comparison are supplemental diagnostics. They include incomplete quantum runs and truncate classical baselines to the same budget. They are not the main full-training final-performance result.
