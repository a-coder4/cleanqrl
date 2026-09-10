# Matched LunarLander-v3 900k dataset

Status: **INCOMPLETE** (3/12 validated runs).

Generated: 2026-09-09T22:45:45-07:00

Source Git commit: `ab1815b7b808b276965d733098cecc403ac7df9e`

## Exact protocol

- Environment: `LunarLander-v3` with native discrete actions and no action remapping.
- Agents: PPO, QRL (the existing hybrid Config C architecture), PPO-tiny, and DQN.
- Seeds: 0, 1, and 2 for every agent.
- Configured/completed budget: exactly 900,000 environment interactions per run.
- Observation preprocessing: none.
- Evaluation: 10 deterministic greedy episodes, exploration disabled, at every 100,000 interactions from 100k through 900k.
- Checkpoints: every 100,000 interactions from 100k through 900k.
- Success: episode reward >= 200.
- The 95% reward intervals use Student-t with `df=2`, `t=4.3026527`; success intervals use Wilson 95% binomial intervals.
- Historical 1M/2M runs and runs merely containing a 900k checkpoint are excluded.

## Included run paths

- `C:\Users\Aadesh\cleanqrl\logs\2026-07-10--14-57-31_ppo_lunarlander_stable_B_900k_seed0` (PPO seed 0)
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-10--15-00-19_ppo_tiny_classical_900k_seed0` (PPO-tiny seed 0)
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-10--15-04-22_dqn_lunarlander_classical_900k_seed0` (DQN seed 0)

## Missing required runs

- PPO seed 1
- PPO seed 2
- QRL seed 0
- QRL seed 1
- QRL seed 2
- PPO-tiny seed 1
- PPO-tiny seed 2
- DQN seed 1
- DQN seed 2

## Excluded candidate paths

- `C:\Users\Aadesh\cleanqrl\logs\2026-01-22--02-39-44_dqn_lunarlander_classical` — seed 42 is outside [0, 1, 2]; configured total_timesteps=500000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-01-24--13-32-47_ppo_lunarlander_stable_B` — seed 42 is outside [0, 1, 2]; configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-01-31--00-06-31_qppo_hybrid_v1` — seed None is outside [0, 1, 2]; configured total_timesteps=1000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-02-10--00-34-25_qppo_hybrid_scaling_v3` — agent 'ppo_quantum_hybrid_scaled' is outside the primary cohort; seed None is outside [0, 1, 2]; configured total_timesteps=1500000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-02-10--03-06-36_qppo_hybrid_configC` — seed None is outside [0, 1, 2]; configured total_timesteps=3000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-02-10--06-56-04_qppo_hybrid_configC` — configured total_timesteps=3000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-02-10--06-56-36_qppo_hybrid_configC` — seed 3 is outside [0, 1, 2]; configured total_timesteps=3000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-02-15--03-02-17_ppo_tiny_classical_seed42` — agent '' is outside the primary cohort; seed None is outside [0, 1, 2]; configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-02-23--17-45-53_qppo_hybrid_configC_best` — configured total_timesteps=1500000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-06-29--13-49-18_ppo_lunarlander_stable_B_seed0` — configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-06-29--13-55-15_dqn_lunarlander_classical_seed0` — configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-06-29--14-06-06_dqn_lunarlander_quantum_seed0` — agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-06-29--15-11-27_ppo_lunarlander_stable_B_seed0` — configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-06-29--15-17-34_ppo_tiny_classical_seed0` — configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-06-29--15-26-31_dqn_lunarlander_classical_seed0` — configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-06-29--15-39-18_dqn_lunarlander_quantum_seed0` — agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-01--00-55-29_qppo_short_trained_lunarlander_seed0` — configured total_timesteps=25000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-10--14-42-41_ppo_lunarlander_stable_B_900k_seed0` — configured total_timesteps=2000000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-10--14-51-39_ppo_lunarlander_stable_B_900k_seed0` — max logged training timestep=896000, expected 900000; evaluation steps=[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000], expected [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000]; evaluation row count=8, expected 9; no training_diagnostic row at 900000; checkpoint steps=[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000], expected [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000]; checkpoint file count=8, expected 9
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-11--23-30-04_ppo_lunarlander_stable_B_seed0` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-11--23-30-35_ppo_tiny_classical_seed0` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-11--23-31-04_dqn_lunarlander_classical_seed0` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-11--23-31-41_dqn_lunarlander_quantum_seed0` — agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--06-37-43_qppo_hybrid_configC_seed0` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--09-28-53_ppo_lunarlander_stable_B_seed1` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--09-29-19_ppo_tiny_classical_seed1` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--09-29-49_dqn_lunarlander_classical_seed1` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--09-30-25_dqn_lunarlander_quantum_seed1` — agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--16-41-12_qppo_hybrid_configC_seed1` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--19-30-57_ppo_lunarlander_stable_B_seed2` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--19-31-23_ppo_tiny_classical_seed2` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--19-31-52_dqn_lunarlander_classical_seed2` — configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-12--19-32-29_dqn_lunarlander_quantum_seed2` — agent 'DQN_quantum' is outside the primary cohort; configured total_timesteps=100000, expected 900000
- `C:\Users\Aadesh\cleanqrl\logs\2026-07-13--02-36-19_qppo_hybrid_configC_seed2` — configured total_timesteps=100000, expected 900000

## Files

- `matched_900k_all_metrics.csv`: normalized selected-run metrics plus explicit protocol/configuration columns.
- `matched_900k_seed_report.csv`: one row per required agent/seed, including explicit missing rows.
- `matched_900k_model_summary.csv`: final reward, Student-t CI, pooled evaluation success, and Wilson interval.
- `matched_900k_parameter_summary.csv`: programmatically instantiated parameter decomposition and classical MAC/FLOP estimates.
- `matched_900k_compute_summary.csv`: per-seed and aggregate wall-clock, SPS, and QRL circuit workload.
- `matched_900k_evaluation_checkpoints.csv`: raw seed values and checkpoint-level statistics.
- `matched_900k_data.xlsx`: not created because the workbook artifact runtime is unavailable to this repository script and no project XLSX dependency is declared.

Classical FLOPs use exactly two FLOPs per linear-layer weight (one multiply plus one add); biases and activation costs are excluded. Quantum resources are reported independently and are never converted to FLOPs.

## Regeneration commands (PowerShell)

```powershell
& 'C:\Users\Aadesh\anaconda3\envs\cleanqrl\python.exe' validate_lunarlander_900k.py
& 'C:\Users\Aadesh\anaconda3\envs\cleanqrl\python.exe' aggregate_lunarlander_900k.py
& 'C:\Users\Aadesh\anaconda3\envs\cleanqrl\python.exe' plot_lunarlander_900k_paper_figures.py
```

While the cohort is incomplete, `aggregate_lunarlander_900k.py --allow-incomplete` may be used to regenerate this clearly marked provisional inventory. Plot generation intentionally refuses incomplete data.
