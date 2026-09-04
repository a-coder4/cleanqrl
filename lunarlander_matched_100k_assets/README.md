# LunarLander matched-100k asset bundle

This flat folder gathers the newest matched-100k LunarLander data and figures available in this repository.

- Source Git ref: `origin/main`
- Source commit: `cad3d6f162fd0c5716d8f4e47b4db3b71f50e393`
- Protocol: 100,000 environment interactions per run
- Models: PPO, PPO-tiny, DQN, QRL (the QPPO/QRL hybrid), and Quantum DQN
- Seeds: 0, 1, and 2 for every model
- Validated matched runs: 15
- Matched metric rows: 15125

## Data

- `matched_100k_all_metrics.csv`: all metric rows for the 15 included matched runs only. The source aggregate's older/excluded runs have been removed.
- `matched_100k_seed_report.csv`: one row per model/seed, including the source run name, row counts, final evaluation reward/success rate, SPS, wall-clock time, and circuit evaluations.
- `matched_100k_model_summary.csv`: five-model aggregate compute/performance summary used by the report figures.
- `matched_100k_data.xlsx`: formatted workbook containing the full metrics, seed report, model summary, and source/protocol notes.

## Figures

- `01_reward_curves_matched_100k.png`: rolling-mean training reward curves for all five models.
- `01b_final_evaluation_rewards_at_100k_all_seeds.png`: all three final evaluation points per model at 100k.
- `02_final_reward_comparison.png`: final-reward distribution across the three seeds per model.
- `03_best_qrl_vs_classical.png`: best QRL seed versus classical model means.
- `04_success_rate_comparison.png`: success rate over environment interactions.
- `05_compute_cost_combined.png`: combined SPS and circuit-evaluation comparison.
- `05a_training_time_wall_clock.png`: mean wall-clock training time.
- `05b_training_throughput_sps.png`: mean steps per second.
- `05c_computational_cost_circuit_evaluations.png`: mean circuit-evaluation cost.

## Interpretation notes

`QRL` is the repository's label for the QPPO/QRL hybrid (`qppo_hybrid_configC`). Final evaluation success is 0 for all 15 runs at 100k steps. The success-rate curve can still contain isolated successful training episodes before the final evaluation. The final-reward distribution uses each run's last-100 training-episode mean, while the seed report and best-QRL bar use the final evaluation reward at 100k.

`gather_bundle.rb` reproduces the repository-sourced CSV and figure files from the source Git ref and validates the five-model, three-seed, 100k protocol before writing outputs.
