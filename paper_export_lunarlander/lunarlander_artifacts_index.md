# LunarLander Paper Artifact Export Index

This folder is intended to be copied directly into an Overleaf project. All file paths below are relative to `paper_export_lunarlander/`.

## Key Metrics To Use In The Paper

- Classical PPO completed the full 2,000,000-step LunarLander benchmark. Final reward: 282.657; success rate: 1; wall-clock: 362.5 s; SPS: 5,516.
- Classical DQN completed the full 2,000,000-step LunarLander benchmark. Final reward: 263.940; success rate: 1; wall-clock: 436.7 s; SPS: 2,853.5.
- Short-trained QPPO/QRL is feasibility-only, not a full 2M-step fair benchmark. It completed 25,000 steps, final reward -626.445, success rate 0, wall-clock 1,395.3 s, SPS 17, and 125,000 simulated circuit evaluations.
- IBM inference was hardware-only: 5 fixed LunarLander states, 100 shots each, action agreement 0.800, total job/script time about 1,617.9 s, and dashboard QPU execution time about 2 s.

## Exported Files

| Relative path | Supports in paper |
| --- | --- |
| `figures/fig_compute_circuit_evaluations_and_shots.png` | Circuit-evaluation and shot-cost comparison. |
| `figures/fig_compute_sps_full_classical_vs_qppo_short.png` | SPS comparison for full classical training versus short QPPO feasibility. |
| `figures/fig_compute_wall_clock_by_result_category.png` | Wall-clock compute comparison by result category. |
| `figures/fig_final_reward_full_classical_with_qppo_feasibility.png` | Final reward comparison with QPPO explicitly marked as short-trained feasibility only. |
| `figures/fig_ibm_qppo_inference_agreement_table.png` | IBM hardware action-agreement figure/table. |
| `figures/fig_parameter_count_total_trainable.png` | Total trainable parameter comparison. |
| `figures/fig_reward_curves_classical_full_and_qppo_short.png` | Main reward-curve figure: completed 2M-step classical runs separated from 25k-step QPPO feasibility. |
| `figures/final_report_figures_index.md` | Figure index with report captions and usage notes. |
| `rollout_visuals/dqn_classical/episode_00.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/dqn_classical/episode_00_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_00_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_00_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_00_touchdown_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_00_touchdown_or_ground_contact.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_01.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/dqn_classical/episode_01_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_01_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_01_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_01_touchdown_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_01_touchdown_or_ground_contact.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_02.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/dqn_classical/episode_02_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_02_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_02_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_02_touchdown_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/dqn_classical/episode_02_touchdown_or_ground_contact.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/lunarlander_simulation_visuals_index.md` | Index for qualitative LunarLander rollout videos and screenshots. |
| `rollout_visuals/ppo_classical/episode_00.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/ppo_classical/episode_00_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_00_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_00_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_00_touchdown_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_00_touchdown_or_ground_contact.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_01.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/ppo_classical/episode_01_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_01_final_state.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_01_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_01_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_01_touchdown_or_ground_contact.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_02.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/ppo_classical/episode_02_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_02_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_02_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_02_touchdown_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/ppo_classical/episode_02_touchdown_or_ground_contact.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_00.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/qppo_qrl_short_trained/episode_00_crash_or_failure_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_00_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_00_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_00_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_01.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/qppo_qrl_short_trained/episode_01_crash_or_failure_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_01_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_01_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_01_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_02.mp4` | Qualitative deterministic rollout video; illustrative only, not quantitative evidence. |
| `rollout_visuals/qppo_qrl_short_trained/episode_02_crash_or_failure_final.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_02_descent.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_02_near_landing.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/qppo_qrl_short_trained/episode_02_start.png` | Qualitative rollout still image from a key episode moment. |
| `rollout_visuals/rollout_manifest.json` | Machine-readable rollout metadata with returns, status, checkpoint paths, and captions. |
| `summaries/compute_cost_comparison.csv` | Compute-cost table separating full classical training, short QPPO feasibility, and IBM inference-only results. |
| `summaries/compute_cost_comparison.md` | Report-ready compute-cost explanation including IBM QPU hardware details. |
| `summaries/fairness_check_report.md` | Fairness audit documenting standardized environment, budget, seeds, evaluation, and exclusions. |
| `summaries/ibm_qppo_short_trained_inference_results.csv` | CSV version of IBM QPU inference results. |
| `summaries/ibm_qppo_short_trained_inference_results.md` | IBM QPU inference table for five fixed LunarLander states. |
| `summaries/lunarlander_aggregate_summary.md` | Aggregated LunarLander result summary for completed and excluded runs. |
| `summaries/lunarlander_report_ready_section.md` | Report-ready discussion of completed results, matched-budget diagnostics, and quantum infeasibility. |
| `summaries/parameter_count_comparison.csv` | Parameter-count table for PPO, PPO-tiny, DQN, and short-trained QPPO/QRL. |
| `summaries/parameter_count_comparison.md` | Report-ready parameter-count explanation and table. |

## Usage Notes

- The files in `figures/` are the preferred paper figures and already use captions/titles that separate full-training classical results, short-trained QPPO feasibility, and IBM inference-only results.
- The files in `rollout_visuals/` are qualitative examples only and should not be used as quantitative performance evidence.
- This export intentionally excludes raw checkpoints, TensorBoard logs, large aggregate row-level logs, and training scripts.