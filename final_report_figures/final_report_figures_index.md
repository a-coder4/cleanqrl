# Final Report Figures Index

This folder contains the curated figures recommended for the final report. Main training figures use only the completed matched 100k LunarLander cohort. IBM figures remain inference-only hardware feasibility.

## Figure 1: `fig_reward_curves_matched_100k_all_algorithms.png`

Use this as the main reward-trajectory figure. It contains only completed matched 100k-step runs for PPO, PPO-tiny, DQN, Quantum DQN, and QRL.

## Figure 2: `fig_final_reward_best_qrl_vs_classical_100k.png`

Use this for the requested bar-chart comparison: PPO, PPO-tiny, and DQN are seed means from the matched 100k cohort, and QRL is the best QRL seed from that same cohort.

## Figure 3: `fig_parameter_count_total_trainable.png`

Use this to discuss parameter efficiency. QPPO has fewer trainable parameters than full PPO but more than PPO-tiny, and parameter count should not be interpreted as performance.

## Figure 4: `fig_compute_sps_full_classical_vs_qppo_short.png`

Use this to compare environment-training throughput for the completed matched 100k LunarLander runs.

## Figure 5: `fig_compute_wall_clock_by_result_category.png`

Use this to compare measured wall-clock cost for the completed matched 100k LunarLander runs.

## Figure 6: `fig_compute_circuit_evaluations_and_shots.png`

Use this to show quantum execution overhead for the matched training cohort. Classical baselines have zero circuit evaluations; quantum rows report simulator circuit evaluations.

## Figure 7: `fig_ibm_qppo_inference_agreement_table.png`

Use this only for the IBM inference feasibility section. It shows 4/5 action agreement between simulator and hardware on fixed states, not training reward or quantum advantage.
