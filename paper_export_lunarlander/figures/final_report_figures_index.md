# Final Report Figures Index

This folder contains the curated figures recommended for the final report. QPPO/QRL figures are labeled as short-trained feasibility unless they refer only to parameter count; IBM figures are inference-only hardware feasibility.

## Figure 1: `fig_reward_curves_classical_full_and_qppo_short.png`

Use this as the main reward-trajectory figure. The left panel contains only completed 2M-step classical runs; the right panel isolates the 25k-step QPPO simulator feasibility run so it is not mistaken for a full fair benchmark.

## Figure 2: `fig_final_reward_full_classical_with_qppo_feasibility.png`

Use this for final reward comparison with care: PPO, PPO-tiny, and DQN are completed 2M-step classical runs, while QPPO is a separate 25k-step feasibility result marked with hatching.

## Figure 3: `fig_parameter_count_total_trainable.png`

Use this to discuss parameter efficiency. QPPO has fewer trainable parameters than full PPO but more than PPO-tiny, and parameter count should not be interpreted as performance.

## Figure 4: `fig_compute_sps_full_classical_vs_qppo_short.png`

Use this to compare environment-training throughput. IBM inference is shown as zero because it has no environment-training SPS.

## Figure 5: `fig_compute_wall_clock_by_result_category.png`

Use this to compare measured wall-clock cost by category. The IBM bar is total script/job time, not the approximately 2-second dashboard QPU execution time.

## Figure 6: `fig_compute_circuit_evaluations_and_shots.png`

Use this to show quantum execution overhead. Classical baselines have zero circuit evaluations, QPPO reports simulated circuit evaluations during short training, and IBM reports 5 circuits x 100 shots.

## Figure 7: `fig_ibm_qppo_inference_agreement_table.png`

Use this only for the IBM inference feasibility section. It shows 4/5 action agreement between simulator and hardware on fixed states, not training reward or quantum advantage.
