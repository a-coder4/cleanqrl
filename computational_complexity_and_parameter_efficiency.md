# Computational Complexity and Parameter Efficiency

Table 1 compares the five agents under the matched protocol of 100,000 environment interactions and seeds 0, 1, and 2. Performance and compute entries are means across the three runs, with sample standard deviations in parentheses where available. Wall-clock time is measured in seconds. Circuit evaluations are simulator-side counts; classical agents require none. Trainable-parameter counts follow the accounting conventions in `parameter_count_comparison.csv`; a directly comparable count is not reported there for Quantum DQN.

| Agent | Final reward, mean ± std | Success rate, mean ± std | SPS, mean ± std | Wall-clock time (s), mean ± std | Circuit evaluations, mean ± std | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | -233.738 ± 91.878 | 0.000 ± 0.000 | 4,319 ± 148 | 23.171 ± 0.805 | 0 ± 0 | 9,797 |
| PPO-tiny | -1,187.776 ± 655.509 | 0.000 ± 0.000 | 4,078 ± 15 | 24.522 ± 0.090 | 0 ± 0 | 741 |
| DQN | -156.345 ± 52.970 | 0.000 ± 0.000 | 2,968 ± 22 | 33.684 ± 0.244 | 0 ± 0 | 11,584 |
| Quantum DQN | -217.484 ± 59.697 | 0.000 ± 0.000 | 3.000 ± 0.000 | 25,599 ± 214 | 358,654,043 ± 49 | Not reported |
| QRL | -815.517 ± 397.937 | 0.000 ± 0.000 | 9.333 ± 0.577 | 10,114 ± 158 | 1,100,000 ± 0 | 5,662 |

At 100,000 interactions, none of the agents recorded a successful evaluation under the configured success criterion. DQN obtained the highest mean final reward, followed by Quantum DQN and PPO; QRL remained substantially below these baselines, although it exceeded PPO-tiny. The matched results therefore provide no evidence that QRL outperforms the classical baselines at this training budget. They also caution against interpreting architectural compactness as learning effectiveness: PPO-tiny has the fewest parameters but the lowest mean reward, while QRL's smaller model does not close the performance gap to full PPO or DQN.

QRL nevertheless exhibits a limited form of parameter efficiency relative to full PPO: its 5,662 trainable parameters are approximately 42% fewer than PPO's 9,797, while its mean reward is better than the much smaller 741-parameter PPO-tiny model. This comparison should remain qualified because the QRL count includes a hybrid actor and classical critic, and parameter count alone does not represent the cost of executing and differentiating a variational circuit. No equivalent conclusion can be drawn for Quantum DQN because its trainable-parameter count is absent from the supplied parameter inventory.

Runtime measurements make the computational trade-off explicit. QRL processed only 9.3 steps per second and required approximately 10,114 seconds (2.81 hours) per run, compared with 4,319 SPS and 23 seconds for PPO; its 1.1 million simulated circuit evaluations dominate this cost. Quantum DQN was slower still at 3 SPS and approximately 25,599 seconds (7.11 hours), accompanied by roughly 358.7 million circuit evaluations. Thus, although QRL reduces parameter count relative to full PPO, simulator circuit-evaluation overhead dominates wall-clock complexity in the present implementation, and the matched experiment does not establish an overall computational or performance advantage.
