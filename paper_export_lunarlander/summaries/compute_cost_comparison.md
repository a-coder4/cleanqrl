# LunarLander Compute-Cost Comparison

| Agent | Category | Steps | Wall-clock (s) | QPU exec. (s) | SPS | Circuit evals | Final reward | Success/agreement | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | Full-training classical | 2,000,000 | 362.5 |  | 5,516 | 0 | 282.657 | 1 | complete full-training result |
| PPO-tiny | Full-training classical | 2,000,000 | 531.9 |  | 3,760 | 0 | 170.232 | 0.500 | complete full-training result |
| DQN | Full-training classical | 2,000,000 | 436.7 |  | 2,853.5 | 0 | 263.940 | 1 | complete full-training result |
| QPPO/QRL short-trained | Short-trained simulator feasibility | 25,000 | 1,395.3 |  | 17 | 125,000 | -626.445 | 0 | short-trained only; not comparable as full-training final performance |
| IBM QPPO hardware inference | IBM inference-only feasibility | 0 | 1,617.9 | 2 |  | 500 |  | 0.800 | inference-only; no training and no environment rollout reward |

## Plots

![SPS comparison](compute_cost_sps.png)

**Caption.** Steps per second for completed classical training runs and the short-trained QPPO/QRL simulator run. IBM hardware inference has no environment-training SPS and is shown as zero.

![Wall-clock comparison](compute_cost_wall_clock_time.png)

**Caption.** Wall-clock seconds for each run category. Classical rows are full-training runs; QPPO/QRL is a short 25k-step simulator training run; IBM is total inference job/script wall-clock time for five fixed states, not the dashboard QPU execution time.

![Circuit evaluation comparison](compute_cost_circuit_evaluations.png)

**Caption.** Circuit evaluation or shot cost. Classical models have zero circuit evaluations, short-trained QPPO/QRL reports simulated circuit evaluations during training, and IBM reports five circuits times 100 shots.

## Report-Ready Interpretation

The compute-cost results separate completed classical full-training runs from short-trained QPPO/QRL and IBM hardware inference-only feasibility runs. QPPO/QRL can reduce trainable parameter count relative to the full PPO actor-critic, but the simulated quantum circuit path introduces major runtime overhead: the short-trained QPPO run completed only 25,000 environment interactions in 1,395 seconds at 17 SPS while accumulating 125,000 simulated circuit evaluations. By contrast, the completed classical baselines reached the full 2,000,000-step budget in minutes with thousands of SPS and no circuit-evaluation cost. The IBM result should be treated separately as inference-only hardware feasibility: it ran five fixed actor circuits with 100 shots each and reports action agreement, not training reward or quantum advantage.

## IBM QPU Hardware Details

| Hardware detail | Value |
| --- | --- |
| Backend | ibm_kingston |
| Processor type | Heron r2 |
| Region | Washington DC us-east |
| Qubits | 156 |
| Couplers | 176 |
| Median 2Q error | 2.01E-3 |
| Layered 2Q error | 3.42E-3 |
| CLOPS | 340K |
| Median readout error | 8.91E-3 |
| Median T1 | 256.51 us |
| Median T2 | 132.57 us |

The IBM hardware run separates end-to-end script/job wall-clock time from actual QPU execution time. The output file reports about 1,618 seconds for the job/script path, while the IBM dashboard showed about 2 seconds of QPU execution on `ibm_kingston`. This means the QPU can execute the small inference circuit quickly once the job reaches hardware, but practical usability is still shaped by queue time, runtime overhead, shot noise, hardware noise, limited free QPU access, and the fact that training was still performed in simulation. The 2-second QPU runtime should not be treated as full training time or as evidence of quantum advantage.

## Notes

- Full-training classical rows come from `lunarlander_aggregate_results.csv` with `included_in_plots == yes`.
- Short-trained QPPO/QRL comes from `logs/*qppo_short_trained_lunarlander*/result.json`.
- IBM hardware inference comes from `results/ibm_qppo_short_trained_inference_results.md`.
- The IBM row's wall-clock time is total script/job time from the output file, while QPU execution time is the dashboard-reported hardware runtime.
- For the IBM inference-only row, the success/agreement column stores action agreement rate, not LunarLander success rate.
