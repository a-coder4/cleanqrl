# Short-Trained QPPO Hardware Inference Feasibility

This demo runs a short-trained small 4-qubit QPPO/QRL actor circuit for five fixed LunarLander states. It does not train on IBM hardware and should not be interpreted as quantum advantage.

| State | Simulator output | IBM hardware output | Selected action | Agreement | Backend | Shots | Job ID | Circuit time (s) |
| ---: | --- | --- | --- | --- | --- | ---: | --- | ---: |
| 0 | `[0.0621, -0.2189, 0.1308, 0.0593]` | `[0.0200, -0.2200, -0.0600, 0.3200]` | fire_main (2) | False | ibm_kingston | 100 | d92dgg357qjs73b7qcug | 1617.874 |
| 1 | `[0.1574, -0.3605, 0.0533, -0.0813]` | `[0.3600, -0.1800, 0.2400, -0.1200]` | noop (0) | True | ibm_kingston | 100 | d92dgg357qjs73b7qcug | 1617.874 |
| 2 | `[-0.0009, -0.1672, 0.2467, -0.0877]` | `[0.0800, -0.2400, 0.1600, 0.0000]` | fire_main (2) | True | ibm_kingston | 100 | d92dgg357qjs73b7qcug | 1617.874 |
| 3 | `[0.1831, -0.3009, 0.0905, -0.0023]` | `[0.2400, -0.1800, 0.1600, 0.0400]` | noop (0) | True | ibm_kingston | 100 | d92dgg357qjs73b7qcug | 1617.874 |
| 4 | `[0.2597, -0.3711, -0.0618, -0.0115]` | `[0.1400, -0.3200, -0.2000, 0.0400]` | noop (0) | True | ibm_kingston | 100 | d92dgg357qjs73b7qcug | 1617.874 |

Action agreement rate: **0.800**
