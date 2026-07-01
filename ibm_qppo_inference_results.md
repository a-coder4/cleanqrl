# IBM Quantum QPPO Inference-Only Demo

This demo runs the small 4-qubit QPPO/QRL actor circuit for five fixed LunarLander states. It does not train on IBM hardware.

| State | Simulator output | IBM hardware output | Selected action | Agreement | Backend | Shots | Job ID | Circuit time (s) |
| ---: | --- | --- | --- | --- | --- | ---: | --- | ---: |
| 0 | `[0.1810, 0.0643, 0.1316, 0.1946]` | `[0.1400, 0.0200, 0.1000, 0.1400]` | fire_right (3) | False | ibm_kingston | 100 | d92cam7qq29s738p30fg | 9.421 |
