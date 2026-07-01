# IBM Quantum QPPO Inference-Only Demo

This demo is intentionally inference-only on IBM hardware. Any training happens locally with the simulator.

## Short local QPPO training

Run a small simulator-only QPPO warmup before hardware inference:

```powershell
python train_qppo_short_lunarlander.py
```

This uses:

- `LunarLander-v3`
- 25,000 environment interactions
- seed `0`
- 4 qubits
- 2 layers
- simulator-only training

The generated checkpoint is saved under:

```text
logs\<timestamp>_qppo_short_trained_lunarlander_seed0\
```

If your current Python environment is missing RL dependencies such as `gymnasium`, `torch`, or `pennylane`, activate the CleanQRL environment first.

The script runs five fixed LunarLander states through the small 4-qubit QPPO/QRL actor circuit:

- Angle embedding with `RX` rotations, matching `qml.AngleEmbedding(..., rotation="X")`
- Two 4-qubit variational layers using `RZ-RY-RZ` rotations and ring `CX` entanglement
- Pauli-Z expectation values mapped to the four LunarLander discrete actions

By default, only the local NumPy statevector simulator is used:

```powershell
python ibm_qppo_inference_demo.py
```

Outputs:

```text
ibm_qppo_short_trained_inference_results.csv
ibm_qppo_short_trained_inference_results.md
```

To run the same five measured circuits on IBM Quantum hardware, first install the optional IBM dependencies:

```powershell
python -m pip install qiskit qiskit-ibm-runtime
```

Set your token as an environment variable in the same PowerShell session. Do not commit it to the repo:

```powershell
$env:IBM_QUANTUM_TOKEN = "paste_token_here"
```

Verify account access and available hardware backends without submitting a QPU job:

```powershell
python ibm_qppo_inference_demo.py --list-backends
```

Verify backend selection and circuit transpilation without submitting a QPU job:

```powershell
python ibm_qppo_inference_demo.py --dry-run-ibm --shots 100 --backend ibm_brisbane
```

Then run with low shots:

```powershell
python ibm_qppo_inference_demo.py --run-ibm --shots 100 --backend ibm_brisbane
```

Hardware execution uses Qiskit Runtime `SamplerV2`, not `backend.run()`.

Safety notes:

- The script refuses to prepare more than 10 circuits.
- IBM hardware submission requires typing `SUBMIT` unless `--yes` is passed.
- More than 5 circuits triggers an additional confirmation prompt.
- Keep `--shots 100` unless you have a specific reason to spend more QPU time.
- The account has only about 10 free QPU minutes, so do not use this for training.

Optional checkpoint use:

```powershell
python ibm_qppo_inference_demo.py --model-path logs\YOUR_QPPO_RUN\YOUR_MODEL.cleanqrl_model
```

If no checkpoint is supplied, the script automatically looks for the newest `qppo_short_trained_lunarlander` checkpoint. If no short-trained checkpoint exists, it uses deterministic demo weights so the IBM workflow can still be tested without requiring a completed QPPO run.

This is labeled as **short-trained QPPO hardware inference feasibility**, not full quantum training and not quantum advantage.
