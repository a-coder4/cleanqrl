import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


FIXED_LUNARLANDER_STATES = np.array(
    [
        [0.0000, 1.4000, 0.0000, -0.2000, 0.0000, 0.0000, 0.0, 0.0],
        [-0.2500, 0.9000, -0.1500, -0.4500, 0.1000, 0.0500, 0.0, 0.0],
        [0.1800, 0.6500, 0.2200, -0.3500, -0.1200, -0.0800, 1.0, 0.0],
        [0.0200, 0.1800, 0.0300, -0.1200, 0.0200, 0.0100, 1.0, 1.0],
        [-0.0800, 0.3200, -0.0500, -0.1800, 0.1800, 0.1200, 0.0, 1.0],
    ],
    dtype=np.float64,
)

ACTION_NAMES = {
    0: "noop",
    1: "fire_left",
    2: "fire_main",
    3: "fire_right",
}


@dataclass
class CircuitParams:
    weights: np.ndarray
    actor_scale: float
    encoder_weight_0: Optional[np.ndarray] = None
    encoder_bias_0: Optional[np.ndarray] = None
    encoder_weight_1: Optional[np.ndarray] = None
    encoder_bias_1: Optional[np.ndarray] = None
    source: str = "deterministic_demo_weights"


def deterministic_demo_params() -> CircuitParams:
    rng = np.random.default_rng(0)
    weights = rng.uniform(-np.pi, np.pi, size=(2, 4, 3))
    return CircuitParams(weights=weights, actor_scale=1.0)


def find_latest_short_qppo_checkpoint(logs_dir="logs"):
    if not os.path.isdir(logs_dir):
        return None
    candidates = []
    for root, _, files in os.walk(logs_dir):
        if "qppo_short_trained_lunarlander" not in root:
            continue
        for filename in files:
            if filename.endswith(".cleanqrl_model"):
                path = os.path.join(root, filename)
                candidates.append((os.path.getmtime(path), path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def deterministic_compress_lunarlander_state(state: np.ndarray) -> np.ndarray:
    """Small deterministic stand-in for the QPPO classical 8->4 encoder."""
    compression = np.array(
        [
            [0.60, 0.15, 0.10, 0.00, 0.25, 0.05, 0.20, 0.10],
            [-0.20, 0.50, 0.00, 0.35, 0.05, -0.10, 0.15, 0.20],
            [0.10, -0.10, 0.55, 0.05, 0.45, 0.20, -0.15, 0.00],
            [0.00, 0.05, -0.20, 0.50, 0.10, 0.40, 0.05, -0.15],
        ],
        dtype=np.float64,
    )
    return np.tanh(compression @ state) * np.pi


def qppo_checkpoint_features(state: np.ndarray, params: CircuitParams) -> np.ndarray:
    if params.encoder_weight_0 is None:
        return deterministic_compress_lunarlander_state(state)
    hidden = np.tanh(params.encoder_weight_0 @ state + params.encoder_bias_0)
    features = np.tanh(params.encoder_weight_1 @ hidden + params.encoder_bias_1)
    return features * np.pi


def load_params_from_torch_model(model_path: Optional[str]) -> CircuitParams:
    if not model_path:
        model_path = find_latest_short_qppo_checkpoint()
    if not model_path:
        params = deterministic_demo_params()
        print("No short-trained QPPO checkpoint found. Using deterministic demo weights.")
        return params

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Loading a QPPO checkpoint requires torch. Omit --model-path or install torch."
        ) from exc

    state_dict = torch.load(model_path, map_location="cpu")
    weights = state_dict.get("quantum_layer.weights")
    actor_scale = state_dict.get("actor_scale")
    if weights is None:
        raise SystemExit(
            f"{model_path} does not contain quantum_layer.weights from the hybrid QPPO model."
        )
    scale_value = float(actor_scale.detach().cpu().numpy().reshape(-1)[0]) if actor_scale is not None else 1.0
    encoder_weight_0 = state_dict.get("network.0.weight")
    encoder_bias_0 = state_dict.get("network.0.bias")
    encoder_weight_1 = state_dict.get("network.2.weight")
    encoder_bias_1 = state_dict.get("network.2.bias")
    if any(value is None for value in [encoder_weight_0, encoder_bias_0, encoder_weight_1, encoder_bias_1]):
        raise SystemExit(
            f"{model_path} does not contain the expected QPPO hybrid encoder weights."
        )
    print(f"Loaded short-trained QPPO checkpoint: {model_path}")
    return CircuitParams(
        weights=weights.detach().cpu().numpy(),
        actor_scale=scale_value,
        encoder_weight_0=encoder_weight_0.detach().cpu().numpy(),
        encoder_bias_0=encoder_bias_0.detach().cpu().numpy(),
        encoder_weight_1=encoder_weight_1.detach().cpu().numpy(),
        encoder_bias_1=encoder_bias_1.detach().cpu().numpy(),
        source=model_path,
    )


def build_qppo_actor_circuit(features, weights, measure=False):
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This demo requires qiskit. Install with `python -m pip install qiskit qiskit-ibm-runtime`."
        ) from exc

    qr = QuantumRegister(4, "q")
    if measure:
        cr = ClassicalRegister(4, "c")
        circuit = QuantumCircuit(qr, cr)
    else:
        circuit = QuantumCircuit(qr)

    # Matches QPPO hybrid actor: AngleEmbedding(..., rotation="X")
    for qubit, angle in enumerate(features):
        circuit.rx(float(angle), qr[qubit])

    # Qiskit implementation of the small 4-qubit StronglyEntangling-style actor block.
    for layer in range(weights.shape[0]):
        for qubit in range(4):
            phi, theta, omega = weights[layer, qubit]
            circuit.rz(float(phi), qr[qubit])
            circuit.ry(float(theta), qr[qubit])
            circuit.rz(float(omega), qr[qubit])
        for qubit in range(4):
            circuit.cx(qr[qubit], qr[(qubit + 1) % 4])

    if measure:
        for qubit in range(4):
            circuit.measure(qr[qubit], cr[qubit])
    return circuit


def rx(theta):
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry(theta):
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta):
    return np.array(
        [[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]],
        dtype=np.complex128,
    )


def apply_single_qubit_gate(state, gate, qubit, num_qubits=4):
    updated = np.zeros_like(state)
    mask = 1 << qubit
    for basis in range(1 << num_qubits):
        bit = 1 if basis & mask else 0
        base = basis & ~mask
        updated[basis] += gate[bit, 0] * state[base]
        updated[basis] += gate[bit, 1] * state[base | mask]
    return updated


def apply_cx(state, control, target, num_qubits=4):
    updated = np.zeros_like(state)
    control_mask = 1 << control
    target_mask = 1 << target
    for basis, amplitude in enumerate(state):
        if basis & control_mask:
            updated[basis ^ target_mask] += amplitude
        else:
            updated[basis] += amplitude
    return updated


def numpy_simulator_expvals(features, weights):
    state = np.zeros(16, dtype=np.complex128)
    state[0] = 1.0
    for qubit, angle in enumerate(features):
        state = apply_single_qubit_gate(state, rx(float(angle)), qubit)
    for layer in range(weights.shape[0]):
        for qubit in range(4):
            phi, theta, omega = weights[layer, qubit]
            state = apply_single_qubit_gate(state, rz(float(phi)), qubit)
            state = apply_single_qubit_gate(state, ry(float(theta)), qubit)
            state = apply_single_qubit_gate(state, rz(float(omega)), qubit)
        for qubit in range(4):
            state = apply_cx(state, qubit, (qubit + 1) % 4)

    expvals = np.zeros(4, dtype=np.float64)
    probabilities = np.abs(state) ** 2
    for basis, probability in enumerate(probabilities):
        for qubit in range(4):
            expvals[qubit] += probability * (1.0 if not (basis & (1 << qubit)) else -1.0)
    return expvals


def counts_to_expvals(counts, shots):
    expvals = np.zeros(4, dtype=np.float64)
    for bitstring, count in counts.items():
        clean_bits = bitstring.replace(" ", "")
        for qubit in range(4):
            bit = clean_bits[::-1][qubit]
            expvals[qubit] += count * (1.0 if bit == "0" else -1.0)
    return expvals / float(shots)


def logits_to_action(expvals, actor_scale):
    logits = np.asarray(expvals, dtype=np.float64) * (1.0 + float(actor_scale))
    return int(np.argmax(logits)), logits


def format_vector(values):
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def confirm_ibm_submission(num_circuits, shots, backend_name, assume_yes):
    print(
        "IBM Quantum submission requested for "
        f"{num_circuits} circuits x {shots} shots on backend={backend_name!r}."
    )
    print(
        "WARNING: this account has only about 10 free QPU minutes. "
        "This script is inference-only and will not train on IBM hardware."
    )
    if num_circuits > 10:
        raise SystemExit("Refusing to submit more than 10 circuits.")
    if num_circuits > 5 and not assume_yes:
        answer = input("More than 5 circuits requested. Type SUBMIT to continue: ")
        if answer.strip() != "SUBMIT":
            raise SystemExit("IBM submission cancelled.")
    if not assume_yes:
        answer = input("Type SUBMIT to send these circuits to IBM Quantum: ")
        if answer.strip() != "SUBMIT":
            raise SystemExit("IBM submission cancelled.")


def get_runtime_service(instance):
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "IBM execution requires qiskit-ibm-runtime. Install with "
            "`python -m pip install qiskit qiskit-ibm-runtime`."
        ) from exc

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    if not token:
        raise SystemExit(
            "Environment variable 'IBM_QUANTUM_TOKEN' is not set. "
            "Set it in your shell before using --run-ibm."
        )
    return QiskitRuntimeService(token=token, instance=instance)


def backend_display_name(backend):
    name = getattr(backend, "name", None)
    return name() if callable(name) else str(name)


def list_ibm_backends(instance):
    service = get_runtime_service(instance)
    backends = service.backends(operational=True, simulator=False)
    print("Available operational IBM Quantum hardware backends:")
    for backend in sorted(backends, key=backend_display_name):
        status = backend.status()
        pending = getattr(status, "pending_jobs", "?")
        if hasattr(backend, "num_qubits"):
            qubits = backend.num_qubits
        elif hasattr(backend, "configuration"):
            qubits = getattr(backend.configuration(), "n_qubits", "?")
        else:
            qubits = "?"
        print(f"  {backend_display_name(backend)} | qubits={qubits} | pending_jobs={pending}")


def extract_sampler_counts(pub_result):
    data = getattr(pub_result, "data", None)
    if data is None:
        raise RuntimeError("Sampler result does not contain a data field.")

    # The circuit measures into ClassicalRegister("c"), so Runtime V2 commonly
    # exposes counts as result[i].data.c.get_counts().
    for register_name in ("c", "meas"):
        register = getattr(data, register_name, None)
        if register is not None and hasattr(register, "get_counts"):
            return register.get_counts()

    for register_name in dir(data):
        if register_name.startswith("_"):
            continue
        register = getattr(data, register_name)
        if hasattr(register, "get_counts"):
            return register.get_counts()

    if hasattr(data, "get_counts"):
        return data.get_counts()

    raise RuntimeError(
        "Could not find bit counts in SamplerV2 result data. "
        f"Available fields: {[name for name in dir(data) if not name.startswith('_')]}"
    )


def run_ibm_circuits(circuits, backend_name, shots, instance, assume_yes, dry_run=False):
    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "IBM execution requires qiskit-ibm-runtime. Install with "
            "`python -m pip install qiskit qiskit-ibm-runtime`."
        ) from exc

    service = get_runtime_service(instance)
    backend = service.backend(backend_name) if backend_name else service.least_busy(operational=True, simulator=False)
    selected_backend_name = backend_display_name(backend)
    transpiled = transpile(circuits, backend=backend, optimization_level=1)

    if dry_run:
        print(
            "IBM dry run OK: authenticated, selected backend "
            f"{selected_backend_name!r}, transpiled {len(transpiled)} circuits, "
            f"shots={shots}. No QPU job submitted."
        )
        return selected_backend_name, "", [], 0.0

    confirm_ibm_submission(len(circuits), shots, selected_backend_name, assume_yes)
    start_time = time.time()
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=shots)
    result = job.result()
    elapsed = time.time() - start_time
    counts = [extract_sampler_counts(result[index]) for index in range(len(circuits))]
    job_id = getattr(job, "job_id", "")
    job_id = job_id() if callable(job_id) else str(job_id)
    return selected_backend_name, job_id, counts, elapsed


def write_csv(rows, path):
    fieldnames = [
        "state_index",
        "simulator_output",
        "ibm_hardware_output",
        "selected_action",
        "selected_action_name",
        "ibm_selected_action",
        "ibm_selected_action_name",
        "action_agreement",
        "action_agreement_rate",
        "backend_name",
        "shots",
        "job_id",
        "circuit_execution_time_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path):
    lines = [
        "# Short-Trained QPPO Hardware Inference Feasibility",
        "",
        "This demo runs a short-trained small 4-qubit QPPO/QRL actor circuit for five fixed LunarLander states. It does not train on IBM hardware and should not be interpreted as quantum advantage.",
        "",
        "| State | Simulator output | IBM hardware output | Selected action | Agreement | Backend | Shots | Job ID | Circuit time (s) |",
        "| ---: | --- | --- | --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {state_index} | `{simulator_output}` | `{ibm_hardware_output}` | {selected_action_name} ({selected_action}) | {action_agreement} | {backend_name} | {shots} | {job_id} | {circuit_execution_time_sec} |".format(
                **row
            )
        )
    agreement_rates = [row["action_agreement_rate"] for row in rows if row["action_agreement_rate"]]
    if agreement_rates:
        lines.extend(["", f"Action agreement rate: **{agreement_rates[0]}**"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Short-trained QPPO hardware inference feasibility demo for LunarLander."
    )
    parser.add_argument("--model-path", default=None, help="Optional QPPO hybrid .cleanqrl_model checkpoint.")
    parser.add_argument("--run-ibm", action="store_true", help="Submit the same circuits to IBM Quantum hardware.")
    parser.add_argument(
        "--dry-run-ibm",
        action="store_true",
        help="Authenticate, select backend, and transpile circuits without submitting a QPU job.",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="List available operational IBM hardware backends and exit without submitting jobs.",
    )
    parser.add_argument("--backend", default=None, help="IBM backend name. Defaults to least busy non-simulator backend.")
    parser.add_argument("--instance", default=None, help="Optional IBM Runtime CRN/instance string.")
    parser.add_argument("--shots", type=int, default=100, help="IBM hardware shots. Default: 100.")
    parser.add_argument("--num-states", type=int, default=5, help="Number of fixed states to run, max 10.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts for <=10 circuits.")
    parser.add_argument("--csv", default="ibm_qppo_short_trained_inference_results.csv", help="Output CSV table.")
    parser.add_argument("--markdown", default="ibm_qppo_short_trained_inference_results.md", help="Output Markdown table.")
    args = parser.parse_args()

    if args.list_backends:
        list_ibm_backends(args.instance)
        return

    if args.num_states < 1:
        raise SystemExit("--num-states must be at least 1.")
    if args.num_states > 10:
        raise SystemExit("Refusing to prepare more than 10 circuits.")
    if args.shots > 100 and args.run_ibm and not args.yes:
        answer = input(f"Shots={args.shots} exceeds the low-shot default. Type SUBMIT to continue: ")
        if answer.strip() != "SUBMIT":
            raise SystemExit("IBM submission cancelled.")

    params = load_params_from_torch_model(args.model_path)
    states = FIXED_LUNARLANDER_STATES[: args.num_states]

    circuits_measured = []
    rows = []
    needs_qiskit_circuits = args.run_ibm or args.dry_run_ibm
    for state_index, state in enumerate(states):
        features = qppo_checkpoint_features(state, params)
        measured_circuit = (
            build_qppo_actor_circuit(features, params.weights, measure=True)
            if needs_qiskit_circuits
            else None
        )
        sim_expvals = numpy_simulator_expvals(features, params.weights)
        sim_action, _ = logits_to_action(sim_expvals, params.actor_scale)
        if measured_circuit is not None:
            circuits_measured.append(measured_circuit)
        rows.append(
            {
                "state_index": state_index,
                "simulator_output": format_vector(sim_expvals),
                "ibm_hardware_output": "",
                "selected_action": sim_action,
                "selected_action_name": ACTION_NAMES[sim_action],
                "ibm_selected_action": "",
                "ibm_selected_action_name": "",
                "action_agreement": "",
                "action_agreement_rate": "",
                "backend_name": "local_statevector",
                "shots": 0,
                "job_id": "",
                "circuit_execution_time_sec": "0.000",
            }
        )

    if args.run_ibm or args.dry_run_ibm:
        backend_name, job_id, counts_list, elapsed = run_ibm_circuits(
            circuits_measured,
            args.backend,
            args.shots,
            args.instance,
            args.yes,
            dry_run=args.dry_run_ibm and not args.run_ibm,
        )
        if args.dry_run_ibm and not args.run_ibm:
            for row in rows:
                row["backend_name"] = backend_name
                row["shots"] = args.shots
                row["job_id"] = "DRY_RUN_NO_JOB"
                row["circuit_execution_time_sec"] = "0.000"
        else:
            for row, counts in zip(rows, counts_list):
                ibm_expvals = counts_to_expvals(counts, args.shots)
                ibm_action, _ = logits_to_action(ibm_expvals, params.actor_scale)
                row["ibm_hardware_output"] = format_vector(ibm_expvals)
                row["ibm_selected_action"] = ibm_action
                row["ibm_selected_action_name"] = ACTION_NAMES[ibm_action]
                row["action_agreement"] = str(int(ibm_action) == int(row["selected_action"]))
                row["backend_name"] = backend_name
                row["shots"] = args.shots
                row["job_id"] = job_id
                row["circuit_execution_time_sec"] = f"{elapsed:.3f}"
            agreements = [row["action_agreement"] == "True" for row in rows]
            agreement_rate = sum(agreements) / len(agreements) if agreements else 0.0
            for row in rows:
                row["action_agreement_rate"] = f"{agreement_rate:.3f}"

    write_csv(rows, args.csv)
    write_markdown(rows, args.markdown)
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.markdown}")
    if not args.run_ibm:
        print("IBM hardware was not used. Add --run-ibm to submit the same circuits after confirmation.")


if __name__ == "__main__":
    main()
