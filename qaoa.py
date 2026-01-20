# template: config_1n_1g_1mpi_noomp
# module: py-qiskit/2.3.0
# srun: srun -N 1 -n 1 -c 72 --gpus-per-task=1

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2


def ring_edges(n):
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
    return edges


def pauli_zz_string(n, i, j):
    # Pauli string ordering is qubit (n-1) down to qubit 0.
    s = ["I"] * n
    s[n - 1 - i] = "Z"
    s[n - 1 - j] = "Z"
    return "".join(s)


def build_maxcut_cost_operator(n, edges):
    # C = sum_{(i,j) in E} (I - Z_i Z_j) / 2
    paulis = []
    coeffs = []

    # 0.5 * |E| * I
    paulis.append("I" * n)
    coeffs.append(0.5 * float(len(edges)))

    # -0.5 * sum Z_i Z_j
    for (i, j) in edges:
        paulis.append(pauli_zz_string(n, i, j))
        coeffs.append(-0.5)

    return SparsePauliOp(paulis, coeffs=np.array(coeffs, dtype=np.float64))


def build_qaoa_maxcut(n, p, edges):
    gammas = [Parameter("gamma_" + str(k)) for k in range(p)]
    betas = [Parameter("beta_" + str(k)) for k in range(p)]

    qc = QuantumCircuit(n)
    qc.h(range(n))

    for k in range(p):
        g = gammas[k]
        b = betas[k]

        # Cost layer: exp(-i * g * Z_i Z_j) using CX-RZ-CX
        for (i, j) in edges:
            qc.cx(i, j)
            qc.rz(2 * g, j)
            qc.cx(i, j)

        # Mixer layer: exp(-i * b * X) == RX(2b)
        qc.rx(2 * b, range(n))

    return qc, gammas, betas


def main():
    n = 31
    p = 2

    edges = ring_edges(n)
    cost_op = build_maxcut_cost_operator(n, edges)
    qc, gammas, betas = build_qaoa_maxcut(n, p, edges)

    # Parameter vector x = [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]
    params = gammas + betas
    param_index = {params[i]: i for i in range(len(params))}
    ordered_params = list(qc.parameters)

    # EstimatorV2: precision controls sampling noise; 0.0 means "no added noise"
    # for Aer EstimatorV2 (exact expectation values). :contentReference[oaicite:3]{index=3}
    estimator = AerEstimatorV2(
        options={
            "default_precision": 0.0,
            "backend_options": {
                "device": "GPU",
                "cuStateVec_enable": True,
            },
        }
    )

    def objective(x):
        # EstimatorV2 expects parameter_values aligned with circuit.parameters
        values = []
        for p_ in ordered_params:
            values.append(float(x[param_index[p_]]))

        # pubs are (circuit, observables, parameter_values)
        job = estimator.run([(qc, cost_op, [values])], precision=0.0)
        job_result = job.result()
        pub_result = job_result[0]
        exp_c = pub_result.data.evs[-1]

        # SciPy minimizes; we maximize <C>
        return -exp_c

    x0 = np.concatenate([
        np.full(p, 0.7),  # gammas
        np.full(p, 0.3),  # betas
    ])

    res = minimize(
        objective,
        x0,
        method="COBYLA",
        options={"maxiter": 60, "disp": True},
    )

    best_x = res.x
    best_c = -res.fun

    print("Best expected cut value:", best_c)
    for k in range(p):
        print("gamma_" + str(k) + " =", best_x[k])
    for k in range(p):
        print("beta_" + str(k) + " =", best_x[p + k])


if __name__ == "__main__":
    main()

