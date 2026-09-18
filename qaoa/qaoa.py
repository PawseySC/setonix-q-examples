"""
QAOA (Quantum Approximate Optimization Algorithm) for MaxCut Problem
=====================================================================

This script implements QAOA to solve the MaxCut problem on a ring graph.

MaxCut Problem:
    Given a graph with nodes and edges, find a partition of nodes into two sets
    that maximizes the number of edges crossing between the sets.

QAOA Overview:
    QAOA is a hybrid quantum-classical algorithm that:
    1. Prepares a quantum state using parameterized "cost" and "mixer" layers
    2. Measures the expectation value of a cost Hamiltonian
    3. Uses a classical optimizer to tune parameters and maximize the cost

For more information, see: https://arxiv.org/abs/1411.4028
"""

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2


# =============================================================================
# Graph Definition
# =============================================================================


def ring_edges(n):
    """
    Create edges for a ring (cycle) graph with n nodes.

    A ring graph connects each node i to node (i+1) mod n, forming a closed loop.
    For example, with n=4: 0-1-2-3-0

    Args:
        n: Number of nodes in the ring graph

    Returns:
        List of tuples representing edges, e.g., [(0,1), (1,2), (2,3), (3,0)]
    """
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
    return edges


# =============================================================================
# Cost Hamiltonian Construction
# =============================================================================


def pauli_zz_string(n, i, j):
    """
    Build a Pauli string with Z operators on qubits i and j, identity elsewhere.

    In Qiskit's Pauli string convention, the rightmost character corresponds to
    qubit 0, and the leftmost to qubit (n-1). This function handles that reversal.

    Args:
        n: Total number of qubits
        i: First qubit index for Z operator
        j: Second qubit index for Z operator

    Returns:
        String like "IIZZI" representing Z_i * Z_j
    """
    # Pauli string ordering is qubit (n-1) down to qubit 0 (little-endian).
    s = ["I"] * n
    s[n - 1 - i] = "Z"
    s[n - 1 - j] = "Z"
    return "".join(s)


def build_maxcut_cost_operator(n, edges):
    """
    Build the MaxCut cost Hamiltonian as a SparsePauliOp.

    The MaxCut cost function counts edges where endpoints are in different sets:
        C = sum_{(i,j) in edges} (1 - Z_i * Z_j) / 2

    When Z_i and Z_j have opposite signs (nodes in different sets), the term
    contributes +1. When they have the same sign, it contributes 0.

    This expands to: C = |E|/2 * I - (1/2) * sum_{(i,j)} Z_i Z_j

    Args:
        n: Number of qubits (nodes in the graph)
        edges: List of edge tuples

    Returns:
        SparsePauliOp representing the cost Hamiltonian
    """
    paulis = []
    coeffs = []

    # Constant term: 0.5 * |E| * I (Identity on all qubits)
    paulis.append("I" * n)
    coeffs.append(0.5 * float(len(edges)))

    # Z_i Z_j terms with coefficient -0.5 for each edge
    for (i, j) in edges:
        paulis.append(pauli_zz_string(n, i, j))
        coeffs.append(-0.5)

    return SparsePauliOp(paulis, coeffs=np.array(coeffs, dtype=np.float64))


# =============================================================================
# QAOA Circuit Construction
# =============================================================================


def build_qaoa_maxcut(n, p, edges):
    """
    Build a parameterized QAOA circuit for MaxCut.

    The QAOA circuit has p layers (also called "rounds" or "depth").
    Each layer consists of:
        1. Cost layer: Applies exp(-i * gamma * Z_i Z_j) for each edge
        2. Mixer layer: Applies exp(-i * beta * X) to each qubit

    The circuit starts with all qubits in the |+⟩ state (equal superposition).

    More layers (higher p) generally give better approximation ratios but
    increase circuit depth and optimization difficulty.

    Args:
        n: Number of qubits
        p: Number of QAOA layers (circuit depth parameter)
        edges: List of graph edges

    Returns:
        Tuple of (QuantumCircuit, gamma_parameters, beta_parameters)
    """
    # Create p gamma and p beta parameters (one pair per layer)
    gammas = [Parameter("gamma_" + str(k)) for k in range(p)]
    betas = [Parameter("beta_" + str(k)) for k in range(p)]

    qc = QuantumCircuit(n)

    # Initialize all qubits in |+⟩ state (equal superposition)
    # This is the standard QAOA initial state
    qc.h(range(n))

    # Apply p layers of cost and mixer unitaries
    for k in range(p):
        g = gammas[k]
        b = betas[k]

        # Cost layer: exp(-i * gamma * Z_i Z_j) for each edge
        # Implemented using the identity: exp(-i*theta*ZZ) = CX . RZ(2*theta) . CX
        # This entangles qubits connected by edges in the graph
        for (i, j) in edges:
            qc.cx(i, j)  # CNOT: entangle qubits i and j
            qc.rz(2 * g, j)  # Rotate qubit j by 2*gamma around Z-axis
            qc.cx(i, j)  # CNOT: undo entanglement (leaves phase)

        # Mixer layer: exp(-i * beta * X) on each qubit
        # Equivalent to RX(2*beta) rotation
        # This "mixes" the computational basis states
        qc.rx(2 * b, range(n))

    return qc, gammas, betas


# =============================================================================
# Main QAOA Optimization Loop
# =============================================================================


def main():
    """
    Run QAOA optimization for MaxCut on a ring graph.

    This demonstrates:
        1. Building the problem (ring graph)
        2. Constructing the QAOA circuit
        3. Setting up the quantum estimator (GPU-accelerated via cuStateVec)
        4. Running classical optimization to find optimal parameters
    """
    # Problem configuration
    n = 31  # Number of qubits (nodes in the graph)
    p = 2  # Number of QAOA layers (higher p = better approximation, deeper circuit)

    # Build the MaxCut problem components
    edges = ring_edges(n)  # Create ring graph edges
    cost_op = build_maxcut_cost_operator(n, edges)  # Cost Hamiltonian
    qc, gammas, betas = build_qaoa_maxcut(n, p, edges)  # QAOA circuit

    # Set up parameter mapping
    # Parameter vector x = [gamma_0, gamma_1, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
    params = gammas + betas
    param_index = {params[i]: i for i in range(len(params))}

    # Get the order that Qiskit stores parameters internally
    # (may differ from our creation order)
    ordered_params = list(qc.parameters)

    # =========================================================================
    # Quantum Estimator Setup (GPU-accelerated)
    # =========================================================================
    # EstimatorV2 computes expectation values <psi|H|psi>
    # precision=0.0 means exact statevector simulation (no shot noise)
    # For real hardware or shot-based simulation, use precision > 0
    estimator = AerEstimatorV2(
        options={
            "default_precision": 0.0,  # Exact expectation values (no sampling noise)
            "backend_options": {
                "device": "GPU",  # Use GPU acceleration
                "cuStateVec_enable": True,  # Enable NVIDIA cuStateVec library
            },
        }
    )

    # =========================================================================
    # Objective Function for Classical Optimizer
    # =========================================================================
    def objective(x):
        """
        Compute the negative expectation value of the cost Hamiltonian.

        The classical optimizer minimizes this function. Since we want to
        maximize the expected cut value, we return the negative.

        Args:
            x: Array of parameters [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]

        Returns:
            Negative expectation value (for minimization)
        """
        # Map optimizer's x vector to Qiskit's parameter ordering
        values = []
        for p_ in ordered_params:
            values.append(float(x[param_index[p_]]))

        # Create a "pub" (primitive unified bloc): (circuit, observable, parameter_values)
        # This is the EstimatorV2 input format
        job = estimator.run([(qc, cost_op, [values])], precision=0.0)
        job_result = job.result()
        pub_result = job_result[0]

        # Extract the expectation value from the result
        # evs contains expectation values; we take the last one
        exp_c = pub_result.data.evs[-1]

        # Return negative because scipy minimizes, but we want to maximize <C>
        return -exp_c

    # =========================================================================
    # Classical Optimization
    # =========================================================================
    # Initial parameter guess
    # gamma values around 0.7 and beta values around 0.3 are reasonable starting points
    x0 = np.concatenate([
        np.full(p, 0.7),  # Initial gamma values
        np.full(p, 0.3),  # Initial beta values
    ])

    # Run the classical optimizer
    # COBYLA is a gradient-free method suitable for noisy objective functions
    print("Starting QAOA optimization...")
    print(f"Graph: {n}-node ring, QAOA depth: p={p}")
    print(f"Total parameters: {2*p} (p gammas + p betas)")
    print("-" * 50)

    res = minimize(
        objective,
        x0,
        method="COBYLA",  # Constrained Optimization BY Linear Approximation
        options={"maxiter": 60, "disp": True},  # Display progress
    )

    # =========================================================================
    # Results
    # =========================================================================
    best_x = res.x  # Optimal parameters found
    best_c = -res.fun  # Best expected cut value (negate back)

    print("-" * 50)
    print("Optimization complete!")
    print(f"Best expected cut value: {best_c:.4f}")
    print(f"Theoretical maximum for ring graph: {n if n % 2 == 0 else n - 1}")
    print("\nOptimal parameters:")
    for k in range(p):
        print(f"  gamma_{k} = {best_x[k]:.4f}")
    for k in range(p):
        print(f"  beta_{k} = {best_x[p + k]:.4f}")


if __name__ == "__main__":
    main()
