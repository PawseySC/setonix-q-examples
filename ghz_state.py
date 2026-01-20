from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

n = 32
shots = 4096

# Build a 32-qubit GHZ state
qc = QuantumCircuit(n, n)
qc.h(0)
for i in range(1, n):
    qc.cx(0, i)
qc.measure(range(n), range(n))

sim = AerSimulator(method = "statevector", device="GPU", cuStateVec_enable=True)
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

print("Measurement results:")
for bitstring in counts:
    print(f"{bitstring} : {counts[bitstring]}")
