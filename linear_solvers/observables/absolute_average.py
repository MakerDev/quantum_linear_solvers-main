from typing import Union, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from .linear_system_observable import LinearSystemObservable


class AbsoluteAverage(LinearSystemObservable):
    def observable(self, num_qubits: int) -> SparsePauliOp:
        """The observable operator.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a SparsePauliOp.
        """
        # Create a SparsePauliOp for the observable operator
        terms = [(Pauli('I').tensor(Pauli('Z')).to_label(), 1.0)]
        return SparsePauliOp.from_list(terms * num_qubits)

    def observable_circuit(
        self, num_qubits: int
    ) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The circuit implementing the absolute average observable.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a QuantumCircuit.
        """
        qc = QuantumCircuit(num_qubits)
        qc.h(qc.qubits)
        return qc

    def post_processing(
        self, solution: Union[float, List[float]], num_qubits: int, scaling: float = 1
    ) -> float:
        """Evaluates the absolute average on the solution to the linear system.

        Args:
            solution: The probability calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            scaling: Scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            ValueError: If the input is not in the correct format.
        """
        if isinstance(solution, list):
            if len(solution) == 1:
                solution = solution[0]
            else:
                raise ValueError(
                    "Solution probability must be given as a single value."
                )

        return np.real(np.sqrt(solution / (2**num_qubits)) / scaling)

    def evaluate_classically(
        self, solution: Union[np.ndarray, QuantumCircuit]
    ) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The solution to the system as a numpy array or the circuit that prepares it.

        Returns:
            The value of the observable.
        """
        # Check if it is QuantumCircuits and get the array from them
        if isinstance(solution, QuantumCircuit):
            solution = Statevector(solution).data
        return np.abs(np.mean(solution))
