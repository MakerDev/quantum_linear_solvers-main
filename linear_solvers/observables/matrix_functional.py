"""The matrix functional of the vector solution to the linear systems."""

from typing import Union, List
import numpy as np
from scipy.sparse import diags

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
from qiskit.primitives import Estimator

from .linear_system_observable import LinearSystemObservable


class MatrixFunctional(LinearSystemObservable):

    def __init__(self, main_diag: float, off_diag: float) -> None:
        """
        Args:
            main_diag: The main diagonal of the tridiagonal Toeplitz symmetric matrix to compute
                the functional.
            off_diag: The off diagonal of the tridiagonal Toeplitz symmetric matrix to compute
                the functional.
        """
        self._main_diag = main_diag
        self._off_diag = off_diag

    def observable(self, num_qubits: int) -> Union[SparsePauliOp, List[SparsePauliOp]]:
        """The observable operators.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of sums of Pauli strings.
        """
        zero_op = (Pauli('I') + Pauli('Z')) / 2
        one_op = (Pauli('I') - Pauli('Z')) / 2
        observables = []
        # First we measure the norm of x
        observables.append(SparsePauliOp.from_list([("I" * num_qubits, 1.0)]))
        for i in range(num_qubits):
            j = num_qubits - i - 1

            if i > 0:
                observables += [
                    SparsePauliOp.from_list([("I" * j + "Z" + "I" * i, 1.0)]),
                    SparsePauliOp.from_list([("I" * j + "Z" + "I" * i, -1.0)])
                ]
            else:
                observables += [
                    SparsePauliOp.from_list([("I" * j + "Z", 1.0)]),
                    SparsePauliOp.from_list([("I" * j + "Z", -1.0)])
                ]

        return observables

    def observable_circuit(
        self, num_qubits: int
    ) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The circuits to implement the matrix functional observable.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a list of QuantumCircuits.
        """
        qcs = []
        # Again, the first value in the list will correspond to the norm of x
        qcs.append(QuantumCircuit(num_qubits))
        for i in range(0, num_qubits):
            qc = QuantumCircuit(num_qubits)
            for j in range(0, i):
                qc.cx(i, j)
            qc.h(i)
            qcs += [qc, qc]

        return qcs

    def post_processing(
        self, solution: Union[float, List[float]], num_qubits: int, scaling: float = 1
    ) -> float:
        """Evaluates the matrix functional on the solution to the linear system.

        Args:
            solution: The list of probabilities calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            scaling: Scaling of the solution.

        Returns:
            The value of the absolute average.

        Raises:
            ValueError: If the input is not in the correct format.
        """
        if not isinstance(solution, list):
            raise ValueError("Solution probabilities must be given in list form.")

        # Calculate the value from the off-diagonal elements
        off_val = 0.0
        for i in range(1, len(solution), 2):
            off_val += (solution[i] - solution[i + 1]) / (scaling**2)
        main_val = solution[0] / (scaling**2)
        return np.real(self._main_diag * main_val + self._off_diag * off_val)

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

        matrix = diags(
            [self._off_diag, self._main_diag, self._off_diag],
            [-1, 0, 1],
            shape=(len(solution), len(solution)),
        ).toarray()

        return np.dot(solution.transpose(), np.dot(matrix, solution))
