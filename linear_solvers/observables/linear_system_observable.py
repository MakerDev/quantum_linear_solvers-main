"""An abstract class for linear systems solvers in Qiskit."""

from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


class LinearSystemObservable(ABC):
    """An abstract class for linear system observables in Qiskit."""

    @abstractmethod
    def observable(self, num_qubits: int) -> Union[SparsePauliOp, List[SparsePauliOp]]:
        """The observable operator.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a sum of Pauli strings.
        """
        raise NotImplementedError

    @abstractmethod
    def observable_circuit(
        self, num_qubits: int
    ) -> Union[QuantumCircuit, List[QuantumCircuit]]:
        """The circuit implementing the observable.

        Args:
            num_qubits: The number of qubits on which the observable will be applied.

        Returns:
            The observable as a QuantumCircuit.
        """
        raise NotImplementedError

    @abstractmethod
    def post_processing(
        self, solution: Union[float, List[float]], num_qubits: int, scaling: float = 1
    ) -> float:
        """Evaluates the given observable on the solution to the linear system.

        Args:
            solution: The probability calculated from the circuit and the observable.
            num_qubits: The number of qubits where the observable was applied.
            scaling: Scaling of the solution.

        Returns:
            The value of the observable.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_classically(
        self, solution: Union[np.ndarray, QuantumCircuit]
    ) -> float:
        """Calculates the analytical value of the given observable from the solution vector to the
         linear system.

        Args:
            solution: The solution to the system as a numpy array or the circuit that prepares it.

        Returns:
            The value of the observable.
        """
        raise NotImplementedError
