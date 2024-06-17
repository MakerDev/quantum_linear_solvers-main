from typing import Optional, Union, List, Callable, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import PhaseEstimation
from qiskit.circuit.library.arithmetic.piecewise_chebyshev import PiecewiseChebyshev
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.primitives import Sampler, Estimator

from .linear_solver import LinearSolver, LinearSolverResult
from .matrices.numpy_matrix import NumPyMatrix
from .observables.linear_system_observable import LinearSystemObservable
from qiskit.primitives import Estimator


class HHL(LinearSolver):
    def __init__(
        self,
        epsilon: float = 1e-2,
        expectation: Optional[Estimator] = None,
        quantum_instance: Optional[Union[Sampler, Estimator]] = None,
    ) -> None:
        super().__init__()

        self._epsilon = epsilon
        self._epsilon_r = epsilon / 3
        self._epsilon_s = epsilon / 3
        self._epsilon_a = epsilon / 6

        self._scaling = None
        self._sampler = None
        self.quantum_instance = quantum_instance

        self._expectation = expectation
        self._exact_reciprocal = True
        self.scaling = 1

    @property
    def quantum_instance(self) -> Optional[Sampler]:
        return None if self._sampler is None else self._sampler.quantum_instance

    @quantum_instance.setter
    def quantum_instance(
        self, quantum_instance: Optional[Union[Sampler, Estimator]]
    ) -> None:
        if quantum_instance is not None:
            self._sampler = quantum_instance
        else:
            self._sampler = None

    @property
    def scaling(self) -> float:
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: float) -> None:
        self._scaling = scaling

    @property
    def expectation(self) -> Estimator:
        return self._expectation

    @expectation.setter
    def expectation(self, expectation: Estimator) -> None:
        self._expectation = expectation

    def _get_delta(self, n_l: int, lambda_min: float, lambda_max: float) -> float:
        formatstr = "#0" + str(n_l + 2) + "b"
        lambda_min_tilde = np.abs(lambda_min * (2**n_l - 1) / lambda_max)
        if np.abs(lambda_min_tilde - 1) < 1e-7:
            lambda_min_tilde = 1
        binstr = format(int(lambda_min_tilde), formatstr)[2::]
        lamb_min_rep = 0
        for i, char in enumerate(binstr):
            lamb_min_rep += int(char) / (2 ** (i + 1))

        return lamb_min_rep
    
    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        na = qc.num_ancillas

        # Create zero_op and one_op using SparsePauliOp
        zero_op = SparsePauliOp.from_list([("I", 1.0), ("Z", 1.0)]) / 2
        one_op = SparsePauliOp.from_list([("I", 1.0), ("Z", -1.0)]) / 2

        # Create the observable for the norm calculation
        observable = one_op.tensor(zero_op.tensor(SparsePauliOp("I" * nb)))

        # Evaluate the norm
        state = Statevector.from_instruction(qc)
        norm_2 = state.expectation_value(observable)

        return np.real(np.sqrt(norm_2) / self.scaling)


    def _calculate_observable(
        self,
        solution: QuantumCircuit,
        ls_observable: Optional[LinearSystemObservable] = None,
        observable_circuit: Optional[QuantumCircuit] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]], int, float], float]
        ] = None,
    ) -> Tuple[float, Union[complex, List[complex]]]:
        """Calculates the value of the observable(s) given.

        Args:
            solution: The quantum circuit preparing the solution x to the system.
            ls_observable: Information to be extracted from the solution.
            observable_circuit: Circuit to be applied to the solution to extract information.
            post_processing: Function to compute the value of the observable.

        Returns:
            The value of the observable(s) and the circuit results before post-processing as a tuple.
        """
        nb = solution.qregs[0].size
        nl = solution.qregs[1].size
        na = solution.num_ancillas
        num_qubits = solution.num_qubits

        if ls_observable is not None:
            observable_circuit = ls_observable.observable_circuit(nb)
            post_processing = ls_observable.post_processing

            if isinstance(ls_observable, LinearSystemObservable):
                observable = ls_observable.observable(nb)
        else:
            observable = SparsePauliOp.from_list([("I" * num_qubits, 1.0)])

        # Create zero_op and one_op using SparsePauliOp
        zero_op = SparsePauliOp.from_sparse_list([("I" * num_qubits, "Z" * num_qubits)], [1.0, 1.0])
        one_op = SparsePauliOp.from_sparse_list([("I" * num_qubits, "Z" * num_qubits)], [1.0, -1.0])

        if not isinstance(observable_circuit, list):
            observable_circuit = [observable_circuit]
            observable = [observable]

        expectations = []
        for circ, obs in zip(observable_circuit, observable):
            circuit = QuantumCircuit(num_qubits)
            circuit.append(solution, circuit.qubits)
            circuit.append(circ, range(nb))

            # Manually construct the tensor product
            tensor_product = one_op
            for _ in range(nl + na):
                tensor_product = tensor_product.tensor(zero_op)
            tensor_product = tensor_product.tensor(obs)

            # Use Estimator to compute the expectation value
            estimator = Estimator()
            job = estimator.run([(circuit, tensor_product)])
            result = job.result()
            expectation_value = result.values[0]
            expectations.append(expectation_value)

        if len(expectations) == 1:
            expectations = expectations[0]

        result = post_processing(expectations, nb, self.scaling)

        return result, expectations


    def construct_circuit(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        neg_vals: Optional[bool] = True,
    ) -> QuantumCircuit:
        if isinstance(vector, QuantumCircuit):
            nb = vector.num_qubits
            vector_circuit = vector
        elif isinstance(vector, (list, np.ndarray)):
            if isinstance(vector, list):
                vector = np.array(vector)
            nb = int(np.log2(len(vector)))
            vector_circuit = QuantumCircuit(nb)
            vector_circuit.initialize(
                vector / np.linalg.norm(vector), list(range(nb)), None
            )

        nf = 1

        if isinstance(matrix, QuantumCircuit):
            matrix_circuit = matrix
        elif isinstance(matrix, (list, np.ndarray)):
            if isinstance(matrix, list):
                matrix = np.array(matrix)

            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Input matrix must be square!")
            if np.log2(matrix.shape[0]) % 1 != 0:
                raise ValueError("Input matrix dimension must be 2^n!")
            if not np.allclose(matrix, matrix.conj().T):
                raise ValueError("Input matrix must be hermitian!")
            if matrix.shape[0] != 2**vector_circuit.num_qubits:
                raise ValueError(
                    "Input vector dimension does not match input "
                    "matrix dimension! Vector dimension: "
                    + str(vector_circuit.num_qubits)
                    + ". Matrix dimension: "
                    + str(matrix.shape[0])
                )
            matrix_circuit = NumPyMatrix(matrix, evolution_time=2 * np.pi)
        else:
            raise ValueError(f"Invalid type for matrix: {type(matrix)}.")

        if hasattr(matrix_circuit, "tolerance"):
            matrix_circuit.tolerance = self._epsilon_a

        if (
            hasattr(matrix_circuit, "condition_bounds")
            and matrix_circuit.condition_bounds() is not None
        ):
            kappa = matrix_circuit.condition_bounds()[1]
        else:
            kappa = 1
        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + neg_vals

        if (
            hasattr(matrix_circuit, "eigs_bounds")
            and matrix_circuit.eigs_bounds() is not None
        ):
            lambda_min, lambda_max = matrix_circuit.eigs_bounds()
            delta = self._get_delta(nl - neg_vals, lambda_min, lambda_max)
            matrix_circuit.evolution_time = (
                2 * np.pi * delta / lambda_min / (2**neg_vals)
            )
            self.scaling = lambda_min
        else:
            delta = 1 / (2**nl)
            print("The solution will be calculated up to a scaling factor.")

        if self._exact_reciprocal:
            reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=neg_vals)
            na = matrix_circuit.num_ancillas
        else:
            num_values = 2**nl
            constant = delta
            a = int(round(num_values ** (2 / 3)))
            r = 2 * constant / a + np.sqrt(np.abs(1 - (2 * constant / a) ** 2))
            degree = min(
                nb,
                int(
                    np.log(
                        1
                        + (
                            16.23
                            * np.sqrt(np.log(r) ** 2 + (np.pi / 2) ** 2)
                            * kappa
                            * (2 * kappa - self._epsilon_r)
                        )
                        / self._epsilon_r
                    )
                ),
            )
            num_intervals = int(np.ceil(np.log((num_values - 1) / a) / np.log(5)))
            breakpoints = []
            for i in range(0, num_intervals):
                breakpoints.append(a * (5**i))
                if i == num_intervals - 1:
                    breakpoints.append(num_values - 1)

            reciprocal_circuit = PiecewiseChebyshev(
                lambda x: np.arcsin(constant / x), degree, breakpoints, nl
            )
            na = max(matrix_circuit.num_ancillas, reciprocal_circuit.num_ancillas)

        qb = QuantumRegister(nb)
        ql = QuantumRegister(nl)
        if na > 0:
            qa = AncillaRegister(na)
        qf = QuantumRegister(nf)

        if na > 0:
            qc = QuantumCircuit(qb, ql, qa, qf)
        else:
            qc = QuantumCircuit(qb, ql, qf)

        qc.append(vector_circuit, qb[:])
        phase_estimation = PhaseEstimation(nl, matrix_circuit)
        if na > 0:
            qc.append(
                phase_estimation, ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas]
            )
        else:
            qc.append(phase_estimation, ql[:] + qb[:])
        if self._exact_reciprocal:
            qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        else:
            qc.append(
                reciprocal_circuit.to_instruction(),
                ql[:] + [qf[0]] + qa[: reciprocal_circuit.num_ancillas],
            )
        if na > 0:
            qc.append(
                phase_estimation.inverse(),
                ql[:] + qb[:] + qa[: matrix_circuit.num_ancillas],
            )
        else:
            qc.append(phase_estimation.inverse(), ql[:] + qb[:])
        return qc

    def solve(
        self,
        matrix: Union[List, np.ndarray, QuantumCircuit],
        vector: Union[List, np.ndarray, QuantumCircuit],
        observable: Optional[
            Union[
                LinearSystemObservable,
                List[LinearSystemObservable],
            ]
        ] = None,
        observable_circuit: Optional[
            Union[QuantumCircuit, List[QuantumCircuit]]
        ] = None,
        post_processing: Optional[
            Callable[[Union[float, List[float]], int, float], float]
        ] = None,
    ) -> LinearSolverResult:
        if observable is not None:
            if observable_circuit is not None or post_processing is not None:
                raise ValueError(
                    "If observable is passed, observable_circuit and post_processing cannot be set."
                )

        solution = LinearSolverResult()
        solution.state = self.construct_circuit(matrix, vector)
        solution.euclidean_norm = self._calculate_norm(solution.state)

        if isinstance(observable, List):
            observable_all, circuit_results_all = [], []
            for obs in observable:
                obs_i, circ_results_i = self._calculate_observable(
                    solution.state, obs, observable_circuit, post_processing
                )
                observable_all.append(obs_i)
                circuit_results_all.append(circ_results_i)
            solution.observable = observable_all
            solution.circuit_results = circuit_results_all
        elif observable is not None or observable_circuit is not None:
            solution.observable, solution.circuit_results = self._calculate_observable(
                solution.state, observable, observable_circuit, post_processing
            )

        return solution
