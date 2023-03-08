from copy import deepcopy
import numpy as np

from qutip import qeye, tensor, destroy, basis
from qutip_qip.device import Model, SCQubitsModel
from qutip_qip.device.modelprocessor import ModelProcessor, _to_array
from qutip_qip.transpiler import to_chain_structure
# from ..compiler import SCQubitsCompiler
from qutip_qip.noise import ZZCrossTalk
from qutip_qip.operations import expand_operator, Gate
from qutip_qip.compiler import GateCompiler, Instruction

from itertools import product
from functools import reduce
from operator import mul

import warnings
import numpy as np
import pytest

import qutip
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations import Gate, gate_sequence_product
from qutip_qip.device import (
    DispersiveCavityQED, LinearSpinChain, CircularSpinChain, SCQubits
    )

from packaging.version import parse as parse_version
from qutip import Options

_tol = 3.e-2


class SCQubits2(ModelProcessor):
    def __init__(self, num_qubits, dims=None, zz_crosstalk=False, **params):
        if dims is None:
            dims = [3] * num_qubits
        model = SCQubitsModel2(
            num_qubits=num_qubits,
            dims=dims,
            zz_crosstalk=zz_crosstalk,
            **params,
        )
        super(SCQubits2, self).__init__(model=model)
        self.native_gates = ["RX", "RY", "CNOT"]
        self._default_compiler = SCQubitsCompiler2
        self.pulse_mode = "continuous"

    def topology_map(self, qc):
        return to_chain_structure(qc)

class SCQubitsModel2(SCQubitsModel):
    def __init__(self, num_qubits, dims=None, zz_crosstalk=False, **params):
        self.num_qubits = num_qubits
        self.dims = dims if dims is not None else [3] * num_qubits
        self.params = {
            "wq": np.array(
                ((0.5, 0.) * int(np.ceil(self.num_qubits / 2)))[
                    : self.num_qubits
                ]
            ),
            "alpha": [-0.3] * self.num_qubits,
            "g": [-0.3] * (self.num_qubits - 1),
            "omega_single": [0.03] * self.num_qubits,
        }
        self.params.update(deepcopy(params))
        self._compute_params()
        self._drift = []
        self._set_up_drift()
        self._controls = self._set_up_controls()
        self._noise = []
        if zz_crosstalk:
            self._noise.append(ZZCrossTalk(self.params))

    def _set_up_drift(self):
        for m in range(self.num_qubits):
            destroy_op = destroy(self.dims[m])
            coeff = 2 * np.pi * self.params["alpha"][m] / 2.0
            self._drift.append(
                (coeff * destroy_op.dag() ** 2 * destroy_op**2, [m])
            )
            self._drift.append(
                (2 * np.pi * self.params["wq"][m] * destroy_op.dag() * destroy_op, [m])
            )

    def _set_up_controls(self):
        """
        Setup the operators.
        We use 2π σ/2 as the single-qubit control Hamiltonian and
        -2πZX/4 as the two-qubit Hamiltonian.
        """
        num_qubits = self.num_qubits
        dims = self.dims
        controls = {}

        for m in range(num_qubits):
            destroy_op = destroy(dims[m])
            op = destroy_op + destroy_op.dag()
            controls["sx" + str(m)] = (2 * np.pi / 2 * op, [m])

        for m in range(num_qubits):
            destroy_op = destroy(dims[m])
            op = destroy_op * (-1.0j) + destroy_op.dag() * 1.0j
            controls["sy" + str(m)] = (2 * np.pi / 2 * op, [m])

        for m in range(num_qubits):
            destroy_op = destroy(dims[m])
            op = destroy_op.dag() * destroy_op
            controls["sz" + str(m)] = (2 * np.pi * op, [m])
        return controls

    def _compute_params(self):
        """
        Compute the dressed frequency and the interaction strength.
        """
        pass

    def get_control_latex(self):
        """
        Get the labels for each Hamiltonian.
        It is used in the method method :meth:`.Processor.plot_pulses`.
        It is a 2-d nested list, in the plot,
        a different color will be used for each sublist.
        """
        num_qubits = self.num_qubits
        labels = [
            {f"sx{n}": r"$\sigma_x" + f"^{n}$" for n in range(num_qubits)},
            {f"sy{n}": r"$\sigma_y" + f"^{n}$" for n in range(num_qubits)},
            {f"sz{n}": r"$\sigma_z" + f"^{n}$" for n in range(num_qubits)},
        ]
        label_zx = {}
        for m in range(num_qubits - 1):
            label_zx[f"zx{m}{m+1}"] = r"$ZX^{" + f"{m}{m+1}" + r"}$"
            label_zx[f"zx{m+1}{m}"] = r"$ZX^{" + f"{m+1}{m}" + r"}$"

        labels.append(label_zx)
        return labels


class SCQubitsCompiler2(GateCompiler):
    def __init__(self, num_qubits, params):
        super(SCQubitsCompiler2, self).__init__(num_qubits, params=params)
        self.gate_compiler.update(
            {
                "RY": self.ry_compiler,
                "RX": self.rx_compiler,
                # "CNOT": self.cnot_compiler,
            }
        )
        self.args = {  # Default configuration
            "shape": "hann",
            "num_samples": 1001,
            "params": self.params,
            # "DRAG": True,
        }

    def _rotation_compiler(self, gate, op_label, param_label, args):
        """
        Single qubit rotation compiler.

        Parameters
        ----------
        gate : :obj:`~.operations.Gate`:
            The quantum gate to be compiled.
        op_label : str
            Label of the corresponding control Hamiltonian.
        param_label : str
            Label of the hardware parameters saved in
            :obj:`GateCompiler.params`.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        targets = gate.targets
        coeff, tlist = self.generate_pulse_shape(
            args["shape"],
            args["num_samples"],
            maximum=self.params[param_label][targets[0]],
            area=gate.arg_value / 2.0 / np.pi,
        )
        # if args["DRAG"]:
        #     pulse_info = self._drag_pulse(op_label, coeff, tlist, targets[0])
        f = 2 * np.pi * self.params["wq"][targets[0]]
        if op_label == "sx":
            pulse_info = [
                ("sx" + str(targets[0]), coeff * np.cos(f * tlist)),
                ("sy" + str(targets[0]), -coeff * np.sin(f * tlist)),
                ]
        elif op_label == "sy":
            pulse_info = [
                ("sx" + str(targets[0]), coeff * np.cos(f * tlist + np.pi/2)),
                ("sy" + str(targets[0]), -coeff * np.sin(f * tlist + np.pi/2)),
                ]
            # pulse_info = [(op_label + str(targets[0]), coeff)]
        else:
            pulse_info = [(op_label + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def _drag_pulse(self, op_label, coeff, tlist, target):
        dt_coeff = np.gradient(coeff, tlist[1] - tlist[0]) / 2 / np.pi
        # Y-DRAG
        alpha = self.params["alpha"][target]
        y_drag = -dt_coeff / alpha
        # Z-DRAG
        z_drag = -(coeff**2) / alpha + (np.sqrt(2) ** 2 * coeff**2) / (
            4 * alpha
        )
        # X-DRAG
        coeff += -(coeff**3 / (4 * alpha**2))

        pulse_info = [
            (op_label + str(target), coeff),
            ("sz" + str(target), z_drag),
        ]
        if op_label == "sx":
            pulse_info.append(("sy" + str(target), y_drag))
        elif op_label == "sy":
            pulse_info.append(("sx" + str(target), -y_drag))
        return pulse_info

    def ry_compiler(self, gate, args):
        return self._rotation_compiler(gate, "sy", "omega_single", args)

    def rx_compiler(self, gate, args):
        return self._rotation_compiler(gate, "sx", "omega_single", args)

    def compile(self, circuit, schedule_mode=None, args=None):
        compiled_tlist_map, compiled_coeffs_map = super().compile(circuit, schedule_mode=None, args=None)
        return compiled_tlist_map, compiled_coeffs_map


def _ket_expaned_dims(qubit_state, expanded_dims):
    all_qubit_basis = list(product([0, 1], repeat=len(expanded_dims)))
    old_dims = qubit_state.dims[0]
    expanded_qubit_state = np.zeros(
        reduce(mul, expanded_dims, 1), dtype=np.complex128)
    for basis_state in all_qubit_basis:
        old_ind = np.ravel_multi_index(basis_state, old_dims)
        new_ind = np.ravel_multi_index(basis_state, expanded_dims)
        expanded_qubit_state[new_ind] = qubit_state[old_ind, 0]
    expanded_qubit_state.reshape((reduce(mul, expanded_dims, 1), 1))
    return qutip.Qobj(
        expanded_qubit_state, dims=[expanded_dims, [1]*len(expanded_dims)])


def get_phase_frame(t, eigenvalues, dims):
    phase_frame_U = np.diag(np.exp((1.0j * eigenvalues * t)))
    return qutip.Qobj(phase_frame_U, dims=dims)

# X gate performance
num_qubits = 1
circuit = QubitCircuit(num_qubits)
gates = [Gate("X", targets=[0])]
for gate in gates:
    circuit.add_gate(gate)
device = SCQubits2(num_qubits)
device.load_circuit(circuit)

H0 = device.get_qobjevo(noisy=True)[0](0)
eigenvalues = np.diag(H0.full())
phase_frame_U = get_phase_frame(device.get_full_tlist()[-1], eigenvalues, H0.dims)

qu, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(qu.to_list(), c_op_list=c_ops, t=qu.tlist)

numeric_gate = qutip.Qobj((phase_frame_U * U_list[-1])[:2,:2])
print("X gate fidelity",
    qutip.average_gate_fidelity(numeric_gate, qutip.sigmax())
    )


# Y gate performance
num_qubits = 1
circuit = QubitCircuit(num_qubits)
gates = [Gate("Y", targets=[0])]
for gate in gates:
    circuit.add_gate(gate)
device = SCQubits2(num_qubits)
device.load_circuit(circuit)

H0 = device.get_qobjevo(noisy=True)[0](0)
eigenvalues = np.diag(H0.full())
phase_frame_U = get_phase_frame(device.get_full_tlist()[-1], eigenvalues, H0.dims)

qu, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(qu.to_list(), c_op_list=c_ops, t=qu.tlist)

numeric_gate = qutip.Qobj((phase_frame_U * U_list[-1])[:2,:2])
print("Y gate fidelity",
    qutip.average_gate_fidelity(numeric_gate, qutip.sigmay())
    )


# XY gate performance
num_qubits = 1
circuit = QubitCircuit(num_qubits)
gates = [Gate("X", targets=[0]), Gate("Y", targets=[0])]
for gate in gates:
    circuit.add_gate(gate)
device = SCQubits2(num_qubits)
device.load_circuit(circuit)

H0 = device.get_qobjevo(noisy=True)[0](0)
eigenvalues = np.diag(H0.full())
phase_frame_U = get_phase_frame(device.get_full_tlist()[-1], eigenvalues, H0.dims)

qu, c_ops = device.get_qobjevo(noisy=True)
U_list = qutip.propagator(qu.to_list(), c_op_list=c_ops, t=qu.tlist)

numeric_gate = qutip.Qobj((phase_frame_U * U_list[-1])[:2,:2])
print("X gate fidelity",
    qutip.average_gate_fidelity(numeric_gate, qutip.qeye(2))
    )
