from functools import partial
import numpy as np
import scipy

from ..circuit import QubitCircuit
from ..operations import Gate
from ..compiler import GateCompiler, Instruction


__all__ = ['CavityQEDCompiler']


class CavityQEDCompiler(GateCompiler):
    """
    Compiler for :obj:`.DispersiveCavityQED`.
    Compiled pulse strength is in the unit of GHz.

    Supported native gates: "RX", "RY", "RZ", "ISWAP", "SQRTISWAP",
    "GLOBALPHASE".

    Default configuration (see :obj:`.GateCompiler.args` and
    :obj:`.GateCompiler.compile`):

        +-----------------+--------------------+
        | key             | value              |
        +=================+====================+
        | ``shape``       | ``rectangular``    |
        +-----------------+--------------------+
        |``params``       | Hardware Parameters|
        +-----------------+--------------------+

    Parameters
    ----------
    num_qubits: int
        The number of qubits in the system.

    params: dict
        A Python dictionary contains the name and the value of the parameters.
        See :meth:`.DispersiveCavityQED.set_up_params` for the definition.

    global_phase: float, optional
        Record of the global phase change and will be returned.

    Attributes
    ----------
    gate_compiler: dict
        The Python dictionary in the form of {gate_name: decompose_function}.
        It saves the decomposition scheme for each gate.
    """
    def __init__(
            self, num_qubits, params, global_phase=0.,
            pulse_dict=None, N=None):
        super(CavityQEDCompiler, self).__init__(
            num_qubits, params=params, pulse_dict=pulse_dict, N=N)
        self.gate_compiler.update({
            "ISWAP": self.iswap_compiler,
            "SQRTISWAP": self.sqrtiswap_compiler,
            "RZ": self.rz_compiler,
            "RX": self.rx_compiler,
            "GLOBALPHASE": self.globalphase_compiler
            })
        self.wq = np.sqrt(self.params["eps"]**2 + self.params["delta"]**2)
        self.Delta = self.wq - self.params["w0"]
        self.global_phase = global_phase

    def _rotation_compiler(self, gate, op_label, param_label, args):
        """
        Single qubit rotation compiler.

        Parameters
        ----------
        gate : :obj:`.Gate`:
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
            args["shape"], args["num_samples"],
            maximum=self.params[param_label][targets[0]],
            # The operator is Pauli Z/X/Y, without 1/2.
            area=gate.arg_value / 2. / np.pi * 0.5)
        pulse_info = [(op_label + str(targets[0]), coeff)]
        return [Instruction(gate, tlist, pulse_info)]

    def rz_compiler(self, gate, args):
        """
        Compiler for the RZ gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        return self._rotation_compiler(gate, "sz", "sz", args)

    def rx_compiler(self, gate, args):
        """
        Compiler for the RX gate

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        return self._rotation_compiler(gate, "sx", "sx", args)

    def _swap_compiler(self, gate, area, correction_angle, args):
        q1, q2 = gate.targets
        pulse_info = []
        pulse_name = "sz" + str(q1)
        coeff = self.wq[q1] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "sz" + str(q2)
        coeff = self.wq[q2] - self.params["w0"]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q1)
        coeff = self.params["g"][q1]
        pulse_info += [(pulse_name, coeff)]
        pulse_name = "g" + str(q2)
        coeff = self.params["g"][q2]
        pulse_info += [(pulse_name, coeff)]

        J = self.params["g"][q1] * self.params["g"][q2] * (
            1. / self.Delta[q1] + 1. / self.Delta[q2]) / 2.
        coeff, tlist = self.generate_pulse_shape(
            args["shape"], args["num_samples"], maximum=J, area=area)
        instruction_list = [Instruction(gate, tlist, pulse_info)]

        # corrections
        gate1 = Gate("RZ", [q1], None, arg_value=correction_angle)
        compiled_gate1 = self.gate_compiler["RZ"](gate1, args)
        instruction_list += compiled_gate1
        gate2 = Gate("RZ", [q2], None, arg_value=correction_angle)
        compiled_gate2 = self.gate_compiler["RZ"](gate2, args)
        instruction_list += compiled_gate2
        gate3 = Gate("GLOBALPHASE", None, None, arg_value=correction_angle)
        self.globalphase_compiler(gate3, args)
        return instruction_list

    def sqrtiswap_compiler(self, gate, args):
        """
        Compiler for the SQRTISWAP gate.

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.

        Notes
        -----
        This version of sqrtiswap_compiler has very low fidelity, please use
        iswap
        """
        # FIXME This decomposition has poor behaviour.
        return self._swap_compiler(
            gate, area=1/4, correction_angle=-np.pi/4, args=args)

    def iswap_compiler(self, gate, args):
        """
        Compiler for the ISWAP gate.

        Parameters
        ----------
        gate : :obj:`.Gate`:
            The quantum gate to be compiled.
        args : dict
            The compilation configuration defined in the attributes
            :obj:`.GateCompiler.args` or given as a parameter in
            :obj:`.GateCompiler.compile`.

        Returns
        -------
        A list of :obj:`.Instruction`, including the compiled pulse
        information for this gate.
        """
        return self._swap_compiler(
            gate, area=1/2, correction_angle=-np.pi/2, args=args)

    def globalphase_compiler(self, gate, args):
        """
        Compiler for the GLOBALPHASE gate
        """
        self.global_phase += gate.arg_value
