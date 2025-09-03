import tempfile
import os
from qiskit import QuantumCircuit
from numpy import pi

import qmuvi
from qmuvi.musical_processing import note_map_c_major_arpeggio


def test_generate_qmuvi(tmp_path):
    qmuvi_name = "single_qubit"
    data_dir = tempfile.mkdtemp(dir=tmp_path)
    output_dir = os.path.join(data_dir, qmuvi_name + "-output")

    circ = QuantumCircuit(1)

    circ.x(0)
    circ.ry(pi / 8, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()
    circ.rz(pi / 4, 0)
    circ.barrier()

    time_list = [(200, 0)] * 8 + [(400, 0)]

    instruments = []
    instruments.append(qmuvi.get_instrument_collection("organ"))

    qmuvi.generate_qmuvi(
        circ,
        qmuvi_name,
        noise_model=None,
        rhythm=time_list,
        instruments=instruments,
        note_map=note_map_c_major_arpeggio,
        invert_colours=True,
        fps=2,
        smooth_transitions=False,
        output_dir=output_dir,
    )
    assert os.path.exists(output_dir)
    assert os.path.isfile(os.path.join(output_dir, "single_qubit.mp4"))
