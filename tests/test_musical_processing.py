import tempfile
import os
import pytest

from qmuvi.musical_processing import convert_midi_to_wav_timidity
from qmuvi.data_manager import DataManager


def create_test_midi_file(tmpdir: str) -> str:
    midi_data = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\x00"  # Minimal valid MIDI header
    midi_path = os.path.join(tmpdir, "test.mid")
    with open(midi_path, "wb") as f:
        f.write(midi_data)
    return midi_path


@pytest.fixture
def tmp_data_manager(tmp_path: str) -> DataManager:
    tmpdir = tempfile.mkdtemp(dir=tmp_path)
    return DataManager(data_dir=tmpdir, default_name="test")


def test_convert_midi_to_wav_timidity(tmp_data_manager):
    midi_file_path = create_test_midi_file(tmp_data_manager.data_dir)

    # Call the function that uses timidity
    convert_midi_to_wav_timidity(tmp_data_manager, log_to_file=True)

    # Check that subprocess.run was called with the expected arguments
    expected_output_path = os.path.join(tmp_data_manager.data_dir, "test.wav")
    print("expected_output_path:", expected_output_path)
    assert os.path.exists(expected_output_path) == True

    # Clean up temporary files
    os.remove(midi_file_path)
    # remove log_files in form of ending in log_file_0, log_file_1, ... from cwd
    for filename in os.listdir():
        if filename.startswith("log-timidity-"):
            os.remove(filename)
