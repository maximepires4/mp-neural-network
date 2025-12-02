import os
import sys

from mpneuralnetwork import backend

# Add the project root directory to sys.path
# This allows imports like "from mpneuralnetwork..." and "from tests.utils..." to work
# regardless of where pytest is run from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_report_header(config):
    backend_name = backend.xp.__name__
    device_info = "GPU (CuPy)" if backend_name == "cupy" else "CPU (NumPy)"
    return [f"MPNeuralNetwork Backend: {device_info}"]
