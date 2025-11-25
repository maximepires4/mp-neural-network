import os
import sys

# Add the project root directory to sys.path
# This allows imports like "from mpneuralnetwork..." and "from tests.utils..." to work
# regardless of where pytest is run from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
