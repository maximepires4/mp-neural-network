#!/bin/bash
set -e

echo "========================================"
echo "   RUNNING TESTS ON CPU (NumPy)"
echo "========================================"
export MPNN_BACKEND=numpy
pytest

echo ""
if command -v nvidia-smi &>/dev/null; then
  echo "========================================"
  echo "   RUNNING TESTS ON GPU (CuPy)"
  echo "========================================"
  export MPNN_BACKEND=cupy
  # We check if cupy is actually importable before running to avoid crash if nvidia-smi exists but cupy is not installed
  if python -c "import cupy" &>/dev/null; then
    pytest
  else
    echo "Warning: NVIDIA GPU detected but 'cupy' python package not found. Skipping GPU tests."
  fi
else
  echo "========================================"
  echo "   SKIPPING GPU TESTS (No GPU)"
  echo "========================================"
fi
