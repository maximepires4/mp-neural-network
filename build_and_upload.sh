#!/bin/bash

set -e

if [ -f .env ]; then
  echo "Loading variables from .env file..."
  set -a
  source .env
  set +a
else
  echo "Warning: .env file not found."
fi

if [ -z "$PYPI_API_TOKEN" ]; then
    echo "Error: PYPI_API_TOKEN environment variable is not set." >&2
    echo "Please define it in a .env file or export it manually." >&2
    exit 1
fi

export TWINE_PASSWORD=$PYPI_API_TOKEN

echo "Installing/updating 'build' and 'twine'..."
pip install --upgrade build twine

if [ -d "dist" ]; then
    echo "Removing old build artifacts..."
    rm -r dist/
fi

echo "Building the project..."
python3 -m build

echo "Uploading to PyPI..."
twine upload dist/*

echo "Project has been successfully built and uploaded to PyPI."
