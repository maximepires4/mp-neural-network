import os

import numpy as np

from mpneuralnetwork import backend, to_device, to_host


def test_backend_identification():
    """Vérifie que le backend chargé correspond à la variable d'environnement."""
    env_backend = os.getenv("MPNN_BACKEND", "numpy").lower()

    # Si on a demandé cupy mais qu'il n'est pas dispo, backend.py fallback sur numpy
    # On doit vérifier ce comportement
    if env_backend == "cupy" and backend._cupy_available:
        assert backend.xp.__name__ == "cupy"
    else:
        assert backend.xp.__name__ == "numpy"


def test_to_device_structure():
    """Vérifie que to_device crée le bon type d'array."""
    data = [1, 2, 3]
    array_on_device = to_device(data)

    assert isinstance(array_on_device, backend.xp.ndarray)

    if backend.xp.__name__ == "cupy":
        import cupy

        assert isinstance(array_on_device, cupy.ndarray)
        # Vérifier que c'est bien sur le device (pas sur le host)
        assert array_on_device.device.id >= 0


def test_to_host_conversion():
    """Vérifie que to_host ramène toujours un numpy array."""
    data = [1.0, 2.0, 3.0]
    array_device = to_device(data)
    array_host = to_host(array_device)

    assert isinstance(array_host, np.ndarray)
    # Les arrays numpy n'ont pas de prop device ou elle est différente
    if backend.xp.__name__ == "cupy":
        import cupy

        assert not isinstance(array_host, cupy.ndarray)
