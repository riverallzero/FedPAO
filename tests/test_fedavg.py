import numpy as np

from src.server.fedavg import communication_bytes, fedavg


def test_uniform_fedavg():
    result = fedavg(
        [
            np.array([0.0, 2.0], dtype=np.float32),
            np.array([2.0, 4.0], dtype=np.float32),
        ]
    )
    assert np.allclose(result, np.array([1.0, 3.0], dtype=np.float32))


def test_weighted_fedavg():
    result = fedavg(
        [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        ],
        [1, 3],
    )
    assert np.allclose(result, np.array([0.75, 0.75], dtype=np.float32))
    assert communication_bytes(result) == 8


if __name__ == "__main__":
    test_uniform_fedavg()
    test_weighted_fedavg()
