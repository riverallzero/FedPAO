import numpy as np

from src.optimizers.cma_es_optimizer import CmaEsOptimizer


def test_cma_evaluates_initial_latent():
    calls = []

    def objective(latent):
        calls.append(latent.copy())
        return float(np.sum(latent**2)), {"accuracy": 1.0}

    initial = np.zeros(3, dtype=np.float32)
    optimizer = CmaEsOptimizer(
        sigma0=0.5,
        population_size=2,
        local_iterations=1,
        seed=1,
    )
    result = optimizer.optimize(initial, objective)
    assert result.num_evaluations == 3
    assert np.allclose(calls[0], initial)
    assert np.allclose(result.best_latent, initial)
    assert result.history[0]["is_initial"] is True


if __name__ == "__main__":
    test_cma_evaluates_initial_latent()

