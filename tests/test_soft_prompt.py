import numpy as np

from src.prompts.soft_prompt import SoftPromptProjector


def test_soft_prompt_shape():
    projector = SoftPromptProjector(
        latent_dim=4,
        prompt_length=3,
        hidden_size=5,
        seed=1,
    )
    prompt = projector.project(np.ones(4, dtype=np.float32))
    assert tuple(prompt.shape) == (3, 5)


def test_soft_prompt_reproducible():
    z = np.ones(4, dtype=np.float32)
    p1 = SoftPromptProjector(4, 3, 5, seed=7).project(z)
    p2 = SoftPromptProjector(4, 3, 5, seed=7).project(z)
    assert np.allclose(p1.numpy(), p2.numpy())


if __name__ == "__main__":
    test_soft_prompt_shape()
    test_soft_prompt_reproducible()

