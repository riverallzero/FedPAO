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
    projector1 = SoftPromptProjector(4, 3, 5, seed=7)
    projector2 = SoftPromptProjector(4, 3, 5, seed=7)
    p1 = projector1.project(z)
    p2 = projector2.project(z)
    assert np.allclose(p1.numpy(), p2.numpy())
    assert projector1.projection_hash() == projector2.projection_hash()


if __name__ == "__main__":
    test_soft_prompt_shape()
    test_soft_prompt_reproducible()
