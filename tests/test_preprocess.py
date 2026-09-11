import numpy as np

from src.environments.atari import make_atari_env
from src.environments.preprocess import (
    make_preprocessed_atari_env,
)


def test_preprocessing():
    env = make_atari_env(
        game="SpaceInvaders",
        stack_frames=1,
    )

    env = make_preprocessed_atari_env(
        env,
        output_size=(84, 84),
        grayscale=True,
        stack_frames=4,
    )

    obs, info = env.reset()

    print()
    print("Observation shape:", obs.shape)
    print("Observation dtype:", obs.dtype)
    print("Observation min:", obs.min())
    print("Observation max:", obs.max())

    assert obs.shape == (4, 84, 84)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0

    env.close()


if __name__ == "__main__":
    test_preprocessing()
    print("✅ Preprocessing test passed!")