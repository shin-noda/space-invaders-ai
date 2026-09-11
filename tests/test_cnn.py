import numpy as np
import torch

from src.networks.cnn import AtariCNN


def test_cnn():
    num_actions = 6

    brain = AtariCNN(
        in_channels=4,
        num_actions=num_actions,
    )

    state = np.random.rand(
        4,
        84,
        84,
    ).astype(
        np.float32
    )

    state = torch.from_numpy(
        state
    ).unsqueeze(0)

    output = brain(state)

    print()
    print("Input shape:", state.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

    assert output.shape == (
        1,
        num_actions,
    )

    assert torch.isfinite(
        output
    ).all()

    print("✅ CNN test passed!")


if __name__ == "__main__":
    test_cnn()