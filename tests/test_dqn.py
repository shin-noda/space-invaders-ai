import os

import numpy as np
import torch

from src.algorithms.dqn import DQN


def test_dqn():
    num_actions = 6
    batch_size = 4

    class DummyBrain(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.num_actions = num_actions

            self.fc = torch.nn.Linear(
                4 * 84 * 84,
                num_actions,
            )

        def forward(self, x):
            x = x.reshape(
                x.size(0),
                -1,
            )

            return self.fc(x)

    brain = DummyBrain()
    target_brain = DummyBrain()

    dqn = DQN(
        brain=brain,
        target_brain=target_brain,
        batch_size=batch_size,
    )

    state = np.random.rand(
        4,
        84,
        84,
    ).astype(np.float32)

    next_state = np.random.rand(
        4,
        84,
        84,
    ).astype(np.float32)

    # --------------------------------------------------
    # Fill replay buffer
    # --------------------------------------------------

    for i in range(10):
        dqn.store(
            state,
            action=i % num_actions,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

    print(
        "Replay buffer size:",
        len(dqn.replay_buffer),
    )

    assert (
        len(dqn.replay_buffer)
        == 10
    )

    # --------------------------------------------------
    # Check sampling through DQN
    # --------------------------------------------------

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = dqn.replay_buffer.sample(
        batch_size
    )

    print(
        "Sampled states:",
        states.shape,
    )

    print(
        "Sampled next states:",
        next_states.shape,
    )

    assert states.shape == (
        batch_size,
        4,
        84,
        84,
    )

    assert next_states.shape == (
        batch_size,
        4,
        84,
        84,
    )

    assert actions.shape == (
        batch_size,
    )

    assert rewards.shape == (
        batch_size,
    )

    assert dones.shape == (
        batch_size,
    )

    # --------------------------------------------------
    # Check DQN training
    # --------------------------------------------------

    loss = dqn.train_step()

    print(
        "DQN loss:",
        loss,
    )

    assert loss is not None

    assert isinstance(
        loss,
        float,
    )

    # --------------------------------------------------
    # Test checkpoint save/load
    # --------------------------------------------------

    checkpoint_path = (
        "tests/test_dqn_checkpoint.pt"
    )

    dqn.total_steps = 1234

    original_total_steps = (
        dqn.total_steps
    )

    dqn.save_checkpoint(
        path=checkpoint_path,
        episode=100,
        epsilon=0.42,
        scores_window=[
            10.0,
            20.0,
            30.0,
        ],
        total_steps=dqn.total_steps,
    )

    print(
        "Checkpoint saved."
    )

    # Create a completely new DQN.
    new_brain = DummyBrain()
    new_target_brain = DummyBrain()

    loaded_dqn = DQN(
        brain=new_brain,
        target_brain=new_target_brain,
        batch_size=batch_size,
    )

    checkpoint = (
        loaded_dqn.load_checkpoint(
            checkpoint_path
        )
    )

    print(
        "Checkpoint loaded."
    )

    # --------------------------------------------------
    # Verify DQN checkpoint metadata
    # --------------------------------------------------

    assert (
        checkpoint["episode"]
        == 100
    )

    assert (
        checkpoint["epsilon"]
        == 0.42
    )

    assert (
        checkpoint["scores_window"]
        == [
            10.0,
            20.0,
            30.0,
        ]
    )

    assert (
        loaded_dqn.total_steps
        == original_total_steps
    )

    # --------------------------------------------------
    # Verify replay buffer is NOT in checkpoint
    # --------------------------------------------------

    assert (
        "replay_buffer"
        not in checkpoint
    )

    print(
        "Replay buffer is correctly "
        "excluded from DQN checkpoint."
    )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    if os.path.exists(
        checkpoint_path
    ):
        os.remove(
            checkpoint_path
        )

    print(
        "✅ DQN test passed!"
    )


if __name__ == "__main__":
    test_dqn()