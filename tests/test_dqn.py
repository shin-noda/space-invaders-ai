import os
import tempfile

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

    # --------------------------------------------------
    # Check compact storage
    # --------------------------------------------------

    assert (
        dqn.replay_buffer.frames.dtype
        == np.uint8
    )

    assert (
        dqn.replay_buffer.frames.shape
        == (100_000, 84, 84)
    )

    # --------------------------------------------------
    # Check sampling
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

    # --------------------------------------------------
    # Check DQN training
    # --------------------------------------------------

    loss = dqn.train_step()

    print(
        "DQN loss:",
        loss,
    )

    assert loss is not None

    # --------------------------------------------------
    # Test checkpoint save/load
    # --------------------------------------------------

    checkpoint_path = (
        "tests/test_dqn_checkpoint.pt"
    )

    dqn.replay_buffer.position = 10
    dqn.replay_buffer.size = 10
    dqn.total_steps = 1234

    original_position = (
        dqn.replay_buffer.position
    )

    original_size = (
        dqn.replay_buffer.size
    )

    original_total_steps = (
        dqn.total_steps
    )

    original_frames = (
        dqn.replay_buffer.frames.copy()
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
    # Verify metadata
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
    # Verify replay buffer
    # --------------------------------------------------

    assert (
        loaded_dqn.replay_buffer.position
        == original_position
    )

    assert (
        loaded_dqn.replay_buffer.size
        == original_size
    )

    assert np.array_equal(
        loaded_dqn.replay_buffer.frames,
        original_frames,
    )

    print(
        "Replay buffer restored correctly."
    )

    # --------------------------------------------------
    # Verify loaded replay buffer can sample
    # --------------------------------------------------

    (
        loaded_states,
        loaded_actions,
        loaded_rewards,
        loaded_next_states,
        loaded_dones,
    ) = (
        loaded_dqn.replay_buffer.sample(
            batch_size
        )
    )

    assert loaded_states.shape == (
        batch_size,
        4,
        84,
        84,
    )

    assert loaded_next_states.shape == (
        batch_size,
        4,
        84,
        84,
    )

    print(
        "Loaded replay buffer can sample."
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
        "✅ DQN checkpoint test passed!"
    )


if __name__ == "__main__":
    test_dqn()