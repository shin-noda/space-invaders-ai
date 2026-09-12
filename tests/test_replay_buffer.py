import os

import numpy as np

from src.algorithms.replay_buffer import ReplayBuffer


def test_replay_buffer():
    capacity = 100_000
    batch_size = 4

    replay_buffer = ReplayBuffer(
        capacity=capacity,
    )

    state = np.random.rand(
        4,
        84,
        84,
    ).astype(np.float32)

    # --------------------------------------------------
    # Fill replay buffer
    # --------------------------------------------------

    for i in range(10):
        replay_buffer.add(
            state,
            action=i % 6,
            reward=1.0,
            done=False,
        )

    print(
        "Replay buffer size:",
        len(replay_buffer),
    )

    assert (
        len(replay_buffer)
        == 10
    )

    # --------------------------------------------------
    # Check compact storage
    # --------------------------------------------------

    assert (
        replay_buffer.frames.dtype
        == np.uint8
    )

    assert (
        replay_buffer.frames.shape
        == (
            capacity,
            84,
            84,
        )
    )

    assert (
        replay_buffer.actions.dtype
        == np.int64
    )

    assert (
        replay_buffer.rewards.dtype
        == np.float32
    )

    assert (
        replay_buffer.dones.dtype
        == np.bool_
    )

    # --------------------------------------------------
    # Check buffer position
    # --------------------------------------------------

    assert (
        replay_buffer.position
        == 10
    )

    assert (
        replay_buffer.size
        == 10
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
    ) = replay_buffer.sample(
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
    # Check sampled data types
    # --------------------------------------------------

    assert (
        states.dtype
        == np.float32
    )

    assert (
        next_states.dtype
        == np.float32
    )

    assert (
        actions.dtype
        == np.int64
    )

    assert (
        rewards.dtype
        == np.float32
    )

    assert (
        dones.dtype
        == np.float32
    )

    # --------------------------------------------------
    # Check reconstructed state range
    # --------------------------------------------------

    assert np.all(
        states >= 0.0
    )

    assert np.all(
        states <= 1.0
    )

    assert np.all(
        next_states >= 0.0
    )

    assert np.all(
        next_states <= 1.0
    )

    # --------------------------------------------------
    # Check episode boundary handling
    # --------------------------------------------------

    replay_buffer = ReplayBuffer(
        capacity=100,
    )

    for i in range(10):
        replay_buffer.add(
            state,
            action=0,
            reward=1.0,
            done=(i == 4),
        )

    assert (
        replay_buffer.size
        == 10
    )

    # A transition immediately after an episode boundary
    # should not use frames from the previous episode.
    assert (
        replay_buffer._valid_index(5)
        is False
    )

    print(
        "Episode boundary handling passed."
    )

    # --------------------------------------------------
    # Test save/load
    # --------------------------------------------------

    replay_buffer_path = (
        "tests/test_replay_buffer.npz"
    )

    original_frames = (
        replay_buffer.frames.copy()
    )

    original_actions = (
        replay_buffer.actions.copy()
    )

    original_rewards = (
        replay_buffer.rewards.copy()
    )

    original_dones = (
        replay_buffer.dones.copy()
    )

    original_position = (
        replay_buffer.position
    )

    original_size = (
        replay_buffer.size
    )

    replay_buffer.save(
        replay_buffer_path
    )

    print(
        "Replay buffer saved."
    )

    loaded_replay_buffer = ReplayBuffer(
        capacity=100,
    )

    loaded_replay_buffer.load(
        replay_buffer_path
    )

    print(
        "Replay buffer loaded."
    )

    # --------------------------------------------------
    # Verify loaded buffer
    # --------------------------------------------------

    assert (
        loaded_replay_buffer.capacity
        == replay_buffer.capacity
    )

    assert (
        loaded_replay_buffer.frame_shape
        == replay_buffer.frame_shape
    )

    assert (
        loaded_replay_buffer.position
        == original_position
    )

    assert (
        loaded_replay_buffer.size
        == original_size
    )

    assert np.array_equal(
        loaded_replay_buffer.frames,
        original_frames,
    )

    assert np.array_equal(
        loaded_replay_buffer.actions,
        original_actions,
    )

    assert np.array_equal(
        loaded_replay_buffer.rewards,
        original_rewards,
    )

    assert np.array_equal(
        loaded_replay_buffer.dones,
        original_dones,
    )

    print(
        "Replay buffer restored correctly."
    )

    # --------------------------------------------------
    # Verify loaded buffer can sample
    # --------------------------------------------------

    (
        loaded_states,
        loaded_actions,
        loaded_rewards,
        loaded_next_states,
        loaded_dones,
    ) = loaded_replay_buffer.sample(
        batch_size
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

    assert loaded_actions.shape == (
        batch_size,
    )

    assert loaded_rewards.shape == (
        batch_size,
    )

    assert loaded_dones.shape == (
        batch_size,
    )

    print(
        "Loaded replay buffer can sample."
    )

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    if os.path.exists(
        replay_buffer_path
    ):
        os.remove(
            replay_buffer_path
        )

    print(
        "✅ ReplayBuffer test passed!" 
    )


if __name__ == "__main__":
    test_replay_buffer()