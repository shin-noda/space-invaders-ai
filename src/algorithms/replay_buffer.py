import random
import numpy as np


class ReplayBuffer:
    """
    Memory-efficient replay buffer for Atari DQN.

    Instead of storing full 4-frame observations as float32,
    the buffer stores individual grayscale frames as uint8.

    A state is reconstructed from the current frame and the
    previous three frames when sampled.
    """

    def __init__(
        self,
        capacity,
        frame_shape=(84, 84),
    ):
        self.capacity = capacity
        self.frame_shape = frame_shape

        self.frames = np.zeros(
            (
                capacity,
                *frame_shape,
            ),
            dtype=np.uint8,
        )

        self.actions = np.zeros(
            capacity,
            dtype=np.int64,
        )

        self.rewards = np.zeros(
            capacity,
            dtype=np.float32,
        )

        self.dones = np.zeros(
            capacity,
            dtype=np.bool_,
        )

        self.position = 0
        self.size = 0

    def add(
        self,
        state,
        action,
        reward,
        done,
    ):
        """
        Store one transition.

        Only the newest frame from the state is stored.
        The next state is reconstructed from subsequent frames.
        """

        state = np.asarray(state)

        frame = state[-1]

        if frame.dtype != np.uint8:
            frame = np.clip(
                frame * 255.0,
                0,
                255,
            ).astype(np.uint8)

        self.frames[
            self.position
        ] = frame

        self.actions[
            self.position
        ] = int(action)

        self.rewards[
            self.position
        ] = float(reward)

        self.dones[
            self.position
        ] = bool(done)

        self.position = (
            self.position + 1
        ) % self.capacity

        self.size = min(
            self.size + 1,
            self.capacity,
        )

    def _get_frame(
        self,
        index,
    ):
        return self.frames[
            index % self.capacity
        ]

    def _get_state(
        self,
        index,
    ):
        """
        Reconstruct a 4-frame state ending at index.
        """

        indices = [
            index - 3,
            index - 2,
            index - 1,
            index,
        ]

        frames = []

        for i in indices:
            frame = self._get_frame(i)

            frames.append(frame)

        return np.stack(
            frames,
            axis=0,
        ).astype(
            np.float32
        ) / 255.0

    def _valid_index(
        self,
        index,
    ):
        """
        Check whether enough history exists to reconstruct
        a state without crossing an episode boundary.
        """

        if self.size < 5:
            return False

        if index < 3:
            return False

        if self.size < self.capacity:
            if index >= self.size - 1:
                return False

        for i in range(
            index - 3,
            index,
        ):
            if self.dones[
                i % self.capacity
            ]:
                return False

        return True

    def _sample_index(self):
        """
        Find a valid transition index.
        """

        for _ in range(1000):
            index = random.randrange(
                3,
                self.size - 1,
            )

            if self._valid_index(index):
                return index

        raise RuntimeError(
            "Could not find a valid replay "
            "buffer index."
        )

    def sample(
        self,
        batch_size,
    ):
        indices = [
            self._sample_index()
            for _ in range(batch_size)
        ]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for index in indices:
            state = self._get_state(
                index
            )

            next_state = self._get_state(
                index + 1
            )

            states.append(state)

            actions.append(
                self.actions[
                    index % self.capacity
                ]
            )

            rewards.append(
                self.rewards[
                    index % self.capacity
                ]
            )

            next_states.append(
                next_state
            )

            dones.append(
                self.dones[
                    index % self.capacity
                ]
            )

        return (
            np.stack(states),
            np.asarray(
                actions,
                dtype=np.int64,
            ),
            np.asarray(
                rewards,
                dtype=np.float32,
            ),
            np.stack(next_states),
            np.asarray(
                dones,
                dtype=np.float32,
            ),
        )

    def save(
        self,
        path,
    ):
        """
        Save the replay buffer to a separate file.
        """

        np.savez_compressed(
            path,
            capacity=self.capacity,
            frame_shape=self.frame_shape,
            frames=self.frames,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            position=self.position,
            size=self.size,
        )

    def load(
        self,
        path,
    ):
        """
        Load the replay buffer from a separate file.
        """

        data = np.load(
            path,
            allow_pickle=False,
        )

        self.capacity = int(
            data["capacity"]
        )

        self.frame_shape = tuple(
            data["frame_shape"]
        )

        self.frames = data["frames"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.dones = data["dones"]

        self.position = int(
            data["position"]
        )

        self.size = int(
            data["size"]
        )

    def __len__(self):
        return self.size