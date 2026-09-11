import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


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
        next_state,
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

    def _get_frame(self, index):
        return self.frames[
            index % self.capacity
        ]

    def _get_state(self, index):
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

    def _valid_index(self, index):
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

    def sample(self, batch_size):
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

    def __len__(self):
        return self.size

    def state_dict(self):
        """
        Return replay buffer state for checkpointing.
        """

        return {
            "capacity": self.capacity,
            "frame_shape": self.frame_shape,
            "frames": self.frames,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "position": self.position,
            "size": self.size,
        }

    def load_state_dict(self, state):
        """
        Restore replay buffer from checkpoint.
        """

        self.frames = state["frames"]
        self.actions = state["actions"]
        self.rewards = state["rewards"]
        self.dones = state["dones"]

        self.position = state[
            "position"
        ]

        self.size = state[
            "size"
        ]


class DQN:
    """
    Deep Q-Network algorithm.

    Handles:
    - epsilon-greedy action selection
    - experience replay
    - Q-learning targets
    - target network updates
    - optimizer updates
    - checkpoint saving/loading
    """

    def __init__(
        self,
        brain,
        target_brain,
        lr=1e-4,
        gamma=0.99,
        replay_capacity=100_000,
        batch_size=32,
        device="cpu",
    ):
        self.brain = brain
        self.target_brain = target_brain

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.optimizer = optim.Adam(
            self.brain.parameters(),
            lr=lr,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
        )

        self.total_steps = 0

        self.update_target_network()

    def select_action(
        self,
        state,
        epsilon=0.0,
    ):
        if random.random() < epsilon:
            return random.randrange(
                self.brain.num_actions
            )

        with torch.no_grad():
            if not isinstance(
                state,
                torch.Tensor,
            ):
                state = torch.from_numpy(
                    np.asarray(
                        state,
                        dtype=np.float32,
                    )
                )

            state = state.to(
                self.device
            )

            if state.ndim == 3:
                state = state.unsqueeze(0)

            q_values = self.brain(
                state
            )

            return q_values.argmax(
                dim=1
            ).item()

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def train_step(self):
        if (
            len(self.replay_buffer)
            < max(self.batch_size, 5)
        ):
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.from_numpy(
            states
        ).float().to(self.device)

        actions = torch.from_numpy(
            actions
        ).long().to(self.device)

        rewards = torch.from_numpy(
            rewards
        ).float().to(self.device)

        next_states = torch.from_numpy(
            next_states
        ).float().to(self.device)

        dones = torch.from_numpy(
            dones
        ).float().to(self.device)

        q_values = self.brain(
            states
        )

        current_q = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q = self.target_brain(
                next_states
            ).max(
                dim=1
            ).values

            target_q = (
                rewards
                + self.gamma
                * next_q
                * (1.0 - dones)
            )

        loss = F.smooth_l1_loss(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.brain.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_brain.load_state_dict(
            self.brain.state_dict()
        )

        self.target_brain.eval()

    def save_checkpoint(
        self,
        path,
        episode,
        epsilon,
        scores_window,
        total_steps=0,
    ):
        torch.save(
            {
                "episode": episode,
                "epsilon": epsilon,
                "total_steps": total_steps,
                "brain_state_dict": (
                    self.brain.state_dict()
                ),
                "target_brain_state_dict": (
                    self.target_brain.state_dict()
                ),
                "optimizer_state_dict": (
                    self.optimizer.state_dict()
                ),
                "scores_window": (
                    list(scores_window)
                ),
                "replay_buffer": (
                    self.replay_buffer.state_dict()
                ),
            },
            path,
        )

    def load_checkpoint(
        self,
        path,
    ):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.brain.load_state_dict(
            checkpoint[
                "brain_state_dict"
            ]
        )

        self.target_brain.load_state_dict(
            checkpoint[
                "target_brain_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if "replay_buffer" in checkpoint:
            self.replay_buffer.load_state_dict(
                checkpoint[
                    "replay_buffer"
                ]
            )

        self.total_steps = checkpoint.get(
            "total_steps",
            0,
        )

        return checkpoint