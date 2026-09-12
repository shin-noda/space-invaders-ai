import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .replay_buffer import ReplayBuffer


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
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
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

        self.total_steps = checkpoint.get(
            "total_steps",
            0,
        )

        return checkpoint