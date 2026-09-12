import csv
import os
from collections import deque

import numpy as np
import torch


class Trainer:
    """
    Generic training loop for Atari reinforcement learning.

    The Trainer handles:
    - environment interaction
    - episode tracking
    - epsilon scheduling
    - target network updates
    - checkpoint saving
    - replay buffer saving
    - episode CSV logging

    The actual learning logic belongs to the algorithm.
    """

    def __init__(
        self,
        env,
        algorithm,
        device="cpu",
        checkpoint_dir="checkpoints",
        replaybuffer_dir="replaybuffer",
        save_interval=50,
    ):
        self.env = env
        self.algorithm = algorithm
        self.device = device

        self.checkpoint_dir = checkpoint_dir
        self.replaybuffer_dir = replaybuffer_dir
        self.save_interval = save_interval

        os.makedirs(
            checkpoint_dir,
            exist_ok=True,
        )

        os.makedirs(
            replaybuffer_dir,
            exist_ok=True,
        )

        self.scores = deque(
            maxlen=100
        )

        # Episode log CSV
        self.csv_path = os.path.join(
            checkpoint_dir,
            "episode_log.csv",
        )

        self._create_csv_if_missing()

    def _create_csv_if_missing(self):
        """
        Create the episode CSV with its header
        if it does not already exist.
        """

        if not os.path.exists(
            self.csv_path
        ):
            with open(
                self.csv_path,
                "w",
                newline="",
            ) as file:
                writer = csv.writer(file)

                writer.writerow(
                    [
                        "episode",
                        "reward",
                        "average_reward",
                        "epsilon",
                        "steps",
                        "loss",
                    ]
                )

    def _prepare_csv(
        self,
        start_episode,
    ):
        """
        Make the CSV match the checkpoint we are
        resuming from.

        If resuming from episode 50, only episodes
        1-50 are kept. Any rows after episode 50
        are discarded.

        If starting from episode 1, the CSV is reset.
        """

        header = [
            "episode",
            "reward",
            "average_reward",
            "epsilon",
            "steps",
            "loss",
        ]

        # Fresh training run.
        if start_episode == 1:
            with open(
                self.csv_path,
                "w",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(header)

            return

        # Nothing to truncate if the CSV does not exist.
        if not os.path.exists(
            self.csv_path
        ):
            with open(
                self.csv_path,
                "w",
                newline="",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(header)

            return

        # Read the existing CSV.
        with open(
            self.csv_path,
            "r",
            newline="",
        ) as file:
            reader = csv.reader(file)

            rows = list(reader)

        # Keep the header plus episodes before
        # the new training run.
        kept_rows = [header]

        for row in rows[1:]:
            if not row:
                continue

            try:
                episode = int(row[0])
            except ValueError:
                continue

            if episode < start_episode:
                kept_rows.append(row)

        # Rewrite the CSV with only valid history.
        with open(
            self.csv_path,
            "w",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerows(kept_rows)

    def train_dqn(
        self,
        num_episodes,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        target_update_interval=1000,
        start_episode=1,
    ):
        """
        Train using DQN.

        The DQN algorithm owns the persistent
        total training step counter.
        """

        # Make sure the CSV matches the checkpoint
        # we are starting from.
        self._prepare_csv(
            start_episode
        )

        epsilon = epsilon_start

        # Restore the DQN's existing step count.
        # If this is a fresh run, it will be 0.
        total_steps = (
            self.algorithm.total_steps
        )

        for episode in range(
            start_episode,
            num_episodes + 1,
        ):
            state, _ = self.env.reset()

            done = False
            episode_reward = 0.0

            while not done:

                action = (
                    self.algorithm.select_action(
                        state,
                        epsilon=epsilon,
                    )
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.env.step(action)

                done = (
                    terminated
                    or truncated
                )

                self.algorithm.store(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = (
                    self.algorithm.train_step()
                )

                total_steps += 1

                # Keep the algorithm's persistent
                # step counter synchronized.
                self.algorithm.total_steps = (
                    total_steps
                )

                if (
                    total_steps
                    % target_update_interval
                    == 0
                ):
                    self.algorithm.update_target_network()

                state = next_state
                episode_reward += reward

            self.scores.append(
                episode_reward
            )

            epsilon = max(
                epsilon_end,
                epsilon * epsilon_decay,
            )

            average_score = np.mean(
                self.scores
            )

            loss_text = (
                "N/A"
                if loss is None
                else f"{loss:.4f}"
            )

            # Record this episode in the CSV.
            with open(
                self.csv_path,
                "a",
                newline="",
            ) as file:
                writer = csv.writer(file)

                writer.writerow(
                    [
                        episode,
                        episode_reward,
                        average_score,
                        epsilon,
                        total_steps,
                        loss_text,
                    ]
                )

            if (
                episode
                % self.save_interval
                == 0
            ):
                # Save DQN checkpoint.
                path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_ep{episode}.pt",
                )

                self.algorithm.save_checkpoint(
                    path=path,
                    episode=episode,
                    epsilon=epsilon,
                    scores_window=self.scores,
                    total_steps=total_steps,
                )

                # Save replay buffer separately.
                replaybuffer_path = os.path.join(
                    self.replaybuffer_dir,
                    f"replay_buffer_ep{episode}.npz",
                )

                self.algorithm.replay_buffer.save(
                    replaybuffer_path
                )

            print(
                f"Episode {episode:5d} | "
                f"Reward {episode_reward:7.2f} | "
                f"Avg {average_score:7.2f} | "
                f"Epsilon {epsilon:.4f} | "
                f"Steps {total_steps:7d} | "
                f"Loss {loss_text}"
            )

        return list(self.scores)