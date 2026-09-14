import csv
import os
from collections import deque

import numpy as np


class Trainer:
    """
    Generic training loop for Atari reinforcement learning.

    The Trainer handles:
    - environment interaction
    - episode tracking
    - raw score tracking
    - clipped reward tracking
    - episode-based epsilon scheduling
    - replay buffer warmup
    - target network updates
    - checkpoint saving
    - replay buffer saving
    - episode CSV logging
    """

    def __init__(
        self,
        env,
        algorithm,
        device,
        checkpoint_dir="checkpoints",
        save_interval=50,
    ):
        self.env = env
        self.algorithm = algorithm
        self.device = device

        self.checkpoint_dir = (
            checkpoint_dir
        )

        self.replaybuffer_dir = os.path.join(
            checkpoint_dir,
            "replay_buffers",
        )

        self.save_interval = (
            save_interval
        )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True,
        )

        os.makedirs(
            self.replaybuffer_dir,
            exist_ok=True,
        )

        self.scores = deque(
            maxlen=100
        )

        self.csv_path = os.path.join(
            self.checkpoint_dir,
            "episode_log.csv",
        )

        self._create_csv_if_missing()

    # --------------------------------------------------
    # CSV
    # --------------------------------------------------

    def _csv_header(self):
        return [
            "episode",
            "score",
            "clipped_reward",
            "average_score",
            "epsilon",
            "steps",
            "loss",
        ]

    def _create_csv_if_missing(self):
        if os.path.exists(
            self.csv_path
        ):
            return

        with open(
            self.csv_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                self._csv_header()
            )

    def _prepare_csv(
        self,
        start_episode,
    ):
        """
        When resuming training, preserve all
        CSV rows before start_episode.

        The resumed episode will be written again.
        """

        if start_episode == 1:

            with open(
                self.csv_path,
                "w",
                newline="",
            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    self._csv_header()
                )

            return

        if not os.path.exists(
            self.csv_path
        ):
            self._create_csv_if_missing()
            return

        with open(
            self.csv_path,
            "r",
            newline="",
        ) as f:

            rows = list(
                csv.reader(f)
            )

        if not rows:
            self._create_csv_if_missing()
            return

        header = rows[0]

        preserved_rows = [
            row
            for row in rows[1:]
            if row
            and int(row[0]) < start_episode
        ]

        with open(
            self.csv_path,
            "w",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(header)

            writer.writerows(
                preserved_rows
            )

    # --------------------------------------------------
    # Score window
    # --------------------------------------------------

    def _restore_scores(
        self,
        scores_window,
    ):
        self.scores.clear()

        if scores_window is None:
            return

        for score in scores_window:
            self.scores.append(
                float(score)
            )

    # --------------------------------------------------
    # Epsilon
    # --------------------------------------------------

    def _calculate_epsilon(
        self,
        episode,
        epsilon_start,
        epsilon_end,
        exploration_episodes,
    ):
        """
        Linearly decay epsilon based on episode number.

        Example:

            episode 1
                -> epsilon_start

            episode exploration_episodes
                -> epsilon_end

            episodes after that
                -> epsilon_end
        """

        if exploration_episodes <= 0:
            return float(epsilon_end)

        progress = (
            (episode - 1)
            / exploration_episodes
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        epsilon = (
            epsilon_start
            - progress
            * (
                epsilon_start
                - epsilon_end
            )
        )

        return max(
            epsilon_end,
            epsilon,
        )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    def train_dqn(
        self,
        num_episodes,
        epsilon_start=1.0,
        epsilon_end=0.1,
        exploration_episodes=9_500,
        warmup_steps=10_000,
        target_update_interval=10_000,
        start_episode=1,
    ):
        """
        Train DQN for the requested number of episodes.

        Epsilon decay is episode-based.

        The target network and replay warmup remain
        step-based.
        """

        self._prepare_csv(
            start_episode
        )

        total_steps = int(
            self.algorithm.total_steps
        )

        for episode in range(
            start_episode,
            num_episodes + 1,
        ):

            state, info = (
                self.env.reset()
            )

            done = False

            clipped_reward = 0.0
            last_loss = None

            # ------------------------------------------
            # Episode-based epsilon
            # ------------------------------------------

            epsilon = (
                self._calculate_epsilon(
                    episode=episode,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    exploration_episodes=(
                        exploration_episodes
                    ),
                )
            )

            # ------------------------------------------
            # Environment interaction
            # ------------------------------------------

            while not done:

                action = (
                    self.algorithm.select_action(
                        state,
                        epsilon,
                    )
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = self.env.step(
                    action
                )

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

                # --------------------------------------
                # Training after replay warmup
                # --------------------------------------

                if (
                    len(
                        self.algorithm.replay_buffer
                    )
                    >= warmup_steps
                ):

                    loss = (
                        self.algorithm.train_step()
                    )

                    if loss is not None:
                        last_loss = loss

                # --------------------------------------
                # Step tracking
                # --------------------------------------

                total_steps += 1

                self.algorithm.total_steps = (
                    total_steps
                )

                # --------------------------------------
                # Target network update
                # --------------------------------------

                if (
                    target_update_interval > 0
                    and total_steps
                    % target_update_interval
                    == 0
                ):

                    self.algorithm.update_target_network()

                state = next_state

                clipped_reward += reward

            # ------------------------------------------
            # Episode score
            # ------------------------------------------

            if (
                "episode" in info
                and info["episode"] is not None
                and "r" in info["episode"]
            ):

                score = float(
                    info["episode"]["r"]
                )

            else:

                score = float(
                    clipped_reward
                )

            self.scores.append(
                score
            )

            average_score = float(
                np.mean(
                    self.scores
                )
            )

            # ------------------------------------------
            # Loss formatting
            # ------------------------------------------

            if last_loss is None:
                loss_text = "N/A"
            else:
                loss_text = (
                    f"{float(last_loss):.4f}"
                )

            # ------------------------------------------
            # CSV
            # ------------------------------------------

            with open(
                self.csv_path,
                "a",
                newline="",
            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        episode,
                        f"{score:.1f}",
                        f"{clipped_reward:.1f}",
                        f"{average_score:.1f}",
                        f"{epsilon:.4f}",
                        total_steps,
                        (
                            "N/A"
                            if last_loss is None
                            else f"{float(last_loss):.6f}"
                        ),
                    ]
                )

            # ------------------------------------------
            # Checkpoint
            # ------------------------------------------

            if (
                self.save_interval > 0
                and episode
                % self.save_interval
                == 0
            ):

                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_ep{episode}.pt",
                )

                self.algorithm.save_checkpoint(
                    path=checkpoint_path,
                    episode=episode,
                    epsilon=epsilon,
                    scores_window=list(
                        self.scores
                    ),
                    total_steps=total_steps,
                )

                replay_path = os.path.join(
                    self.replaybuffer_dir,
                    f"replay_ep{episode}.npz",
                )

                self.algorithm.save_replay_buffer(
                    replay_path
                )

            # ------------------------------------------
            # Console output
            # ------------------------------------------

            print(
                f"Episode {episode:5d} | "
                f"Score {score:7.1f} | "
                f"Clipped {clipped_reward:5.1f} | "
                f"Avg {average_score:7.1f} | "
                f"Epsilon {epsilon:.4f} | "
                f"Steps {total_steps:8d} | "
                f"Loss {loss_text}"
            )

        return list(
            self.scores
        )