import time

import numpy as np
import torch


class Evaluator:
    """
    Evaluate a trained agent without exploration.

    The agent always selects the greedy action.
    """

    def __init__(
        self,
        env,
        brain,
        device="cpu",
    ):
        self.env = env
        self.brain = brain
        self.device = device

    def select_action(self, state):
        """
        Select the action with the highest predicted value.
        """

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

        with torch.no_grad():
            values = self.brain(state)

        return values.argmax(
            dim=1
        ).item()

    def evaluate(
        self,
        num_episodes=10,
        render=False,
        render_delay=0.02,
    ):
        """
        Run evaluation episodes.

        Returns
        -------
        list[float]
            Total reward for each episode.
        """

        scores = []

        for episode in range(
            1,
            num_episodes + 1,
        ):
            state, _ = self.env.reset()

            done = False
            total_reward = 0.0

            while not done:

                action = self.select_action(
                    state
                )

                state, reward, terminated, truncated, _ = (
                    self.env.step(action)
                )

                done = (
                    terminated
                    or truncated
                )

                total_reward += reward

                if render:
                    time.sleep(
                        render_delay
                    )

            scores.append(
                total_reward
            )

            print(
                f"Evaluation Episode "
                f"{episode:3d} | "
                f"Reward: {total_reward:7.2f}"
            )

        return scores

    @staticmethod
    def summarize(scores):
        """
        Print basic evaluation statistics.
        """

        if not scores:
            return

        print()
        print(
            f"Average: {np.mean(scores):.2f}"
        )
        print(
            f"Best:    {np.max(scores):.2f}"
        )
        print(
            f"Worst:   {np.min(scores):.2f}"
        )