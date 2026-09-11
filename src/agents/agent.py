from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Generic interface for an agent interacting with an environment.

    The agent is responsible for deciding which action to take.
    Learning logic belongs to the algorithm implementation.
    """

    def __init__(self, brain):
        self.brain = brain

    @abstractmethod
    def select_action(self, state, training=True):
        """
        Select an action given the current state.

        Parameters
        ----------
        state : observation
            Current environment observation.
        training : bool
            Whether the agent is currently training.

        Returns
        -------
        int
            Action index.
        """
        pass

    def save(self, path):
        """
        Save the agent's brain parameters.
        """
        import torch

        torch.save(
            self.brain.state_dict(),
            path,
        )

    def load(self, path, device="cpu"):
        """
        Load the agent's brain parameters.
        """
        import torch

        state_dict = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        self.brain.load_state_dict(state_dict)