import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class REINFORCE:
    """
    REINFORCE policy-gradient algorithm.

    This class handles:
    - action sampling from policy logits
    - discounted returns
    - policy loss
    - entropy regularization
    - gradient accumulation
    - optimizer updates
    """

    def __init__(
        self,
        brain,
        lr=1e-4,
        gamma=0.99,
        entropy_coef=0.01,
        device="cpu",
    ):
        self.brain = brain
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device

        self.optimizer = optim.Adam(
            self.brain.parameters(),
            lr=lr,
        )

    def select_action(self, state):
        """
        Sample an action from the policy.

        Returns
        -------
        action : int
        log_prob : torch.Tensor
        entropy : torch.Tensor
        """

        logits = self.brain(state)

        distribution = Categorical(
            logits=logits,
        )

        action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return (
            action,
            log_prob,
            entropy,
        )

    def calculate_returns(self, rewards):
        """
        Calculate discounted returns for one trajectory.
        """

        R = 0.0
        returns = []

        for reward in reversed(rewards):
            reward = np.sign(reward)

            if reward != 0:
                R = 0.0

            R = reward + self.gamma * R

            returns.insert(0, R)

        returns = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        if len(returns) > 1:
            returns = (
                returns - returns.mean()
            ) / (
                returns.std() + 1e-8
            )

        return returns

    def calculate_loss(
        self,
        log_probs,
        entropies,
        rewards,
    ):
        """
        Calculate the REINFORCE loss for one trajectory.
        """

        returns = self.calculate_returns(
            rewards
        )

        log_probs = torch.stack(
            log_probs
        )

        entropies = torch.stack(
            entropies
        )

        policy_loss = -(
            log_probs * returns
        ).sum()

        entropy_loss = entropies.sum()

        loss = (
            policy_loss
            - self.entropy_coef * entropy_loss
        )

        return loss

    def backward(
        self,
        log_probs,
        entropies,
        rewards,
        batch_size=1,
    ):
        """
        Calculate the loss and accumulate gradients.

        The optimizer is not stepped here.
        """

        loss = self.calculate_loss(
            log_probs,
            entropies,
            rewards,
        )

        loss = loss / batch_size

        loss.backward()

        return loss.item()

    def step(self, max_grad_norm=1.0):
        """
        Perform one optimizer update.
        """

        nn.utils.clip_grad_norm_(
            self.brain.parameters(),
            max_norm=max_grad_norm,
        )

        self.optimizer.step()
        self.optimizer.zero_grad()

    def save_checkpoint(
        self,
        path,
        episode,
        scores_window,
    ):
        """
        Save the algorithm state and training metadata.
        """

        torch.save(
            {
                "episode": episode,
                "brain_state_dict": (
                    self.brain.state_dict()
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
        """
        Load the algorithm state and return
        the saved training metadata.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.brain.load_state_dict(
            checkpoint["brain_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        return checkpoint