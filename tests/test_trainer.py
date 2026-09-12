import numpy as np

from src.environments.atari import make_atari_env
from src.environments.preprocess import (
    make_preprocessed_atari_env,
)
from src.networks.cnn import AtariCNN
from src.algorithms.dqn import DQN
from src.training.trainer import Trainer


def test_trainer():
    device = "cpu"

    # Environment
    env = make_atari_env(
        game="SpaceInvaders",
        stack_frames=1,
    )

    env = make_preprocessed_atari_env(
        env,
        output_size=(84, 84),
        grayscale=True,
        stack_frames=4,
    )

    # Networks
    brain = AtariCNN(
        in_channels=4,
        num_actions=6,
    )

    target_brain = AtariCNN(
        in_channels=4,
        num_actions=6,
    )

    # DQN
    dqn = DQN(
        brain=brain,
        target_brain=target_brain,
        lr=1e-4,
        gamma=0.99,
        replay_capacity=100,
        batch_size=4,
        device=device,
    )

    # Trainer
    trainer = Trainer(
        env=env,
        algorithm=dqn,
        device=device,
        checkpoint_dir="checkpoints/test",
        replaybuffer_dir="replaybuffer/test",
        save_interval=10,
    )

    # Run only one episode.
    scores = trainer.train_dqn(
        num_episodes=1,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay=1.0,
        target_update_interval=100,
    )

    print()
    print("Scores:", scores)

    assert len(scores) == 1
    assert np.isfinite(scores[0])

    env.close()

    print("✅ Trainer test passed!")


if __name__ == "__main__":
    test_trainer()