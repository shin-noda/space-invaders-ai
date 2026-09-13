import glob
import os

import torch

from src.environments.atari import make_atari_env
from src.environments.preprocess import make_preprocessed_atari_env
from src.networks.cnn import AtariCNN
from src.algorithms.dqn import DQN
from src.training.trainer import Trainer


GAME = "SpaceInvaders"
NUM_EPISODES = 10_000

STACK_FRAMES = 4
NUM_ACTIONS = 6

LEARNING_RATE = 1e-4
GAMMA = 0.99

REPLAY_CAPACITY = 100_000
BATCH_SIZE = 32

TARGET_UPDATE_INTERVAL = 10_000
WARMUP_STEPS = 10_000
EXPLORATION_STEPS = 1_000_000

EPSILON_START = 1.0
EPSILON_END = 0.1

SAVE_INTERVAL = 50

CHECKPOINT_DIR = "checkpoints"


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
)


def find_latest_checkpoint(
    checkpoint_dir,
):
    pattern = os.path.join(
        checkpoint_dir,
        "checkpoint_ep*.pt",
    )

    checkpoints = glob.glob(
        pattern
    )

    if not checkpoints:
        return None

    def episode_number(path):
        filename = os.path.basename(
            path
        )

        number = filename[
            len("checkpoint_ep"):-3
        ]

        return int(number)

    return max(
        checkpoints,
        key=episode_number,
    )


env = make_atari_env(
    game=GAME,
    stack_frames=1,
)

env = make_preprocessed_atari_env(
    env,
    output_size=(84, 84),
    grayscale=True,
    stack_frames=STACK_FRAMES,
)


brain = AtariCNN(
    in_channels=STACK_FRAMES,
    num_actions=NUM_ACTIONS,
)

target_brain = AtariCNN(
    in_channels=STACK_FRAMES,
    num_actions=NUM_ACTIONS,
)

brain = brain.to(
    DEVICE
)

target_brain = target_brain.to(
    DEVICE
)


algorithm = DQN(
    brain=brain,
    target_brain=target_brain,
    lr=LEARNING_RATE,
    gamma=GAMMA,
    replay_capacity=REPLAY_CAPACITY,
    batch_size=BATCH_SIZE,
    device=DEVICE,
)


trainer = Trainer(
    env=env,
    algorithm=algorithm,
    device=DEVICE,
    checkpoint_dir=CHECKPOINT_DIR,
    save_interval=SAVE_INTERVAL,
)


if __name__ == "__main__":

    print()
    print("========================================")
    print(" Atari DQN Training")
    print("========================================")
    print(f"Game:     {GAME}")
    print(f"Device:   {DEVICE}")
    print(f"Episodes: {NUM_EPISODES}")
    print("========================================")
    print()

    latest_checkpoint = (
        find_latest_checkpoint(
            CHECKPOINT_DIR
        )
    )

    if latest_checkpoint is None:

        print(
            "No checkpoint found."
        )

        start_episode = 1
        epsilon = EPSILON_START

    else:

        print(
            f"Resuming from: "
            f"{latest_checkpoint}"
        )

        checkpoint = (
            algorithm.load_checkpoint(
                latest_checkpoint
            )
        )

        last_episode = checkpoint[
            "episode"
        ]

        epsilon = checkpoint[
            "epsilon"
        ]

        total_steps = checkpoint.get(
            "total_steps",
            0,
        )

        start_episode = (
            last_episode + 1
        )

        print(
            f"Last checkpoint: "
            f"Episode {last_episode}"
        )

        print(
            f"Starting from:   "
            f"Episode {start_episode}"
        )

        print(
            f"Epsilon:         "
            f"{epsilon:.4f}"
        )

        print(
            f"Total steps:     "
            f"{total_steps:,}"
        )

        print(
            f"Replay buffer:   "
            f"{len(algorithm.replay_buffer):,}"
            f" transitions"
        )

    print()

    trainer.train_dqn(
        num_episodes=NUM_EPISODES,
        start_episode=start_episode,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        exploration_steps=EXPLORATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        target_update_interval=(
            TARGET_UPDATE_INTERVAL
        ),
    )

    env.close()