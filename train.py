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

# Epsilon decays based on episodes.
# It reaches EPSILON_END around episode 9,500.
EXPLORATION_EPISODES = 9_500

EPSILON_START = 1.0
EPSILON_END = 0.1

SAVE_INTERVAL = 50

CHECKPOINT_DIR = "checkpoints"
REPLAYBUFFER_DIR = os.path.join(
    CHECKPOINT_DIR,
    "replay_buffers",
)


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


def find_replay_buffer(
    replaybuffer_dir,
    episode,
):
    replay_path = os.path.join(
        replaybuffer_dir,
        f"replay_ep{episode}.npz",
    )

    if os.path.exists(
        replay_path
    ):
        return replay_path

    return None


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
    print(
        f"Epsilon:  {EPSILON_START:.1f}"
        f" -> {EPSILON_END:.1f}"
        f" over {EXPLORATION_EPISODES:,} episodes"
    )
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

        total_steps = int(
            checkpoint.get(
                "total_steps",
                0,
            )
        )

        # Continue from the checkpoint's
        # environment step count.
        algorithm.total_steps = (
            total_steps
        )

        start_episode = (
            last_episode + 1
        )

        # Restore the rolling score window.
        scores_window = checkpoint.get(
            "scores_window",
            [],
        )

        trainer._restore_scores(
            scores_window
        )

        epsilon = checkpoint.get(
            "epsilon",
            EPSILON_START,
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
            f"Saved epsilon:   "
            f"{epsilon:.4f}"
        )

        print(
            f"Total steps:     "
            f"{total_steps:,}"
        )

        # Load the replay buffer that belongs
        # to the same checkpoint.
        replay_path = (
            find_replay_buffer(
                REPLAYBUFFER_DIR,
                last_episode,
            )
        )

        if replay_path is None:

            print(
                "WARNING: Replay buffer "
                f"for episode {last_episode} "
                "was not found."
            )

            print(
                "Starting with an empty "
                "replay buffer."
            )

        else:

            algorithm.load_replay_buffer(
                replay_path
            )

            print(
                f"Replay buffer:   "
                f"{len(algorithm.replay_buffer):,}"
                f" transitions"
            )

        print(
            f"Score window:    "
            f"{len(trainer.scores)} episodes"
        )

    print()

    trainer.train_dqn(
        num_episodes=NUM_EPISODES,
        start_episode=start_episode,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        exploration_episodes=EXPLORATION_EPISODES,
        warmup_steps=WARMUP_STEPS,
        target_update_interval=(
            TARGET_UPDATE_INTERVAL
        ),
    )

    env.close()