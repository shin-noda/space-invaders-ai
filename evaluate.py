import sys

import torch

from src.environments.atari import make_atari_env
from src.environments.preprocess import make_preprocessed_atari_env
from src.networks.cnn import AtariCNN
from src.algorithms.dqn import DQN
from src.evaluation.evaluator import Evaluator


GAME = "SpaceInvaders"
STACK_FRAMES = 4
NUM_ACTIONS = 6


# --------------------------------------------------
# Command-line argument
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python evaluate.py <checkpoint>")
    print("Example: python evaluate.py checkpoints/checkpoint_ep3500.pt")
    sys.exit(1)

CHECKPOINT = sys.argv[1]


# --------------------------------------------------
# Device
# --------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
)

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print()


# --------------------------------------------------
# Environment
# --------------------------------------------------

env = make_atari_env(
    game=GAME,
    stack_frames=1,
    render_mode="human",
)

env = make_preprocessed_atari_env(
    env,
    output_size=(84, 84),
    grayscale=True,
    stack_frames=STACK_FRAMES,
)


# --------------------------------------------------
# Networks
# --------------------------------------------------

brain = AtariCNN(
    in_channels=STACK_FRAMES,
    num_actions=NUM_ACTIONS,
)

target_brain = AtariCNN(
    in_channels=STACK_FRAMES,
    num_actions=NUM_ACTIONS,
)

brain = brain.to(DEVICE)
target_brain = target_brain.to(DEVICE)


# --------------------------------------------------
# DQN
# --------------------------------------------------

algorithm = DQN(
    brain=brain,
    target_brain=target_brain,
    lr=1e-4,
    gamma=0.99,
    replay_capacity=100_000,
    batch_size=32,
    device=DEVICE,
)


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

algorithm.load_checkpoint(CHECKPOINT)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

evaluator = Evaluator(
    env=env,
    brain=brain,
    device=DEVICE,
)

scores = evaluator.evaluate(
    num_episodes=1,
    render=True,
    render_delay=0.02,
)

evaluator.summarize(scores)


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

env.close()