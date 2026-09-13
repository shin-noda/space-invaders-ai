from collections import deque
import cv2
import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box


class AtariPreprocess(Wrapper):
    """
    Preprocess Atari image observations.

    Handles:
    - Frame max-pooling across consecutive steps to eliminate sprite flickering
    - Cropping, grayscaling, and resizing to target output size
    - Normalizing frame pixel values to [0.0, 1.0]
    - Reward clipping via sign(reward) to prevent gradient instability
    """

    def __init__(
        self,
        env,
        crop=None,
        output_size=(84, 84),
        grayscale=True,
    ):
        super().__init__(env)

        self.crop = crop
        self.output_size = output_size
        self.grayscale = grayscale

        # Buffer to keep track of the last 2 raw frames for max-pooling
        self._frame_buffer = deque(maxlen=2)

        if grayscale:
            shape = (
                output_size[1],
                output_size[0],
            )
        else:
            shape = (
                3,
                output_size[1],
                output_size[0],
            )

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=shape,
            dtype=np.float32,
        )

    def _preprocess(self, frame):
        """
        Convert one raw Atari frame into a normalized
        observation.
        """

        # Crop: (top, bottom, left, right)
        if self.crop is not None:
            top, bottom, left, right = self.crop

            frame = frame[
                top:bottom,
                left:right,
            ]

        # Convert RGB -> grayscale
        if self.grayscale:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2GRAY,
            )

        # Resize
        frame = cv2.resize(
            frame,
            self.output_size,
            interpolation=cv2.INTER_AREA,
        )

        # Normalize to [0, 1]
        frame = frame.astype(
            np.float32
        ) / 255.0

        # For grayscale: keep shape as (84, 84).
        # For RGB: convert (H, W, C) -> (C, H, W).
        if not self.grayscale:
            frame = np.transpose(
                frame,
                (2, 0, 1),
            )

        return frame

    def reset(self, **kwargs):
        obs, info = self.env.reset(
            **kwargs
        )

        self._frame_buffer.clear()
        self._frame_buffer.append(obs)

        return (
            self._preprocess(obs),
            info,
        )

    def step(self, action):
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        self._frame_buffer.append(obs)

        # Max-pool across last two frames to eliminate Atari sprite flicker
        if len(self._frame_buffer) == 2:
            max_frame = np.maximum(
                self._frame_buffer[0],
                self._frame_buffer[1],
            )
        else:
            max_frame = obs

        # Clip reward to [-1, 1]
        clipped_reward = np.sign(reward)

        return (
            self._preprocess(max_frame),
            clipped_reward,
            terminated,
            truncated,
            info,
        )


def make_preprocessed_atari_env(
    env,
    crop=None,
    output_size=(84, 84),
    grayscale=True,
    stack_frames=4,
):
    """
    Apply Atari preprocessing and frame stacking.
    """

    env = AtariPreprocess(
        env,
        crop=crop,
        output_size=output_size,
        grayscale=grayscale,
    )

    if stack_frames > 1:
        import gymnasium as gym

        env = gym.wrappers.FrameStackObservation(
            env,
            stack_size=stack_frames,
        )

    return env