import gymnasium as gym
import ale_py


def make_atari_env(
    game,
    render_mode=None,
    stack_frames=4,
):
    """
    Create a single Atari environment wrapped with episode stat tracking.

    Parameters
    ----------
    game : str
        Atari game name, e.g. "Pong" or "SpaceInvaders".
    render_mode : str | None
        Gymnasium render mode, e.g. "human".
    stack_frames : int
        Number of consecutive frames to stack.

    Returns
    -------
    gym.Env
        Configured Atari environment.
    """

    gym.register_envs(ale_py)

    env = gym.make(
        f"ALE/{game}-v5",
        render_mode=render_mode,
    )

    # Track raw unclipped episode score ("r") and length ("l") in info dict
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if stack_frames > 1:
        env = gym.wrappers.FrameStackObservation(
            env,
            stack_size=stack_frames,
        )

    return env


def make_vector_atari_env(
    game,
    num_envs=8,
    stack_frames=4,
):
    """
    Create multiple Atari environments running in parallel.

    Parameters
    ----------
    game : str
        Atari game name, e.g. "Pong" or "SpaceInvaders".
    num_envs : int
        Number of parallel environments.
    stack_frames : int
        Number of consecutive frames to stack.

    Returns
    -------
    gym.vector.VectorEnv
        Vectorized Atari environment.
    """

    def env_fn():
        return make_atari_env(
            game=game,
            render_mode=None,
            stack_frames=stack_frames,
        )

    return gym.vector.SyncVectorEnv(
        [env_fn for _ in range(num_envs)]
    )


if __name__ == "__main__":
    env = make_atari_env(
        game="SpaceInvaders",
        stack_frames=4,
    )

    obs, info = env.reset()

    print("✅ Atari environment initialized!")
    print(f"Observation Shape: {obs.shape}")
    print(f"Observation Dtype: {obs.dtype}")

    env.close()