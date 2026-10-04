"""
Loading PPO models in a way that is portable across Python versions.

Stable-Baselines3 also stores the learning rate in the .zip file. If it is a
Python function (as in the models up to v8), it is serialised as bytecode,
which changes between Python versions: a model saved with Python 3.11 and
loaded with 3.14 crashes with a segmentation fault. For this reason the saved
learning rate is never deserialised: it is replaced with LR_SCHEDULE, a
Stable-Baselines3 class (serialised by reference, hence portable).
"""
from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule

# Learning rate decreasing linearly from 3e-4 to 0 over the whole training.
LR_SCHEDULE = LinearSchedule(start=3e-4, end=0.0, end_fraction=1.0)


def load_model(path, env=None, device: str = "cpu", **kwargs) -> PPO:
    """PPO.load without deserialising the learning rate stored in the model.

    For inference only the learning rate is irrelevant; for --resume
    LR_SCHEDULE is used, i.e. the one of the current code.
    """
    custom = {"learning_rate": LR_SCHEDULE, "lr_schedule": LR_SCHEDULE}
    return PPO.load(path, env=env, device=device, custom_objects=custom, **kwargs)


def env_kwargs_for(model) -> dict:
    """SatAttitudeEnv parameters compatible with the model's observation.

    Models up to v8 observe 6 + N values (no wheel speeds), from v9 on
    6 + 2N. The format is inferred from the model itself, so evaluation and
    comparison work with both without manual options.
    """
    from rl.adcs_env import SatAttitudeEnv
    n_obs = model.observation_space.shape[0]
    for flag in (True, False):
        if SatAttitudeEnv(wheel_speed_obs=flag).observation_space.shape[0] == n_obs:
            return {"wheel_speed_obs": flag}
    raise ValueError(f"The model observes {n_obs} values: no version of the environment "
                     f"matches. It must be retrained with the current code.")
