"""Verifiche dell'ambiente RL e della separazione training / GUI."""
import subprocess
import sys

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from rl.adcs_env import SatAttitudeEnv  # noqa: E402

GUI_MODULES = ("PySide6", "pyqtgraph", "OpenGL", "gui")


def test_training_is_headless():
    """Importare lo script di training non deve caricare nessun modulo grafico."""
    code = ("import sys, rl.train; "
            f"print([m for m in {GUI_MODULES!r} if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_check_env():
    from stable_baselines3.common.env_checker import check_env
    check_env(SatAttitudeEnv(), warn=True)


def test_action_is_rate_limited_torque_change():
    env = SatAttitudeEnv()
    obs, _ = env.reset(seed=0)
    n = env.n_wheels
    assert obs.shape == (7 + n,)
    np.testing.assert_array_equal(obs[7:], 0.0)          # coppia iniziale nulla

    # Azione massima per un passo: la coppia sale di DTAU_MAX_FRAC · T_max
    obs, *_ = env.step(np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(obs[7:], env.dtau_max / env.tau_max, rtol=1e-6)

    # Dopo molti passi resta limitata a ±T_max
    for _ in range(20):
        obs, *_ = env.step(np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(obs[7:], 1.0, rtol=1e-6)

    # Azione nulla: la coppia resta invariata
    obs2, *_ = env.step(np.zeros(n, dtype=np.float32))
    np.testing.assert_allclose(obs2[7:], obs[7:])


def test_reward_is_zero_on_target_at_rest():
    env = SatAttitudeEnv()
    env.reset(seed=0)
    env.engine.reset()                     # assetto = target, ω = 0
    env._prev_omega = env.engine.omega
    env._tau[:] = 0.0
    _, r, *_ = env.step(np.zeros(env.n_wheels, dtype=np.float32))
    assert abs(r) < 1e-3
