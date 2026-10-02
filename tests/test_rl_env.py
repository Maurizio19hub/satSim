"""Verifiche dell'ambiente RL e della separazione training / GUI."""
import subprocess
import sys

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from rl.adcs_env import SatAttitudeEnv, log_attitude_error  # noqa: E402
from satsim.quaternion import quat_from_axis_angle  # noqa: E402

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
    assert obs.shape == (6 + 2 * n,)
    T = slice(6, 6 + n)                                   # coppia corrente nell'osservazione
    np.testing.assert_array_equal(obs[T], 0.0)            # coppia iniziale nulla

    # Azione massima per un passo: la coppia sale di Δτ_max = max_torque_rate · Δt
    np.testing.assert_allclose(env.dtau_max, 0.2 * env.tau_max)
    obs, *_ = env.step(np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(obs[T], env.dtau_max / env.tau_max, rtol=1e-6)

    # Dopo molti passi resta limitata a ±T_max
    for _ in range(20):
        obs, *_ = env.step(np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(obs[T], 1.0, rtol=1e-6)

    # Azione nulla: la coppia resta invariata
    obs2, *_ = env.step(np.zeros(n, dtype=np.float32))
    np.testing.assert_allclose(obs2[T], obs[T])


def test_reward_is_bonus_on_target_at_rest():
    env = SatAttitudeEnv()
    env.reset(seed=0)
    env.engine.reset()                     # assetto = target, ω = 0
    env._prev_omega = env.engine.omega
    env._tau[:] = 0.0
    _, r, *_ = env.step(np.zeros(env.n_wheels, dtype=np.float32))
    c = env.reward_config
    assert abs(r - (c["bonus"] + c["bonus2"])) < 1e-3


def test_log_attitude_error():
    axis = np.array([0.0, 0.6, 0.8])
    np.testing.assert_array_equal(log_attitude_error(np.array([1.0, 0, 0, 0])), 0.0)
    prev = 0.0
    for deg in (0.01, 0.1, 0.45, 10, 60, 179.9):
        e = log_attitude_error(quat_from_axis_angle(axis, np.radians(deg)))
        g = np.linalg.norm(e)
        np.testing.assert_allclose(e / g, axis, atol=1e-9)    # direzione = asse
        assert prev < g <= 1.0                                 # monotona, limitata
        prev = g
    assert np.linalg.norm(log_attitude_error(quat_from_axis_angle(axis, np.radians(0.45)))) > 0.2


def test_comparison_matches_evaluate():
    """La modalità confronto (rl/compare.py) riproduce rl/evaluate.py episodio per episodio."""
    from rl.compare import Comparison
    from rl.evaluate import evaluate
    c = Comparison("rl/pretrained/ppo_adcs_v8", seed=104)
    c.advance(10_000)
    assert c.done and abs(c.t - 100.0) < 1e-9
    ref_ppo = evaluate(c.runs[0].policy, seeds=[104])["ret"][0]
    ref_pd = evaluate("PD", seeds=[104])["ret"][0]
    assert abs(c.runs[0].ret - ref_ppo) < 1e-6
    assert abs(c.runs[1].ret - ref_pd) < 1e-6


def _serialized_functions(zip_path) -> list:
    import json
    import zipfile
    data = json.loads(zipfile.ZipFile(zip_path).read("data"))
    return [k for k, v in data.items()
            if isinstance(v, dict) and "function" in str(v.get(":type:", ""))]


def test_saved_models_are_portable(tmp_path):
    """Nessuna funzione Python serializzata nei modelli: il bytecode cambia tra
    versioni di Python (3.11 → 3.14 dava segmentation fault al caricamento)."""
    from pathlib import Path
    from stable_baselines3 import PPO
    from rl.train import PPO_CONFIG
    for p in Path("rl/pretrained").glob("*.zip"):
        assert _serialized_functions(p) == [], p
    cfg = {**PPO_CONFIG, "verbose": 0}
    PPO(env=SatAttitudeEnv(), **cfg).save(tmp_path / "m")
    assert _serialized_functions(tmp_path / "m.zip") == []


def test_wheel_speed_in_observation():
    """v9: le ultime N componenti sono Ω/Ω_max dalla telemetria; v8 senza."""
    env = SatAttitudeEnv()
    env.reset(seed=0)
    for _ in range(30):
        obs, *_ = env.step(np.ones(env.n_wheels, dtype=np.float32))
    n = env.n_wheels
    np.testing.assert_allclose(obs[6 + n:], env.last_tel.wheel_speed / env.omega_w_max, rtol=1e-5)
    assert np.all(np.abs(obs[6 + n:]) <= 1.0 + 1e-6)
    assert SatAttitudeEnv(wheel_speed_obs=False).observation_space.shape == (6 + n,)


def test_old_models_still_load():
    """Il modello v8 (osservazione senza ruote) viene associato all'ambiente giusto."""
    from rl.models import env_kwargs_for, load_model
    assert env_kwargs_for(load_model("rl/pretrained/ppo_adcs_v8")) == {"wheel_speed_obs": False}
