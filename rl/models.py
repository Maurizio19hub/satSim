"""
Caricamento dei modelli PPO, portabile tra versioni di Python.

Stable-Baselines3 salva nel file .zip anche il learning rate. Se è una
funzione Python (come nei modelli fino alla v8), viene serializzata come
bytecode, che cambia tra versioni di Python: un modello salvato con
Python 3.11 e caricato con 3.14 va in segmentation fault. Per questo il
learning rate salvato non viene mai deserializzato: lo si sostituisce con
LR_SCHEDULE, una classe di Stable-Baselines3 (serializzata per riferimento,
quindi portabile).
"""
from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.utils import LinearSchedule

# Learning rate lineare da 3e-4 a 0 su tutto l'addestramento.
LR_SCHEDULE = LinearSchedule(start=3e-4, end=0.0, end_fraction=1.0)


def load_model(path, env=None, device: str = "cpu", **kwargs) -> PPO:
    """PPO.load senza deserializzare il learning rate salvato nel modello.

    Per la sola inferenza il learning rate non conta; per --resume si usa
    LR_SCHEDULE, cioè quello del codice attuale.
    """
    custom = {"learning_rate": LR_SCHEDULE, "lr_schedule": LR_SCHEDULE}
    return PPO.load(path, env=env, device=device, custom_objects=custom, **kwargs)


def env_kwargs_for(model) -> dict:
    """Parametri di SatAttitudeEnv compatibili con l'osservazione del modello.

    I modelli fino alla v8 osservano 6 + N valori (senza velocità delle ruote),
    dalla v9 6 + 2N. Si deduce dal modello stesso, così valutazione e
    confronto funzionano con entrambi senza opzioni a mano.
    """
    from rl.adcs_env import SatAttitudeEnv
    n_obs = model.observation_space.shape[0]
    for flag in (True, False):
        if SatAttitudeEnv(wheel_speed_obs=flag).observation_space.shape[0] == n_obs:
            return {"wheel_speed_obs": flag}
    raise ValueError(f"Il modello osserva {n_obs} valori: nessuna versione dell'ambiente "
                     f"corrisponde. Va riaddestrato con il codice attuale.")
