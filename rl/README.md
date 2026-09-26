# satSim — Reinforcement Learning (PPO)

Questo documento tiene traccia **solo della logica di Reinforcement Learning** del progetto: formulazione del problema, spazi, reward, algoritmo e scelte di addestramento. La fisica del simulatore è descritta nel [README principale](../README.md).

**Stato:** scheletro. L'ambiente e la configurazione di Stable-Baselines3 sono impostati. I metodi dell'ambiente sono vuoti.

---

## Indice

1. [Obiettivo](#1-obiettivo)
2. [Struttura della cartella](#2-struttura-della-cartella)
3. [Formulazione come MDP](#3-formulazione-come-mdp)
4. [Metodi dell'ambiente](#4-metodi-dellambiente)
5. [Reward](#5-reward)
6. [Algoritmo: PPO](#6-algoritmo-ppo)
7. [Addestramento](#7-addestramento)
8. [Decisioni aperte](#8-decisioni-aperte)
9. [Registro di sviluppo RL](#9-registro-di-sviluppo-rl)

---

## 1. Obiettivo

Addestrare un agente che porti il CubeSat dall'assetto iniziale all'assetto target `q_target` e ce lo mantenga. L'agente comanda direttamente le coppie motore delle ruote di reazione e sostituisce il controllore PD sui quaternioni (`satsim/controller.py`).

Il PD resta il **baseline** di confronto: l'agente deve eguagliarlo o superarlo su precisione, tempo di assestamento, energia consumata e margine di saturazione delle ruote.

## 2. Struttura della cartella

```
rl/
├── __init__.py
├── adcs_env.py     # ambiente Gymnasium SatAttitudeEnv + configurazione e training PPO (SB3)
└── README.md       # questo documento
```

Dipendenze aggiuntive (in `requirements.txt`): `gymnasium`, `stable-baselines3`, `tensorboard`.

Le dipendenze vanno in un solo verso: `rl → satsim`. L'engine non importa nulla da `rl/`.

## 3. Formulazione come MDP

| Elemento | Scelta prevista | Stato |
|---|---|---|
| **Passo di controllo** | Un passo dell'engine, Δt = `params.dt` (0.05 s). Azione mantenuta costante sul passo (zero-order hold). | deciso |
| **Azione** | `Box([-1, 1]^N)`: coppie motore delle N ruote normalizzate. Denormalizzazione `tau = action · T_max`. I limiti fisici (coppia e saturazione in velocità) restano applicati dall'engine. | deciso |
| **Osservazione** | Candidati: `q_err` (4), `ω` (3), velocità delle ruote normalizzate `Ω/Ω_max` (N). | da definire |
| **Stato iniziale** | Candidato: assetto casuale a 40–80° dal target, `ω` casuale in ±0.05 rad/s, ruote ferme (come `ClosedLoopSimulation.reset`). | da definire |
| **Terminazione** | Da definire (es. divergenza di `ω` oltre una soglia). | da definire |
| **Troncamento** | `max_episode_steps` (default 2000 passi = 100 s). | deciso |

## 4. Metodi dell'ambiente

`SatAttitudeEnv` in `adcs_env.py` segue l'API di Gymnasium, richiesta da Stable-Baselines3.

| Metodo | Compito | Stato |
|---|---|---|
| `__init__(params, max_episode_steps, render_mode)` | Crea `SatelliteEngine`, definisce `action_space` e `observation_space`, inizializza contatori e pesi della reward. | vuoto |
| `reset(seed, options)` | Chiama `super().reset(seed=seed)`, campiona le condizioni iniziali con `self.np_random`, resetta l'engine. Ritorna `(obs, info)`. | vuoto |
| `step(action)` | Denormalizza l'azione, chiama `engine.step(tau)`, calcola obs, reward, `terminated`, `truncated`, `info`. | vuoto |
| `compute_reward(tel, action)` | Calcola la reward scalare del passo dalla telemetria. | vuoto |
| `_get_obs(tel)` | Costruisce il vettore d'osservazione dalla telemetria. | vuoto |

I metodi vuoti sollevano `NotImplementedError`.

## 5. Reward

Da definire. Termini candidati, tutti disponibili nella `Telemetry` restituita da `engine.step()`:

| Termine | Grandezza | Scopo |
|---|---|---|
| Errore d'assetto | `tel.att_err_deg` oppure `tel.q_err` | raggiungere il target |
| Velocità angolare | `tel.omega` | smorzare il moto, evitare oscillazioni |
| Sforzo di controllo / energia | `action`, `tel.power_total` | ridurre il consumo |
| Saturazione ruote | `tel.wheel_saturation` | mantenere margine di momento angolare |

Forma, pesi e scala sono ancora da scegliere.

## 6. Algoritmo: PPO

Si usa **PPO** (*Proximal Policy Optimization*) di Stable-Baselines3, con policy `MlpPolicy`. PPO è on-policy, supporta azioni continue (`Box`) e funziona con ambienti vettorizzati.

Iperparametri iniziali (`PPO_CONFIG` in `adcs_env.py`, default di SB3, da tarare):

| Parametro | Valore |
|---|---|
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |

## 7. Addestramento

```bash
pip install -r requirements.txt
python -m rl.adcs_env                          # avvia train() con TRAIN_CONFIG
tensorboard --logdir runs/ppo_adcs             # curve di apprendimento
```

`train()` esegue, in ordine:
1. `check_env` di SB3 per verificare la conformità all'API Gymnasium;
2. `make_vec_env` con `n_envs = 4` ambienti paralleli;
3. `PPO.learn` per `total_timesteps = 1 000 000`;
4. salvataggio del modello in `models/ppo_adcs.zip`.

Le cartelle `runs/` e `models/` sono escluse da git.

> Finché i metodi dell'ambiente sono vuoti, `train()` si ferma subito con `NotImplementedError`.

## 8. Decisioni aperte

- Contenuto e normalizzazione dell'osservazione.
- Forma e pesi della reward.
- Condizioni di terminazione anticipata.
- Distribuzione delle condizioni iniziali ed eventuale curriculum.
- Disturbi ambientali attivi o no durante l'addestramento.
- Normalizzazione di osservazioni e reward con `VecNormalize`.
- Metriche di valutazione rispetto al PD.

## 9. Registro di sviluppo RL

### 2026-09-26 — Scheletro
- Creata la cartella `rl/` con `adcs_env.py`.
- `SatAttitudeEnv(gym.Env)` con metodi vuoti: `__init__`, `reset`, `step`, `compute_reward`, `_get_obs`.
- Configurazione PPO (`PPO_CONFIG`) e training (`TRAIN_CONFIG`, `train()`) con Stable-Baselines3.
- Aggiunte le dipendenze `gymnasium`, `stable-baselines3`, `tensorboard`.
