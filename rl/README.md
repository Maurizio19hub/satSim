# satSim — Reinforcement Learning (PPO)

Questo documento tiene traccia **solo della logica di Reinforcement Learning** del progetto: formulazione del problema, spazi, reward, algoritmo e scelte di addestramento. La fisica del simulatore è descritta nel [README principale](../README.md).

**Stato:** v2 dell'ambiente. Osservazione = assetto + velocità angolare. Reward = − errore d'assetto (lineare) − accelerazioni oltre soglia. Non ancora addestrato a convergenza.

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

| Elemento | Scelta | Stato |
|---|---|---|
| **Passo di controllo** | Un passo dell'engine, Δt = `params.dt` (0.05 s). Azione mantenuta costante sul passo (zero-order hold). | deciso |
| **Azione** | `Box([-1, 1]^N)`: coppie motore delle N ruote normalizzate. Denormalizzazione `tau = action · T_max`. I limiti fisici (coppia e saturazione in velocità) restano applicati dall'engine. | deciso |
| **Osservazione** | `Box(7)`: `[q_err (4) , ω / OMEGA_SCALE (3)]`, cioè "posizione" (assetto rispetto al target) e velocità angolare. `OMEGA_SCALE = 0.1 rad/s`. | v1 |
| **Stato iniziale** | Assetto casuale a 40–80° dal target (asse casuale), `ω` uniforme in ±0.05 rad/s, ruote ferme. Campionato con `self.np_random`. | v1 |
| **Terminazione** | Nessuna: `terminated` è sempre `False`. | v1 |
| **Troncamento** | `max_episode_steps` (default 2000 passi = 100 s). | deciso |

### Osservazione

- **Posizione = `q_err`**, il quaternione d'errore `q_target* ⊗ q` con `q_e0 ≥ 0` (rotazione più breve). È la rappresentazione dell'assetto relativa al target: vale `[1, 0, 0, 0]` quando il satellite è allineato. Le componenti sono già in [−1, 1].
- **Velocità = `ω`** in body, divisa per `OMEGA_SCALE` per portarla a valori dell'ordine di 1 (la rete neurale di PPO lavora meglio con ingressi normalizzati).
### Azione: coppia assoluta, non variazione di coppia

L'azione è la **coppia applicata direttamente** a ogni ruota nel passo, non una variazione rispetto al passo precedente. Tra due passi consecutivi (0.05 s) l'agente può quindi passare da +T_max a −T_max.

- La penalità sull'accelerazione limita l'**ampiezza** della coppia (α ∝ coppia), ma non la sua **variazione** nel tempo (il *jerk*, dα/dt).
- Rischio: una policy a "commutazione" (*chattering*) tra valori opposti. Sul satellite reale produce vibrazioni (jitter), picchi di corrente e usura dei motori.
- Alternative da valutare:
  1. azione = Δτ con τ limitata in variazione. In questo caso τ va aggiunta all'osservazione, altrimenti l'agente non conosce la coppia attuale;
  2. coppia assoluta + penalità su |a_t − a_{t−1}|;
  3. coppia assoluta + filtro passa-basso sull'azione.

Per ora si mantiene la coppia assoluta.

### Ruote non osservate

- **Velocità delle ruote non osservate** (scelta iniziale voluta). Conseguenza: lo stato non è completamente osservabile. Il momento delle ruote `h_rw` entra nella dinamica (accoppiamento giroscopico) e determina la saturazione. Con ruote che partono ferme ed episodi da 100 s l'effetto è piccolo (con il PD le ruote non superano ~900 RPM su 6000). Diventa rilevante con disturbi forti o episodi lunghi: in quel caso si aggiunge `Ω/Ω_max` all'osservazione.

## 4. Metodi dell'ambiente

`SatAttitudeEnv` in `adcs_env.py` segue l'API di Gymnasium, richiesta da Stable-Baselines3.

| Metodo | Compito |
|---|---|
| `__init__(params, max_episode_steps, render_mode, reward_config)` | Crea `SatelliteEngine`, definisce `action_space` e `observation_space`, legge i pesi della reward (`REWARD_CONFIG`, sovrascrivibili). |
| `reset(seed, options)` | Campiona le condizioni iniziali, resetta l'engine, salva `ω` iniziale come "passo precedente". Ritorna `(obs, info)`. |
| `step(action)` | Denormalizza l'azione, chiama `engine.step(tau)`, calcola l'accelerazione `α = (ω − ω_prev)/Δt`, la reward, aggiorna i valori precedenti. Ritorna `(obs, reward, terminated, truncated, info)`. |
| `_compute_reward(tel, action)` | Reward del passo (§5). |
| `_get_obs(tel)` | Vettore d'osservazione (§3). |
| `_info(tel)` | Diagnostica per ogni passo: `att_err_deg`, `alpha_deg`, `power`, `wheel_saturation`. |

Nota: il metodo si chiama `_compute_reward` e non `compute_reward` perché Stable-Baselines3 riserva quel nome agli ambienti *goal-conditioned* (`GoalEnv`) e `check_env` fallirebbe.

## 5. Reward

$$
r_t = -\,k_{err}\,\theta_t \;-\; k_{acc}\sum_{i\in\{x,y,z\}} \max\!\left(0,\ |\alpha_{i,t}| - \alpha_{max}\right)
$$

| Simbolo | Significato | Valore (`REWARD_CONFIG`) |
|---|---|---|
| $\theta_t$ | errore d'assetto `tel.att_err_deg` [°] | — |
| $\alpha_t$ | accelerazione angolare $(\omega_t-\omega_{t-1})/\Delta t$ [°/s²] | — |
| $k_{err}$ | penalità per grado di errore, per passo | `k_err = 0.01` |
| $\alpha_{max}$ | soglia di accelerazione per asse | `alpha_max_deg = 2.0` °/s² (da confermare) |
| $k_{acc}$ | penalità per °/s² oltre soglia | `k_accel = 0.01` |

**Termine d'errore.** Penalità lineare e sempre attiva: a 60° vale −0.6 per passo, a 1° vale −0.01. La reward è sempre ≤ 0 e il massimo (0) si ha solo sul target. Sull'episodio la penalità è proporzionale all'area sotto la curva θ(t). Quindi premia sia l'**arrivare presto** sia il **restare** sul target: è il motivo per cui ha sostituito il termine differenziale della v1 (§ limiti della v1).

**Termine di accelerazione.** Nullo sotto soglia, lineare oltre. L'accelerazione è ricavata dalla differenza di `ω` tra due passi, cioè è l'accelerazione media sul passo. Include anche l'effetto dei disturbi, ma questi valgono ~1e-6 N·m e sono trascurabili rispetto alla soglia.

**Scelta della soglia.** Riferimenti misurati sul modello:

| Grandezza | Valore |
|---|---|
| Accelerazione massima ottenibile dalle ruote, assi x/y | ≈ 10.6 °/s² |
| Accelerazione massima ottenibile dalle ruote, asse z | ≈ 53 °/s² (J_zz è 5 volte più piccolo) |
| PD, percentile 99 di \|α\| (20 episodi) | 1.4 °/s² |
| PD, massimo di \|α\| | 8.4 °/s² |

Riferimenti reali:

| Fonte | Valore |
|---|---|
| MinXSS-1 (CubeSat 3U, ADCS BCT XACT): default operativo di picco | 1 °/s² di accelerazione, 6 °/s di velocità |
| MinXSS-1: capacità dell'hardware | ~25 °/s² |
| Star tracker ST200: velocità massima tollerata | 0.3 °/s (tip/tilt), 0.6 °/s (roll) |
| XACT: errore di tracking d'assetto invariato fino a | ~1.1 °/s |

Non esiste una soglia universale di "sicurezza" sull'accelerazione angolare. Per un 3U rigido, senza pannelli dispiegati, le sollecitazioni strutturali sono trascurabili: a 10 °/s² l'accelerazione lineare all'estremità del satellite è ~0.03 m/s². I limiti reali sono operativi:
- la **velocità** angolare massima che star tracker e giroscopi tollerano;
- il momento angolare disponibile nelle ruote;
- la potenza;
- il jitter, e i modi flessibili se ci sono appendici.

Il PD del simulatore non è stato progettato con un limite di accelerazione: il suo picco di 8.4 °/s² dipende dai guadagni scelti, non da un requisito. Opzioni per $\alpha_{max}$:
- **1 °/s²**: default operativo reale (MinXSS);
- **2 °/s²**: valore attuale, appena sopra il comportamento tipico del PD (p99 = 1.4);
- **picco del PD (~8.4 °/s²)**: "mai più brusco del PD", ma vicino al limite fisico delle ruote, quindi la penalità non sarebbe quasi mai attiva.

Fonti: [MinXSS-1 On-Orbit Pointing and Power Performance (arXiv:1706.06967)](https://arxiv.org/abs/1706.06967); [Nanobob, ST200 (arXiv:1711.01886)](https://arxiv.org/pdf/1711.01886).

**Verifica v2 (episodio con seed 1, 2000 passi).**

| Policy | Return | Errore finale |
|---|---|---|
| Azioni nulle (satellite libero) | −2539 | 121° |
| Azioni casuali | −2915 | 103° |
| PD al posto dell'agente | −75 | 0.003° |

### Storia: v1 (reward differenziale)

La v1 usava $r_t = k_{prog}(\theta_{t-1}-\theta_t)$ al posto del termine d'errore. È stata abbandonata per due motivi:
- **Somma telescopica.** Sull'episodio vale $k_{prog}(\theta_0-\theta_T)$: conta solo dove il satellite finisce, non quanto velocemente ci arriva.
- **Nessun incentivo a restare sul target.** Una volta raggiunto il target il segnale è ~0, quindi oscillarci attorno non costa nulla.

Verifica v1 (seed 1): azioni nulle −43.4, PD +76.5.

### Limiti noti

1. **Scala dei due termini.** Con una policy casuale la penalità d'accelerazione è dello stesso ordine del termine d'errore (~0.1–0.3 per passo). Se l'agente resta bloccato nel minimo locale "azioni piccole", si riduce `k_accel`.
2. **Coppia assoluta senza limite sulla variazione** (chattering), vedi §3.
3. **Osservazione parziale** (ruote non osservate), vedi §3.

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

### Ambienti paralleli (`n_envs = 4`)

- **Cosa sono.** 4 copie indipendenti del simulatore, cioè 4 satelliti con condizioni iniziali diverse. **C'è un solo agente**, cioè una sola rete neurale: a ogni passo calcola in un colpo le 4 azioni, una per satellite, e ogni copia avanza di Δt.
- **Rollout.** Ogni aggiornamento di PPO usa `n_steps × n_envs = 2048 × 4 = 8192` transizioni. Quando un episodio finisce in una copia, quella si resetta da sola e le altre continuano.
- **Conteggio dei passi.** `total_timesteps` conta la somma dei passi di tutte le copie: 1 M passi totali = 250 k per copia.
- **Esecuzione.** Con `DummyVecEnv` (default di `make_vec_env`) le 4 copie girano **in sequenza nello stesso processo**, quindi non c'è vero parallelismo di CPU. Il vantaggio è statistico: i dati di ogni aggiornamento vengono da 4 episodi diversi e sono meno correlati. Per usare più core si passa `vec_env_cls=SubprocVecEnv`.

Throughput misurato nel cloud: ~460 passi/s con 4 ambienti, cioè ~1 M passi in ~35 min.

**Prova breve (40 000 passi).** La pipeline funziona end-to-end. `ep_rew_mean` passa da −378 a −366: sono troppo pochi passi (~20 episodi per ambiente) per imparare la manovra. Serve un addestramento lungo.

## 8. Decisioni aperte

- Valore di `alpha_max_deg`: 1, 2 o picco del PD (§5).
- Azione: coppia assoluta o variazione di coppia (§3).
- Taratura di `k_err` e `k_accel`.
- Aggiunta delle velocità delle ruote all'osservazione.
- Condizioni di terminazione anticipata.
- Distribuzione delle condizioni iniziali ed eventuale curriculum.
- Disturbi ambientali attivi o no durante l'addestramento.
- Normalizzazione di osservazioni e reward con `VecNormalize`.
- Metriche di valutazione rispetto al PD.

## 9. Registro di sviluppo RL

### 2026-09-26 — Scheletro
- Creata la cartella `rl/` con `adcs_env.py`.
- `SatAttitudeEnv(gym.Env)` con metodi vuoti: `__init__`, `reset`, `step`, `_compute_reward`, `_get_obs`.
- Configurazione PPO (`PPO_CONFIG`) e training (`TRAIN_CONFIG`, `train()`) con Stable-Baselines3.
- Aggiunte le dipendenze `gymnasium`, `stable-baselines3`, `tensorboard`.

### 2026-09-28 — v1 dell'ambiente
- Implementati `__init__`, `reset`, `step`, `_get_obs`, `_compute_reward`.
- Osservazione: `q_err` + `ω` normalizzata (7 valori).
- Reward: progresso verso il target (differenza di errore in gradi) − penalità sulle accelerazioni angolari oltre 2 °/s².
- `compute_reward` rinominato `_compute_reward` (conflitto con `GoalEnv` in `check_env`).
- `check_env` superato. Prova PPO da 40k passi eseguita.

### 2026-09-28 — v2 della reward
- Termine differenziale sostituito da penalità lineare sull'errore: `−k_err · θ` con `k_err = 0.01`.
- Termine sulle accelerazioni oltre soglia invariato.
- Documentati: azione come coppia assoluta (e rischio di chattering), riferimenti reali per la soglia d'accelerazione, funzionamento degli ambienti paralleli.
