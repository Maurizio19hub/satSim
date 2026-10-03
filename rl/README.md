# satSim — Reinforcement Learning (PPO)

Questo documento tiene traccia **solo della logica di Reinforcement Learning** del progetto: formulazione del problema, spazi, reward, algoritmo e scelte di addestramento. La fisica del simulatore è descritta nel [README principale](../README.md).

**Stato:** v11 dell'ambiente (ramo `claude/rl-v10-wheel-penalty`, da addestrare e testare in locale).
- **Azione:** variazione di coppia delle ruote.
- **Osservazione:** errore d'assetto in scala logaritmica + velocità angolare + coppia corrente + velocità delle ruote.
- **Reward:** − errore d'assetto (lineare + logaritmica) − accelerazioni oltre soglia − velocità delle ruote nello spazio nullo + bonus sotto 0.1° e 0.01°.
- **Modello di riferimento:** v8, in `rl/pretrained/`. Il modello v9 è ancora da addestrare.
- Il dettaglio di ogni versione è nel registro (§9).

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
├── adcs_env.py     # ambiente Gymnasium SatAttitudeEnv (solo fisica, niente SB3 né GUI)
├── train.py        # addestramento PPO con Stable-Baselines3 (headless)
├── evaluate.py     # confronto di un modello con PD e riferimenti sui seed di test
├── compare.py      # confronto passo-passo PPO vs PD per la GUI (python main.py --compare)
├── pretrained/
│   └── ppo_adcs_v8.zip   # modello v8 migliore (1.7 M passi), criteri di successo raggiunti
└── README.md       # questo documento
```

Dipendenze aggiuntive (in `requirements.txt`): `gymnasium`, `stable-baselines3`, `tensorboard`.

Le dipendenze vanno in un solo verso: `rl → satsim`. L'engine non importa nulla da `rl/`.

**Training senza GUI.** `rl/` non importa mai `gui/`, PySide6, pyqtgraph o OpenGL: la simulazione grafica parte solo con `python main.py`. Durante l'addestramento gira soltanto l'engine fisico. Il test `tests/test_rl_env.py::test_training_is_headless` verifica che importare `rl.train` non carichi nessun modulo grafico.

## 3. Formulazione come MDP

| Elemento | Scelta | Stato |
|---|---|---|
| **Passo di controllo** | Un passo dell'engine, Δt = `params.dt` (0.05 s). Azione mantenuta costante sul passo (zero-order hold). | deciso |
| **Azione** | `Box([-1, 1]^N)`: **variazione** della coppia motore di ogni ruota. `τ ← clip(τ + action · Δτ_max, ±T_max)` con `Δτ_max = max_torque_rate · Δt` (parametro fisico in `satsim/config.py`, = 0.2 · T_max). I limiti fisici (coppia e saturazione in velocità) restano applicati dall'engine. | v3 |
| **Osservazione** | `Box(6 + 2N)`: `[e_log (3) , ω / OMEGA_SCALE (3) , τ / T_max (N) , Ω / Ω_max (N)]`, cioè errore d'assetto in scala logaritmica, velocità angolare, coppia motore corrente e velocità delle ruote. `OMEGA_SCALE = 0.1 rad/s`. Con `wheel_speed_obs=False` si ottiene il formato `Box(6 + N)` dei modelli fino alla v8. | v9 |
| **Stato iniziale** | Assetto casuale a 40–80° dal target (asse casuale), `ω` uniforme in ±0.05 rad/s, ruote ferme, coppia nulla. Campionato con `self.np_random`. | v3 |
| **Terminazione** | Nessuna: `terminated` è sempre `False`. | v1 |
| **Troncamento** | `max_episode_steps` (default 2000 passi = 100 s). | deciso |

### Osservazione

- **Posizione = `e_log`** (v5), errore d'assetto in scala logaritmica:

  $$
  \mathbf e_{log} = \hat{\mathbf n}\;\frac{\ln(1+\theta/\theta_0)}{\ln(1+\pi/\theta_0)},\qquad \theta_0 = 0.1°
  $$

  Qui $\hat{\mathbf n}$ è l'asse e $\theta = 2\,\mathrm{atan2}(|\mathbf q_{vec}|, q_0)$ l'angolo del quaternione d'errore `q_err` (con $q_0 \ge 0$, rotazione più breve). Vale 0 sul target e ha modulo in [0, 1].

  | θ | 0.01° | 0.1° | 0.45° | 1° | 10° | 60° | 180° |
  |---|---|---|---|---|---|---|---|
  | \|e_log\| | 0.013 | 0.092 | 0.23 | 0.32 | 0.62 | 0.85 | 1 |
  | \|q_vec\| (v1–v4) | 0.0001 | 0.0009 | 0.0039 | 0.0087 | 0.087 | 0.5 | 1 |

  Motivo: con `q_err` un errore di 0.45° entrava come ~0.003 e la rete non lo distingueva da zero (vedi registro v4). Con la scala logaritmica lo stesso errore vale 0.23, senza saturare alle grandi rotazioni. Limite: vicino a 180° l'asse cambia bruscamente (fuori dalla distribuzione degli stati iniziali, ≤ 80°).
- **Velocità = `ω`** in body, divisa per `OMEGA_SCALE` per portarla a valori dell'ordine di 1 (la rete neurale di PPO lavora meglio con ingressi normalizzati).
- **Coppia corrente = `τ / T_max`**, in [−1, 1]. Serve perché l'azione è una variazione: senza conoscere la coppia attuale l'agente non saprebbe che coppia sta applicando (stato non Markoviano).

### Azione: variazione di coppia (rate limit)

L'azione è la **variazione** della coppia motore di ogni ruota in un passo, non la coppia stessa. La coppia è quindi uno stato interno dell'ambiente (`self._tau`), azzerato a ogni `reset`:

$$
\tau_t = \mathrm{clip}\left(\tau_{t-1} + a_t\,\Delta\tau_{max},\ -T_{max},\ T_{max}\right), \qquad \Delta\tau_{max} = \texttt{DTAU\_MAX\_FRAC}\cdot T_{max}
$$

- Con `max_torque_rate = 8 mN·m/s` e Δt = 0.05 s, Δτ_max = 0.4 mN·m = 0.2 · T_max: la coppia va da 0 a T_max in 5 passi (0.25 s) e da −T_max a +T_max in 10 passi (0.5 s). Dal 2026-10-01 lo stesso limite è applicato dall'engine a qualunque controllore, compreso il PD.
- Motivo: con la coppia assoluta l'agente poteva commutare da +T_max a −T_max in un solo passo (*chattering*), con vibrazioni, picchi di corrente e usura dei motori. Ora la variazione è limitata per costruzione.
- Il comando `τ` è quello richiesto all'engine. L'engine può applicarne meno se la ruota è vicina alla saturazione in velocità.
- `action = 0` significa "mantieni la coppia attuale", non "coppia nulla".

### Ruote non osservate

- **Velocità delle ruote non osservate** (scelta iniziale voluta). Conseguenza: lo stato non è completamente osservabile. Il momento delle ruote `h_rw` entra nella dinamica (accoppiamento giroscopico) e determina la saturazione. Con ruote che partono ferme ed episodi da 100 s l'effetto è piccolo (con il PD le ruote non superano ~900 RPM su 6000). Diventa rilevante con disturbi forti o episodi lunghi: in quel caso si aggiunge `Ω/Ω_max` all'osservazione.

## 4. Metodi dell'ambiente

`SatAttitudeEnv` in `adcs_env.py` segue l'API di Gymnasium, richiesta da Stable-Baselines3.

| Metodo | Compito |
|---|---|
| `__init__(params, max_episode_steps, render_mode, reward_config)` | Crea `SatelliteEngine`, definisce `action_space` e `observation_space`, legge i pesi della reward (`REWARD_CONFIG`, sovrascrivibili) e il limite di variazione della coppia. |
| `reset(seed, options)` | Campiona le condizioni iniziali, resetta l'engine, azzera la coppia, salva `ω` iniziale come "passo precedente". Ritorna `(obs, info)`. |
| `step(action)` | Aggiorna la coppia `τ ← clip(τ + action · Δτ_max)`, chiama `engine.step(τ)`, calcola l'accelerazione `α = (ω − ω_prev)/Δt`, la reward, aggiorna i valori precedenti. Ritorna `(obs, reward, terminated, truncated, info)`. |
| `_compute_reward(tel, action)` | Reward del passo (§5). |
| `_get_obs(tel)` | Vettore d'osservazione (§3). |
| `_info(tel)` | Diagnostica per ogni passo: `att_err_deg`, `alpha_deg`, `tau`, `power`, `wheel_saturation`. |

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
| $\alpha_{max}$ | soglia di accelerazione per asse | `alpha_max_deg = 2.0` °/s² (confermato) |
| $k_{acc}$ | penalità per °/s² oltre soglia | `k_accel = 0.01` |
| $\Omega_j/\Omega_{max}$ | velocità della ruota j, frazione di 6000 RPM | — |
| $k_{null}$ | peso della penalità sulla deriva nello spazio nullo (v11) | `k_null = 0.1` |

Dalla v11 la reward ha un terzo termine, $-\,k_{null}\,\dfrac{|(I - A^+A)\,\boldsymbol\Omega|}{\sqrt N\,\Omega_{max}}$.
- $(I - A^+A)$ proietta le velocità delle ruote sullo spazio nullo della piramide: la combinazione $(-,+,-,+)/2$, che non produce momento sul corpo.
- Diviso per $\sqrt N$ il termine vale la velocità di deriva di ciascuna ruota in frazione di $\Omega_{max}$ (tutte le componenti del vettore nullo hanno modulo 1/2): 0.5 = 3000 RPM di deriva su ogni ruota, penalità 0.05 per passo.
- La velocità utile alla manovra (spazio immagine) non è penalizzata. Il PD, che alloca con la pseudo-inversa, ha deriva nulla per costruzione.

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

**Verifica v3 (episodio con seed 1, 2000 passi, azione = variazione di coppia).**

| Policy | Return | Errore finale | \|α\| max |
|---|---|---|---|
| Azioni nulle (coppia sempre 0, satellite libero) | −2539 | 121° | 0.03 °/s² |
| Azioni casuali | −2496 | 95° | 46 °/s² |
| PD, passato attraverso lo stesso rate limit | −76 | 0.003° | 8.1 °/s² |

Nella v2 (coppia assoluta) il PD otteneva −75: il rate limit non peggiora in modo apprezzabile il controllo.

### Storia: v1 (reward differenziale)

La v1 usava $r_t = k_{prog}(\theta_{t-1}-\theta_t)$ al posto del termine d'errore. È stata abbandonata per due motivi:
- **Somma telescopica.** Sull'episodio vale $k_{prog}(\theta_0-\theta_T)$: conta solo dove il satellite finisce, non quanto velocemente ci arriva.
- **Nessun incentivo a restare sul target.** Una volta raggiunto il target il segnale è ~0, quindi oscillarci attorno non costa nulla.

Verifica v1 (seed 1): azioni nulle −43.4, PD +76.5.

### Limiti noti

1. **Scala dei due termini.** Con una policy casuale la penalità d'accelerazione è dello stesso ordine del termine d'errore (~0.1–0.3 per passo). Se l'agente resta bloccato nel minimo locale "azioni piccole", si riduce `k_accel`.
2. **Osservazione parziale** (ruote non osservate), vedi §3.

## 6. Algoritmo: PPO

Si usa **PPO** (*Proximal Policy Optimization*) di Stable-Baselines3, con policy `MlpPolicy`. PPO è on-policy, supporta azioni continue (`Box`) e funziona con ambienti vettorizzati.

Iperparametri iniziali (`PPO_CONFIG` in `train.py`, default di SB3, da tarare):

| Parametro | Valore |
|---|---|
| `learning_rate` | 3e-4 → 0 lineare (`linear_schedule`, dalla v6) |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `device` | `cpu` (con una rete piccola la CPU è più veloce della GPU) |

## 7. Addestramento

```bash
pip install -r requirements.txt
python -m rl.train                             # 2 M passi, 4 ambienti in sequenza
python -m rl.train --subproc                   # 4 ambienti in 4 processi (più veloce)
python -m rl.train --timesteps 24576           # prova breve: misura la velocità del proprio PC
python -m rl.train --subproc --resume models/ppo_adcs --timesteps 1000000 --seed 1 --out models/ppo_adcs_2M
                                               # continua un training già fatto per altri 1 M passi
tensorboard --logdir runs/ppo_adcs             # curve di apprendimento
```

`train()` esegue, in ordine:
1. `check_env` di SB3 per verificare la conformità all'API Gymnasium;
2. `make_vec_env` con `n_envs = 4` ambienti (`DummyVecEnv`, oppure `SubprocVecEnv` con `--subproc`);
3. `PPO.learn` per `total_timesteps = 1 000 000`;
4. stampa della durata e salvataggio del modello in `models/ppo_adcs.zip` (o nel percorso `--out`).

### Validazione e modello migliore (dalla v6)

- Ogni `--eval-every` passi (default 100 k) la policy deterministica viene valutata sui seed di **validazione** 200–209.
- Se la reward media è la migliore vista finora, il modello è salvato in `<out>_best.zip`. A fine training ci sono quindi due modelli: il finale (`<out>.zip`) e il migliore (`<out>_best.zip`).
- I seed di **test** 100–109 (`rl/evaluate.py`) non vengono mai usati per scegliere il modello: servono solo al confronto finale con il PD.
- Costo: ~25 s per valutazione, ~8 min su 2 M passi. I valori vanno anche su TensorBoard (`eval/mean_reward`, `eval/final_err_deg`).

### Continuare un addestramento (`--resume`)

- `--resume <modello>` carica pesi della rete e stato dell'ottimizzatore di un modello salvato e continua per altri `--timesteps` passi. Gli iperparametri sono quelli salvati nel modello.
- Il contatore dei passi prosegue (es. da 1 007 616) e TensorBoard continua la stessa curva.
- Conviene usare un `--seed` diverso dal training precedente: con lo stesso seed l'ambiente ripeterebbe la stessa sequenza di condizioni iniziali.
- `--out` evita di sovrascrivere il modello di partenza.

Le cartelle `runs/` e `models/` sono escluse da git.

### Ambienti paralleli (`n_envs = 4`)

- **Cosa sono.** 4 copie indipendenti del simulatore, cioè 4 satelliti con condizioni iniziali diverse. **C'è un solo agente**, cioè una sola rete neurale: a ogni passo calcola in un colpo le 4 azioni, una per satellite, e ogni copia avanza di Δt.
- **Rollout.** Ogni aggiornamento di PPO usa `n_steps × n_envs = 2048 × 4 = 8192` transizioni. Quando un episodio finisce in una copia, quella si resetta da sola e le altre continuano.
- **Conteggio dei passi.** `total_timesteps` conta la somma dei passi di tutte le copie: 1 M passi totali = 250 k per copia.
- **Esecuzione.** Con `DummyVecEnv` (default di `make_vec_env`) le 4 copie girano **in sequenza nello stesso processo**, quindi non c'è vero parallelismo di CPU. Il vantaggio è statistico: i dati di ogni aggiornamento vengono da 4 episodi diversi e sono meno correlati. Per usare più core si usa `--subproc` (`SubprocVecEnv`).

Throughput misurato nel cloud: ~460 passi/s con 4 ambienti, cioè ~1 M passi in ~35 min.

**Prova breve (40 000 passi).** La pipeline funziona end-to-end. `ep_rew_mean` passa da −378 a −366: sono troppo pochi passi (~20 episodi per ambiente) per imparare la manovra. Serve un addestramento lungo.

### Tempi di addestramento

Misure nel container cloud (Intel Xeon 2.8 GHz, 4 vCPU, PyTorch su CPU):

| Configurazione | Passi/s | 1 M passi |
|---|---|---|
| Solo fisica, 1 ambiente, senza PPO | 809 | 21 min |
| PPO, 4 ambienti in sequenza (`DummyVecEnv`) | 411 | ~41 min |
| PPO, 4 ambienti in 4 processi (`--subproc`) | 591 | ~28 min |

Circa metà del tempo è la fisica: ~1.2 ms per passo, con 5 valutazioni delle derivate (4 stadi RK4 + 1 per la telemetria). L'altra metà è PPO: inferenza della rete e 10 epoche di aggiornamento ogni 8192 passi. Per stimare il tempo su un altro PC basta lanciare `python -m rl.train --timesteps 24576` e moltiplicare la durata stampata per ~40.

## 8. Decisioni aperte

- Valore di `max_torque_rate` (§3).
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

### 2026-09-28 — v3: variazione di coppia e training separato
- Azione = variazione di coppia per ruota, limitata a `0.2 · T_max` per passo. La coppia corrente `τ / T_max` è aggiunta all'osservazione (7 + N valori).
- Soglia d'accelerazione confermata a 2 °/s².
- Training spostato in `rl/train.py` (headless, CLI con `--timesteps`, `--n-envs`, `--subproc`, `--seed`). `rl/adcs_env.py` contiene solo l'ambiente.
- Nuovi test `tests/test_rl_env.py`: training senza moduli GUI, `check_env`, rate limit della coppia, reward nulla sul target.
- Misurati i tempi di addestramento (§7).

### 2026-09-28 — Primo addestramento (1 M passi) e ripresa del training
- Primo training nel cloud: 1 M passi, `--subproc`, 19 min. `ep_rew_mean` da −2720 a −72.
  - Fino a ~200 k passi resta piatta, tra 250 k e 480 k sale rapidamente, poi affina lentamente e a 1 M non è ancora piatta.
- Confronto deterministico su 10 episodi (seed 100–109):

  | | PPO (1 M) | PD |
  |---|---|---|
  | Reward per episodio | −61.7 | −53.5 |
  | Tempo per scendere sotto 1° | 15.1 s | 12.2 s |
  | Errore finale | 0.64° (identico in tutti gli episodi) | 0.003° |
  | \|α\| max | 10.1 °/s² | 6.2 °/s² |
  | Energia | 69.7 J | 61.3 J |

- Aggiunte a `rl/train.py` le opzioni `--resume` e `--out` per continuare un training esistente.
- Decisione: prima di introdurre un bonus vicino al target, si continua l'addestramento per trovare il vero plateau.

### 2026-09-28 — Ripresa: da 1 M a 2 M passi
- Ripreso da 1 M con `--resume`, seed 1, 18 min.
- `ep_rew_mean` da −72 a −65: sale ancora, ma più lentamente. La deviazione standard della policy è scesa da 0.26 a 0.15.
- Confronto deterministico su 10 episodi (seed 100–109):

  | | PPO 1 M | PPO 2 M | PD |
  |---|---|---|---|
  | Reward per episodio | −61.7 | −56.2 | −53.5 |
  | Errore finale | 0.64° | 0.40° | 0.003° |
  | Tempo per scendere sotto 1° | 15.1 s | 15.0 s | 12.2 s |
  | \|α\| max | 10.1 °/s² | 8.7 °/s² | 6.2 °/s² |
  | Energia | 69.7 J | 68.9 J | 61.3 J |

- L'errore finale resta un offset costante, identico in tutti gli episodi: la policy converge a un punto fisso a ~0.4° dal target.

### 2026-09-28 — v4: bonus vicino al target (1 M passi da zero)
- Reward: `+0.02` per ogni passo con θ < 0.1°. Nuovo `rl/evaluate.py` per il confronto sui seed 100–109.
- Baseline v4:

  | | Reward | Errore finale |
  |---|---|---|
  | PD | −19.5 | 0.0031° |
  | Satellite libero | −2300 | 131° |
  | Azioni casuali | −2759 | 130° |

- Training di 18 min. La curva `ep_rew_mean` è praticamente identica a quella senza bonus: da −2720 a −64.5.
- Valutazione PPO: reward −56.9, errore finale 0.448° (identico su tutti i seed), 16.2 s per scendere sotto 1°, \|α\| max 10.8 °/s², 65.0 J. Criteri di successo non raggiunti.
- Diagnosi: il bonus non viene quasi mai raccolto, quindi non guida l'apprendimento.
  - Dopo 30 s la policy stocastica sta sotto 0.1° solo nello 0.06 % dei passi.
  - Al punto fisso l'osservazione vale q_vec ≈ (−0.0027, 0.0013, 0.0025) e l'azione deterministica è esattamente 0: la rete non reagisce a un errore di 0.45°.
  - Conferma il limite "segnale in ingresso troppo piccolo" (§3). Il prossimo passo è riscalare l'errore d'assetto nell'osservazione.
- Nota: al punto fisso resta una coppia residua (+,−,+,−)·0.0067·T_max. È nello spazio nullo della piramide, quindi non agisce sul corpo, ma consuma energia.

### 2026-09-28 — v5: errore d'assetto in scala logaritmica
- `q_err` (4 valori) sostituito da `e_log` (3 valori, §3): l'osservazione passa da 11 a 10 valori. Reward invariata rispetto alla v4, quindi la baseline v4 resta valida.
- Training v5 (1 M passi da zero, 18 min).
  - `ep_rew_mean` da −2590 a −73.5. Parte più lento della v4 (piatta fino a ~350 k passi) e a 1 M passi sale ancora ripidamente: −108 → −73.5 negli ultimi 80 k.
  - La deviazione standard della policy è ancora 0.49 (v4: 0.26).
- Valutazione (seed 100–109):

  | | PPO v5 | PD |
  |---|---|---|
  | Reward | −29.8 | −19.5 |
  | Errore finale | **0.0042°** (v4: 0.448°) | 0.0031° |
  | Tempo per scendere sotto 1° | **12.0 s** | 12.2 s |
  | Tempo per scendere sotto 0.1° | **13.7 s** | 15.1 s |
  | \|α\| max | 10.4 °/s² | 6.2 °/s² |
  | Energia | 71.6 J | 61.3 J |

  - `[--]` reward ≥ PD; `[OK]` errore finale ≤ 0.01° su tutti i seed.
- Scomposizione della reward (media sui seed):

  | | Termine d'errore | Termine d'accelerazione | Bonus |
  |---|---|---|---|
  | PPO | −62.1 | −2.1 | +34.5 |
  | PD | −52.6 | −0.9 | +34.0 |

  Il distacco viene quasi tutto dal termine d'errore, cioè dall'area sotto θ(t) durante la manovra grande. Sui seed 101, 103 e 107 l'agente arriva sotto 0.1° solo dopo 17–21 s.
- Conclusione: la scala logaritmica ha eliminato l'offset. Resta da migliorare la manovra grande su alcuni seed.

### 2026-09-28 — v5: ripresa da 1 M a 2 M passi
- `ep_rew_mean`: sale da −72.7 a −41.7 (1.42 M), poi ricade a −66.4 (1.61 M) e risale a −47.3 (2.02 M). Andamento instabile. La deviazione standard della policy scende da 0.50 a 0.26.
- Valutazione deterministica (seed 100–109): **peggiore** del modello a 1 M passi.

  | | v5 1 M | v5 2 M | PD |
  |---|---|---|---|
  | Reward | −29.8 | −34.4 | −19.5 |
  | Errore finale | 0.0042° | 0.018° | 0.0031° |
  | Tempo per scendere sotto 1° | 12.0 s | 12.8 s | 12.2 s |

- Reward per seed, PPO 2 M vs PD:

  | Seed | 100 | 101 | 102 | 103 | 104 | 105 | 106 | 107 | 108 | 109 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | PPO 2 M | −2.9 | −58.5 | −26.1 | −82.7 | −4.6 | −30.6 | 4.8 | −110.5 | −11.3 | −21.4 |
  | PD | −3.0 | −25.2 | −16.5 | −45.8 | −7.3 | −24.6 | −5.5 | −37.2 | −14.9 | −14.8 |

  PPO batte il PD su 4 seed (100, 104, 106, 108) e perde nettamente su 101, 103 e 107.
- Seed difficili:
  - 103 e 107 hanno l'angolo iniziale più grande (74°, 69°) e ω0 che allontana dal target (+2.4 °/s lungo l'asse d'errore);
  - 101 ha θ0 = 64°.
  - Sono difficili anche per il PD, ma lì il distacco di PPO è massimo: la coda della distribuzione iniziale (θ0 vicino a 80°) è imparata peggio.
- Problemi emersi:
  1. Senza checkpoint il modello migliore del training (intorno a 1.42 M) è andato perso.
  2. `ep_rew_mean` (policy stocastica) e valutazione deterministica non vanno nella stessa direzione.
  3. Con learning rate costante, la fase di rifinitura è instabile.

### 2026-09-28 — v6: learning rate decrescente e modello migliore
- Learning rate lineare da 3e-4 a 0. `total_timesteps` di default portato a 2 M.
- Valutazione periodica sui seed di validazione 200–209, con salvataggio del modello migliore (`--eval-every`).
- Nessuna modifica ad ambiente, osservazione o reward (baseline v4 valida).
- Opzione rimandata: campionare più spesso le condizioni iniziali difficili (θ0 vicino a 80°).
- Training v6 (2 M passi da zero, 44 min comprese le valutazioni).
  - `ep_rew_mean` da −2590 a −30, crescita regolare senza ricadute: −222 a 811 k, −82 a 1 M, −45 a 1.4 M, −30 a 2 M.
  - Validazione (seed 200–209): reward da −2506 a **−8.6**, migliorata quasi a ogni valutazione. Il modello migliore coincide praticamente con il finale (2 M).
  - Errore finale in validazione: minimo 0.0024° a 1.1 M, poi risale a 0.029°. La reward continua a migliorare perché l'agente diventa più veloce.
- Test (seed 100–109), modello migliore:

  | | PPO v6 | PD |
  |---|---|---|
  | Reward | **−17.1** | −19.5 |
  | Errore finale | 0.029° | 0.0031° |
  | Tempo per scendere sotto 1° | **9.1 s** | 12.2 s |
  | \|α\| max | 11.5 °/s² | 6.2 °/s² |
  | Energia | 68.6 J | 61.3 J |

  - `[OK]` reward ≥ PD; `[--]` errore finale ≤ 0.01°.
  - Più rapido del PD su 9 seed su 10. Reward migliore su 7 seed su 10, peggiore su 103 (−69.5 vs −45.8), 107 (−39.4 vs −37.2) e 108 (−25.0 vs −14.9).
- Osservazione: sotto 0.1° la reward è quasi indifferente all'errore. Il termine lineare a 0.03° vale −0.0003 per passo e il bonus è già preso. Il criterio "≤ 0.01°" non è quindi rappresentato nella reward: l'agente ha scambiato precisione residua per velocità.

### 2026-09-29 — v7: secondo bonus di precisione
- Reward: al bonus `+0.02` per θ < 0.1° si somma un secondo bonus `+0.02` per θ < 0.01°. Sotto 0.01° il premio per passo vale quindi 0.04. Parametri `bonus2`, `bonus2_theta_deg` in `REWARD_CONFIG`.
- Motivo: nella v6 la reward era quasi indifferente all'errore sotto 0.1°, quindi il criterio "errore finale ≤ 0.01°" non veniva premiato.
- Baseline v7 (seed 100–109):

  | | Reward | Errore finale |
  |---|---|---|
  | PD | **+10.0** | 0.0031° |
  | Satellite libero | −2300 | 131° |
  | Azioni casuali | −2759 | 130° |

  Il PD sta sotto 0.01° per gran parte dell'episodio e guadagna +29.5 rispetto alla v6.
- Training: 2 M passi da zero, stesse impostazioni della v6.
- Training v7 (2 M passi, 46 min).
  - `ep_rew_mean` quasi identica alla v6: da −2590 a −31. La reward d'addestramento (policy stocastica) raccoglie poco il bonus di precisione.
  - Validazione: reward da −2506 a **+24.7** (1.8 M passi, modello migliore). Errore finale 0.003–0.009° da 1.1 M in poi.
  - All'ultima valutazione (2 M) la validazione crolla a −8.6 (errore 0.0109°): il modello finale è peggiore del migliore.
- Test (seed 100–109):

  | | PPO v7 migliore | PPO v7 finale | PD |
  |---|---|---|---|
  | Reward | **+13.4** | −17.3 | +10.0 |
  | Errore finale medio | 0.0131° | 0.0107° | 0.0031° |
  | Tempo per scendere sotto 1° | **9.2 s** | 9.3 s | 12.2 s |
  | \|α\| max | 10.9 °/s² | 10.7 °/s² | 6.2 °/s² |
  | Energia | 71.9 J | 69.2 J | 61.3 J |

  - Modello migliore: `[OK]` reward ≥ PD; `[--]` errore finale ≤ 0.01° (8 seed su 10).
  - Errore finale per seed: 0.008° su 8 seed, 0.0145° sul seed 104, 0.0526° sul seed 109.
  - Reward migliore del PD su 7 seed su 10, peggiore su 103 (−45.0 vs −16.7), 107 (−11.7 vs −7.8) e 108 (6.8 vs 14.6).
- Osservazione: l'agente si ferma appena dentro l'ultima soglia di bonus. Nella v6 si fermava a 0.03° con soglia 0.1°, nella v7 a 0.008° con soglia 0.01°. Sotto l'ultima soglia la reward non premia altra precisione, quindi l'errore residuo è determinato dalla posizione della soglia.

### 2026-09-29 — v8: penalità logaritmica sull'errore
- Reward: alla penalità lineare si aggiunge `−0.02 · ln(1 + θ/0.01°)` (`k_log`, `log_theta0_deg`). Bonus e penalità sulle accelerazioni sono invariati.
  - La parte lineare domina agli angoli grandi e mantiene la spinta a fare in fretta la manovra.
  - La parte logaritmica domina sotto ~1° e premia ogni miglioramento di precisione anche sotto le soglie dei bonus: da 0.01° a 0.001° vale 0.012 per passo, contro 0.00009 della sola lineare.
- Motivo: nelle v6 e v7 l'agente si fermava appena dentro l'ultima soglia di bonus (0.03° con soglia 0.1°, 0.008° con soglia 0.01°).
- Penalità d'errore totale per passo:

  | θ | 60° | 10° | 1° | 0.1° | 0.01° | 0.001° |
  |---|---|---|---|---|---|---|
  | Penalità | 0.77 | 0.24 | 0.10 | 0.049 | 0.014 | 0.002 |

- Baseline v8 (seed 100–109):

  | | Reward | Errore finale |
  |---|---|---|
  | PD | **−42.9** | 0.0031° |
  | Satellite libero | −2670 | 131° |
  | Azioni casuali | −3135 | 130° |
- Training v8 (2 M passi, 46 min).
  - `ep_rew_mean` da −2970 a −84.
  - Validazione: reward da −2970 a **−11.2** (1.7 M passi, modello migliore). Errore finale 0.0003–0.002° da 1.2 M passi in poi.
- Test (seed 100–109):

  | | PPO v8 migliore | PPO v8 finale | PD |
  |---|---|---|---|
  | Reward | **−27.8** | −48.5 | −42.9 |
  | Errore finale | **0.0003°** (tutti i seed) | 0.14° (max 1.39°) | 0.0031° |
  | Tempo per scendere sotto 1° | **10.8 s** | 10.6 s | 12.2 s |
  | \|α\| max | 14.1 °/s² | 14.2 °/s² | 6.2 °/s² |
  | Energia | 69.1 J | 70.6 J | 61.3 J |

  - Modello migliore: **`[OK]` reward ≥ PD; `[OK]` errore finale ≤ 0.01° su tutti i seed.** Entrambi i criteri di successo sono soddisfatti.
  - Reward migliore del PD su 8 seed su 10. Sul seed 101 è peggiore (−56.4 vs −49.2), sul 103 è praticamente pari (−74.9 vs −74.4).
  - L'errore finale è 10 volte più piccolo di quello del PD.
- Il modello finale (2 M) è molto peggiore del migliore: su un seed si ferma a 1.39°. Conferma che il salvataggio del modello migliore è indispensabile.
- Restano peggiori del PD accelerazioni (14.1 vs 6.2 °/s²) ed energia (69.1 vs 61.3 J).
- Il modello migliore v8 è pubblicato in `rl/pretrained/ppo_adcs_v8.zip` (156 KB), come eccezione alla regola di non versionare i modelli (`models/` resta in `.gitignore`). Per verificarlo: `python -m rl.evaluate rl/pretrained/ppo_adcs_v8`. È valido solo con il codice v8: osservazione 10 valori, azione come variazione di coppia.

### 2026-10-01 — Limite di coppia nell'engine e modalità confronto
- Il limite sulla variazione della coppia passa dall'ambiente all'engine (`max_torque_rate` in `satsim/config.py`): stesso valore (0.2 · T_max per passo), ora applicato anche al PD della GUI. `DTAU_MAX_FRAC` rimosso. Risultati del modello v8 invariati (`rl.evaluate`: −27.8 vs −42.9).
- `SatAttitudeEnv` espone `last_tel` (telemetria dell'ultimo passo) e `T_external` (coppia esterna di prova, 0 nel training).
- `rl/compare.py` + `python main.py --compare`: confronto visivo PPO vs PD. Coincide con `rl.evaluate` (test).
- **Scoperta dal confronto visivo: le ruote dell'agente girano nello spazio nullo.**
  - Sui seed 100–109, a fine episodio il 91–100 % della velocità delle ruote di PPO sta nello spazio nullo della piramide: combinazioni (+,−,+,−) che non producono coppia sul corpo. Per il PD la quota è 0–9 %, perché l'allocazione con pseudo-inversa è a norma minima.
  - Picco di velocità delle ruote: PPO 1941–6000 RPM, PD 658–2377 RPM. Sui seed 101 (a 7.1 s) e 107 (a 11.3 s) PPO porta una ruota a **saturazione**: proprio due dei seed in cui va peggio del PD. Una ruota saturata perde capacità di controllo e produce picchi d'accelerazione.
  - Causa probabile: l'agente non osserva le velocità delle ruote e la reward non le penalizza, quindi la componente nello spazio nullo, che non ha effetti sull'assetto, deriva liberamente. È anche la causa principale del maggior consumo d'energia rispetto al PD.

### 2026-10-02 — Modelli portabili tra versioni di Python
- **Bug:** caricare `ppo_adcs_v8` con Python 3.14, sul PC locale, causava un *segmentation fault* sia in `pytest` sia in `main.py --compare`.
  - Il learning rate era una funzione Python (closure `linear_schedule`). Stable-Baselines3 la salvava nel `.zip` come bytecode di Python 3.11, la versione del cloud.
  - Caricata con Python 3.14, quel bytecode faceva andare in crash l'interprete. Lo stack del crash indicava `rl/train.py, in schedule`.
- **Correzione:**
  - `rl/models.py`: `LR_SCHEDULE = LinearSchedule(3e-4 → 0)` di Stable-Baselines3. È una classe, serializzata per riferimento, quindi portabile. Stessi valori della funzione precedente.
  - `load_model()`: carica un modello sostituendo il learning rate salvato con `LR_SCHEDULE`, senza mai deserializzarlo. Usato da `evaluate.py`, `compare.py` e `train.py --resume`.
  - `rl/pretrained/ppo_adcs_v8.zip` risalvato in formato portabile: stessi pesi, risultati identici (−27.8 vs −42.9).
- Test `test_saved_models_are_portable`: nessuna funzione serializzata nei modelli pubblicati né in un modello appena salvato.
- Regola: caricare i modelli sempre con `rl.models.load_model`, mai con `PPO.load` diretto.

### 2026-10-02 — v9: velocità delle ruote nell'osservazione
- **Motivo:** l'agente v8 accumula il 91–100 % della velocità delle ruote nello spazio nullo, fino alla saturazione sui seed 101 e 107, senza poterlo vedere (registro del 2026-10-01).
- **Modifica:** in `_get_obs` si aggiunge `tel.wheel_speed / Ω_max`, cioè le velocità delle 4 ruote prese dalla telemetria dell'ultimo passo. Il dato fisico è in rad/s; diviso per Ω_max = 628 rad/s (6000 RPM) vale tra −1 e +1, dove ±1 significa saturazione.
  - L'osservazione passa da 10 a 14 valori.
  - Senza normalizzazione un ingresso fino a ±628, contro gli altri intorno a ±1, dominerebbe la rete.
- **Reward invariata:** una modifica alla volta. Baseline PD invariata (−42.9).
- **Compatibilità con i modelli fino alla v8:**
  - opzione `SatAttitudeEnv(wheel_speed_obs=False)`;
  - `rl.models.env_kwargs_for(model)` sceglie il formato in base al numero di ingressi del modello;
  - usata da `rl.evaluate`, dal confronto (`--compare`) e da `rl.train --resume`. Il modello v8 continua a funzionare ovunque, con risultati invariati.
- Test: `test_wheel_speed_in_observation`, `test_old_models_still_load` (18 test in totale).
- **Aspettativa:** osservare le ruote dà all'agente l'informazione, ma non un incentivo a tenerle lente, perché la reward non le penalizza. Il passo successivo previsto è una penalità sulla velocità delle ruote.
- **Risultati:** training da fare (2 M passi, in locale).

### 2026-10-03 — v10: penalità sulla velocità delle ruote (da addestrare e testare)
- **Risultato v9 (osservazione a 14 valori, reward invariata):** errore finale 0.0008° e reward −34.5, peggio del v8 (−27.8, errore 0.0003°) e del PD solo sull'energia/accelerazioni. Osservare le ruote senza un incentivo a tenerle lente non basta, come previsto.
- **Misura (seed 100–109, picco di |Ω|/Ω_max):** PD 0.11–0.40 (media nel tempo 0.04); v8 0.32–1.00 (media nel tempo 0.52, saturazione sui seed 101 e 107).
- **Modifica:** reward −`k_wheel · Σ_j max(0, |Ω_j|/Ω_max − 0.3)`, con `k_wheel = 0.05`.
  - Soglia 0.3 (1800 RPM): il PD la supera solo su 2 seed su 10, di poco, quindi la manovra "normale" non è penalizzata e la precisione dell'agente non è toccata direttamente.
  - Hinge lineare, come per l'accelerazione: la penalità interviene solo sulla deriva nello spazio nullo e sull'avvicinamento alla saturazione. A Ω = Ω_max su una ruota costa 0.035 per passo, confrontabile con la penalità d'errore a ~1° (0.10).
- **Baseline PD con la nuova reward:** −43.0 (era −42.9). Il modello v8, che non osserva le ruote, scende a −127.4: conferma che la penalità è attiva sul suo comportamento.
- Test: `test_wheel_speed_penalty` (19 test, tutti passati). Nessun cambio di osservazione: il formato a 14 valori è quello della v9.
- **Da fare:** training da zero 2 M passi (`python -m rl.train --subproc`), poi `rl.evaluate` sul modello `_best`. Se l'errore finale peggiora, ridurre `k_wheel` (0.02) o alzare la soglia; se le ruote superano ancora ~0.5, alzare `k_wheel`.

### 2026-10-03 — `rl.evaluate`: picco e spazio nullo delle ruote
- **Risultato v10 (reward `k_wheel = 0.05`, soglia 0.3; training locale `local_training_v3`):** reward −67.9 vs PD −43.0, errore finale 0.0011°, t<1° 14.4 s (PD 12.2), |α| max 15.5, energia 67.8 J. La manovra è più lenta e la reward peggiora rispetto a v8/v9: da capire se la penalità limita la manovra o se è variabilità del training.
- **Modifica (solo metrica, nessun effetto sul training):** `rl.evaluate` stampa due colonne in più, il picco di |Ω| in RPM (media/max sui seed) e la quota di energia cinetica delle ruote nello spazio nullo a fine episodio, `1 − |A⁺AΩ|²/|Ω|²`. Sotto la tabella c'è il dettaglio per seed per PPO e PD.
- Verifica sul v8: PPO picco 1941–6000 RPM e 83–100 % nello spazio nullo; PD 659–2378 RPM e 0–1 %. Coincide con i valori del registro del 2026-10-01.
- Test: `test_null_space_share` (20 test).

### 2026-10-03 — v11: penalità sulla deriva nello spazio nullo (sostituisce la v10)
- **Analisi del v10 con le nuove metriche** (`local_training_v3`): picco medio 3116 RPM (max 5333), quota nello spazio nullo 71 % (v8: 3879 RPM, 97 %; PD: 1250 RPM, 0 %). 9 seed su 10 sopra la soglia di 1800 RPM.
  - La penalità sulla velocità totale ha ridotto la deriva solo in parte e ha reso la manovra più lenta (14.4 s).
  - Mescola due cose diverse: la velocità utile alla manovra (spazio immagine di A) e la deriva inutile (spazio nullo). La seconda è quella da eliminare, e si può eliminare senza toccare l'assetto: una coppia lungo (+,−,+,−) ferma la deriva senza agire sul corpo.
- **Modifica:** il termine v10 `−k_wheel · Σ max(0, |Ω_j|/Ω_max − 0.3)` è sostituito da `−k_null · |(I − A⁺A)Ω| / (√N·Ω_max)`, con `k_null = 0.1` (§5). Nessuna soglia: il PD ha deriva nulla. `wheel_speed_thr` e `k_wheel` rimossi. Osservazione invariata (14 valori, v9).
- **Baseline con la nuova reward:** PD −43.2 (v10: −43.0). Il modello v8 scende a −131.9 (~104 punti persi per la deriva).
- Test: `test_null_space_penalty` sostituisce `test_wheel_speed_penalty` (20 test).
- **Da fare:** training da zero 2 M passi, poi `rl.evaluate`. Atteso: quota nello spazio nullo vicina a 0 % senza peggiorare t<1°. Se la manovra rallenta ancora, ridurre `k_null`; se la deriva resta sopra ~20 %, aumentarlo.
