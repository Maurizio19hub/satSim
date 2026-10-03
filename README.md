# satSim — Simulatore ADCS per CubeSat 3U con ruote di reazione

Simulatore fisico con visualizzazione 3D della dinamica d'assetto di un CubeSat 3U controllato da ruote di reazione (*Reaction Wheels*, RW). Il progetto serve come base per addestrare un agente di **Reinforcement Learning** (PPO/SAC) per il controllo d'assetto (ADCS). Per ora il satellite è stabilizzato da un **controllore PD sui quaternioni**, che fa da riferimento (*baseline*) per l'agente.

L'engine fisico è **completamente disaccoppiato** dalla GUI: il pacchetto `satsim/` non importa nulla di Qt e da lì verrà estratto l'ambiente Gymnasium.

---

## Indice

1. [Installazione e avvio](#1-installazione-e-avvio)
2. [Architettura del software](#2-architettura-del-software)
3. [Principio fisico di funzionamento](#3-principio-fisico-di-funzionamento)
4. [Sistemi di riferimento e convenzioni](#4-sistemi-di-riferimento-e-convenzioni)
5. [Equazione 1 — Matrice d'inerzia](#5-equazione-1--matrice-dinerzia-j)
6. [Equazione 2 — Dinamica delle ruote e consumo elettrico](#6-equazione-2--dinamica-delle-ruote-di-reazione-e-consumo-elettrico)
7. [Equazione 3 — Equazioni di Eulero con accoppiamento giroscopico](#7-equazione-3--equazioni-di-eulero-con-accoppiamento-giroscopico)
8. [Equazione 4 — Cinematica dei quaternioni](#8-equazione-4--cinematica-dei-quaternioni)
9. [Equazione 5 — Coppie di disturbo ambientali](#9-equazione-5--coppie-di-disturbo-ambientali)
10. [Integrazione numerica RK4](#10-integrazione-numerica-rk4)
11. [Controllore PD sui quaternioni](#11-controllore-pd-sui-quaternioni)
12. [Verifica e validazione](#12-verifica-e-validazione)
13. [Parametri di default](#13-parametri-di-default)
14. [Limiti del modello e sviluppi futuri](#14-limiti-del-modello-e-sviluppi-futuri)
15. [Registro di sviluppo](#15-registro-di-sviluppo)

---

## 1. Installazione e avvio

Richiede Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy PySide6 pyqtgraph PyOpenGL qtawesome pytest
# oppure: pip install -r requirements.txt

python main.py                         # 4 ruote in piramide, tempo reale
python main.py --wheels 3 --speed 5    # 3 ruote ortogonali, 5× tempo reale
python main.py --dt 0.02 --seed 42     # passo diverso, condizioni iniziali riproducibili
python main.py --compare --seed 101    # confronto PPO (sinistra) vs PD (destra), vedi sotto

python -m pytest -q                    # test di verifica della fisica
```

### Interfaccia

| Area | Contenuto |
|---|---|
| **Vista 3D** (sinistra) | CubeSat 3U (faccia +Z dorata, faccia +X viola). Terna **inerziale/target** fissa (`X_I, Y_I, Z_I`, colori scuri). Terna **body** solidale al satellite (`x_B, y_B, z_B`, colori vivi). Assi di rotazione delle ruote (`RW1…RW4`). Direzione del Sole (gialla) e del nadir (ciano). Il mouse ruota e zooma la camera. |
| **Comandi** | Pausa/Riprendi (anche con `Spazio`), Reset con tumbling casuale, perturbazione istantanea di 45°, impulso di coppia di 5 mN·m per 1 s, velocità di simulazione (0.25×…20×), attivazione di controllore e disturbi, **slider della coppia di disturbo manuale** su x/y/z (±5 mN·m). |
| **Grafici** (centro) | Finestra mobile di 60 s: quaternione, errore d'assetto in gradi, ω, velocità delle ruote in RPM (con i limiti ±6000 tratteggiati), potenza elettrica. |
| **Telemetria** (destra, in alto) | Valori istantanei: q, q_err, angolo d'errore, angoli RPY, ω, **saturazione delle ruote in %** (verde < 70 %, giallo < 95 %, rosso oltre), potenza istantanea e energia accumulata (J e Wh), modulo delle coppie di disturbo. |
| **Modello matematico** (destra, in basso) | Matrice J e le 5 equazioni del modello, con i valori numerici correnti di ogni termine. |

**Prove suggerite**
- Con il PD attivo, porta lo slider `T_z` a +2 mN·m: le ruote accumulano momento angolare finché saturano (barre rosse) e da quel momento il satellite perde il controllo. È il motivo per cui i satelliti reali fanno il *momentum dumping*.
- Disattiva il PD durante un tumbling: il moto diventa quello libero di Eulero-Poinsot.

### Modalità confronto PPO vs PD

```bash
python main.py --compare                 # seed 0, modello rl/pretrained/ppo_adcs_v11
python main.py --compare --seed 101 --model models/ppo_adcs_best
```

Richiede anche `gymnasium` e `stable-baselines3`; la GUI normale no. Due satelliti partono dalla **stessa condizione iniziale** (stesso seed, stessa estrazione di `ClosedLoopSimulation.reset` e dell'ambiente RL) e avanzano insieme, con gli stessi disturbi e lo stesso limite sulla coppia.

| Area | Contenuto |
|---|---|
| **Viste 3D** | Sinistra: agente PPO. Destra: PD. |
| **Comandi** | Campo seed + Avvia, seed casuale, pausa (`Spazio`), velocità (0.25×…10×), impulso di 5 mN·m per 1 s con la stessa direzione su entrambi. |
| **Confronto** | Tabella PPO / PD: errore, \|ω\|, \|α\| max, max \|Ω\| ruote, energia, reward accumulata, tempi per scendere sotto 1° e 0.01°. A 100 s l'episodio si chiude con il riepilogo. |
| **Grafici** | Curve sovrapposte (PPO blu, PD arancione): errore in scala log (soglie 1° e 0.01°), \|ω\|, max\|α_i\| (soglia 2 °/s²), max \|Ω\| ruote (limite 6000 RPM), potenza. |

I numeri coincidono con `python -m rl.evaluate` sullo stesso seed (verificato da `tests/test_rl_env.py::test_comparison_matches_evaluate`).

---

## 2. Architettura del software

```
satSim/
├── main.py                  # punto d'ingresso (CLI + avvio GUI)
├── satsim/                  # ─── ENGINE (nessuna dipendenza da Qt) ───
│   ├── config.py            # dataclass dei parametri + costanti fisiche
│   ├── quaternion.py        # algebra dei quaternioni, DCM, prodotto vettoriale veloce
│   ├── environment.py       # orbita, Sole, eclisse, campo B, T_gg / T_srp / T_mag
│   ├── physics_engine.py    # J, ruote, stato, derivate, RK4, telemetria
│   ├── controller.py        # PD sui quaternioni (da sostituire con l'agente RL)
│   └── simulation.py        # anello chiuso + coppie manuali + storico
├── gui/                     # ─── VISUALIZZAZIONE ───
│   ├── view3d.py            # scena OpenGL (pyqtgraph.opengl)
│   ├── compare_window.py    # finestra di confronto PPO vs PD (--compare)
│   ├── theme.py             # tema grafico condiviso: colori, foglio di stile Qt, icone
│   ├── dashboard.py         # grafici, telemetria numerica, pannello del modello
│   └── main_window.py       # layout, comandi, loop temporale
├── rl/                      # ─── REINFORCEMENT LEARNING (PPO, SB3) — vedi rl/README.md ───
│   ├── adcs_env.py          # ambiente Gymnasium (solo fisica, niente GUI)
│   ├── train.py             # addestramento PPO headless: python -m rl.train
│   ├── evaluate.py          # confronto numerico di un modello con il PD
│   ├── compare.py           # logica del confronto PPO vs PD passo-passo (senza Qt)
│   └── pretrained/          # modelli addestrati (ppo_adcs_v8.zip, ppo_adcs_v11.zip)
└── tests/
    ├── test_physics.py      # verifica: conservazione e soluzioni analitiche
    └── test_rl_env.py       # ambiente RL e training senza GUI
```

Le dipendenze vanno in un solo verso: `gui → satsim`, mai il contrario.

**API dell'engine (già pronta per Gymnasium)**

```python
from satsim import SatelliteEngine, SimParams

eng = SatelliteEngine(SimParams(dt=0.05))
tel = eng.reset(q0=[...], omega0=[...])       # -> Telemetry
tel = eng.step(tau_wheels, T_manual=None)     # tau_wheels: coppie motore delle N ruote [N·m]
tau = eng.allocate(T_body_desiderata)         # allocazione con pseudo-inversa
```

Nel futuro ambiente Gym l'**azione** sarà `tau_wheels` (normalizzata in [−1, 1] · T_max). L'**osservazione** sarà costruita da `tel.q_err`, `tel.omega` e `tel.wheel_speed`, e la **reward** da `tel.att_err_deg`, `tel.omega` e `tel.power_total`.

---

## 3. Principio fisico di funzionamento

Una ruota di reazione è un volano azionato da un motore elettrico fissato alla struttura del satellite. Quando il motore applica al rotore una coppia $T_{rw}$, per il **terzo principio della dinamica** il rotore applica alla struttura una coppia uguale e opposta $-T_{rw}$.

Il sistema satellite + ruote è isolato, salvo le piccole coppie ambientali. Vale quindi la **conservazione del momento angolare totale**:

$$
\mathbf{H} = \underbrace{J\,\boldsymbol\omega}_{\text{corpo}} + \underbrace{\mathbf h_{rw}}_{\text{ruote}}
\qquad\Longrightarrow\qquad
\left.\frac{d\mathbf H}{dt}\right|_{I} = \mathbf T_{est}
$$

Accelerando una ruota in un verso, il corpo ruota nel verso opposto. Il satellite *scambia* momento angolare con le ruote, senza consumare propellente. Il controllore ADCS sfrutta questo scambio per orientare il satellite.

Le coppie esterne di disturbo (gravità, Sole, magnetismo) invece **aggiungono** momento angolare al sistema. Le ruote lo assorbono finché raggiungono la velocità massima (**saturazione**). Da quel momento non possono più controllare l'asse interessato. Questa dinamica, con i suoi compromessi tra precisione, energia e margine di saturazione, è ciò che l'agente RL dovrà imparare a gestire.

Il simulatore risolve il sistema di equazioni differenziali ordinarie (ODE) non lineari

$$
\dot{\mathbf x} = f(t, \mathbf x, \mathbf u), \qquad
\mathbf x = [\,\mathbf q,\ \boldsymbol\omega,\ \boldsymbol\Omega,\ E\,]\in\mathbb R^{7+N+1}
$$

dove $\mathbf u$ sono le coppie motore delle $N$ ruote ed $E$ è l'energia elettrica consumata.

---

## 4. Sistemi di riferimento e convenzioni

| Simbolo | Significato |
|---|---|
| **I** | Riferimento inerziale (ECI, centrato nella Terra). Coincide con l'**assetto target** (quaternione identità). |
| **B** | Riferimento body, solidale al satellite e centrato nel baricentro. $z_B$ è l'asse lungo del 3U. |
| $\mathbf q=[q_0,q_1,q_2,q_3]$ | Quaternione unitario, **scalare per primo**, prodotto di **Hamilton**. Ruota i vettori da B a I: $\mathbf v_I = R(\mathbf q)\,\mathbf v_B$. |
| $\boldsymbol\omega$ | Velocità angolare di B rispetto a I, **espressa in B** [rad/s]. |
| $\Omega_i$ | Velocità di rotazione della ruota $i$ attorno al proprio asse [rad/s]. |
| $A\in\mathbb R^{3\times N}$ | Matrice di distribuzione: la colonna $i$ è l'asse di rotazione della ruota $i$ in B. |

Matrice di rotazione ricavata dal quaternione:

$$
R(\mathbf q)=\begin{bmatrix}
1-2(q_2^2+q_3^2) & 2(q_1q_2-q_0q_3) & 2(q_1q_3+q_0q_2)\\
2(q_1q_2+q_0q_3) & 1-2(q_1^2+q_3^2) & 2(q_2q_3-q_0q_1)\\
2(q_1q_3-q_0q_2) & 2(q_2q_3+q_0q_1) & 1-2(q_1^2+q_2^2)
\end{bmatrix}
$$

Tutte le grandezze sono in unità SI.

---

## 5. Equazione 1 — Matrice d'inerzia J

Il CubeSat 3U è modellato come un **parallelepipedo omogeneo** di massa $m = 3$ kg e lati $a\times b\times c = 0.1\times0.1\times0.3$ m. Il tensore d'inerzia rispetto al baricentro, negli assi di simmetria, vale:

$$
J_{xx}=\frac{m}{12}(b^2+c^2),\qquad
J_{yy}=\frac{m}{12}(a^2+c^2),\qquad
J_{zz}=\frac{m}{12}(a^2+b^2)
$$

Numericamente $J_{xx}=J_{yy}=0.025$ kg·m² e $J_{zz}=0.005$ kg·m². Il satellite è quindi un corpo **allungato** (*prolato*), con l'asse z di minima inerzia.

Un satellite reale non è omogeneo: batterie, schede e payload lo rendono asimmetrico. Per questo si aggiungono piccoli **prodotti d'inerzia** $J_{xy}, J_{xz}, J_{yz}\sim10^{-5}$ kg·m²:

$$
J=\begin{bmatrix}
0.025 & 2\cdot10^{-5} & -1\cdot10^{-5}\\
2\cdot10^{-5} & 0.025 & 1.5\cdot10^{-5}\\
-1\cdot10^{-5} & 1.5\cdot10^{-5} & 0.005
\end{bmatrix}\ \text{kg·m}^2
$$

In questo modo gli assi body non sono esattamente assi principali: nasce un accoppiamento tra gli assi che il controllore deve gestire. Il codice verifica che J sia simmetrica e definita positiva (autovalori > 0), condizione necessaria perché rappresenti un corpo fisico.

*Codice:* `physics_engine.cubesat_inertia()`.

---

## 6. Equazione 2 — Dinamica delle ruote di reazione e consumo elettrico

### 6.1 Dinamica

Ogni ruota è un rotore simmetrico con inerzia assiale $I_{rw}$. Il motore gli applica la coppia $T_{rw,i}$:

$$
\frac{d\Omega_i}{dt} = \frac{T_{rw,i}}{I_{rw}}
$$

Il momento angolare dell'array di ruote, espresso in body, è

$$
\mathbf h_{rw} = A\,I_{rw}\,\boldsymbol\Omega
$$

La coppia di reazione trasmessa al corpo è $\mathbf T_{rw,azione} = -A\,\mathbf T_{rw}$.

> **Nota sulla precisione.** $\Omega$ è trattata come la velocità assiale **assoluta** (inerziale) del rotore. Con questa scelta $I_{rw}\dot\Omega = T_{rw}$ vale in modo esatto per un rotore simmetrico, e il momento angolare totale $J\boldsymbol\omega + \mathbf h_{rw}$ si conserva esattamente (§12). La velocità relativa misurata dall'encoder sarebbe $\Omega_i - \mathbf a_i^T\boldsymbol\omega$. La differenza vale $\sim10^{-2}$ rad/s contro $\sim10^{2}$ rad/s, quindi è trascurabile per la visualizzazione.

### 6.2 Configurazioni

- `orthogonal3`: tre ruote allineate con $x_B, y_B, z_B$, quindi $A = I_3$.
- `pyramid4` (default): quattro ruote inclinate di $\beta = 54.74°$ rispetto a $z_B$ e disposte ad azimut di 45°, 135°, 225° e 315°:
  $$\mathbf a_i = [\sin\beta\cos\varphi_i,\ \sin\beta\sin\varphi_i,\ \cos\beta]^T$$
  Questa configurazione è **ridondante**: se una ruota si guasta, le altre tre controllano ancora tutti e tre gli assi.

### 6.3 Allocazione della coppia

Data la coppia desiderata sul corpo $\mathbf T_{cmd}$, le coppie motore sono

$$
\mathbf T_{rw} = -A^{+}\,\mathbf T_{cmd},\qquad A^{+}=A^T(AA^T)^{-1}
$$

$A^{+}$ è la pseudo-inversa di Moore-Penrose. Con 4 ruote fornisce la soluzione a **norma minima**, cioè la distribuzione più "economica" tra le ruote.

### 6.4 Saturazioni

1. **Coppia:** $|T_{rw,i}|\le T_{max}$ (2 mN·m).
2. **Variazione della coppia (rate limit):** $|T_{rw,i}(t+\Delta t)-T_{rw,i}(t)|\le \dot T_{max}\,\Delta t$, con $\dot T_{max}$ = `max_torque_rate` = 8 mN·m/s. Da 0 a $T_{max}$ servono 0.25 s, da $-T_{max}$ a $+T_{max}$ 0.5 s. Modella il driver del motore, che non può cambiare la corrente (∝ coppia) istantaneamente: un salto brusco di coppia causerebbe vibrazioni, picchi di corrente e usura. Il limite è applicato dall'engine (`ReactionWheelArray.rate_limit`) a qualunque controllore, PD o agente RL.
3. **Velocità:** $|\Omega_i|\le\Omega_{max}$ (6000 RPM ≈ 628 rad/s).

La coppia è costante durante il passo $\Delta t$ (vedi §10), quindi l'equazione 2 si integra in modo **esatto**: $\Omega_i(t+\Delta t)=\Omega_i+T_{rw,i}\Delta t/I_{rw}$. Prima dell'integrazione il comando viene limitato a

$$
\frac{(-\Omega_{max}-\Omega_i)\,I_{rw}}{\Delta t}\ \le\ T_{rw,i}\ \le\ \frac{(\Omega_{max}-\Omega_i)\,I_{rw}}{\Delta t}
$$

Così la ruota arriva *esattamente* a $\Omega_{max}$ senza superarla. Non serve tagliare lo stato a posteriori (*clipping*), un'operazione che violerebbe la conservazione del momento angolare. A saturazione la ruota accetta coppia solo nel verso che la rallenta.

### 6.5 Modello di potenza elettrica

$$
P_i = k_1\,|T_{rw,i}| + k_2\,|T_{rw,i}\,\Omega_i| + P_{statica}
$$

| Termine | Origine fisica |
|---|---|
| $k_1\lvert T\rvert$ | In un motore brushless la coppia è proporzionale alla corrente ($T=k_t i$). Le perdite ohmiche $R i^2$ e le perdite del driver crescono con la coppia richiesta (qui approssimate linearmente). |
| $k_2\lvert T\Omega\rvert$ | La potenza meccanica $T\Omega$ scambiata con il rotore, divisa per il rendimento ($k_2 = 1/\eta \approx 1.2$). Si usa il valore assoluto perché si assume che l'energia di frenata **non** venga recuperata. |
| $P_{statica}$ | Elettronica di controllo, encoder e cuscinetti, sempre attivi. |

L'energia $E=\int P\,dt$ è una **variabile di stato** integrata con RK4 insieme alla dinamica. Così è accurata anche quando $\Omega$ varia molto nel passo. È una grandezza chiave per la futura reward dell'agente RL.

*Codice:* `physics_engine.ReactionWheelArray`.

---

## 7. Equazione 3 — Equazioni di Eulero con accoppiamento giroscopico

Si parte dal teorema del momento angolare in un riferimento inerziale, $\dot{\mathbf H}|_I=\mathbf T_{est}$. Si porta la derivata nel riferimento rotante B con il **teorema di trasporto**, $\dot{\mathbf H}|_I=\dot{\mathbf H}|_B+\boldsymbol\omega\times\mathbf H$. Con $\mathbf H = J\boldsymbol\omega + \mathbf h_{rw}$ e $J$ costante in B si ottiene:

$$
J\dot{\boldsymbol\omega} + \dot{\mathbf h}_{rw} + \boldsymbol\omega\times(J\boldsymbol\omega+\mathbf h_{rw}) = \mathbf T_{dist}
$$

Si sostituisce $\dot{\mathbf h}_{rw}=A\,I_{rw}\dot{\boldsymbol\Omega}=A\,\mathbf T_{rw}$ e si porta la reazione delle ruote a destra:

$$
\boxed{\,J\,\dot{\boldsymbol\omega} = \mathbf T_{tot} - \boldsymbol\omega\times\left(J\boldsymbol\omega + \mathbf h_{rw}\right),\qquad \mathbf T_{tot} = -A\mathbf T_{rw} + \mathbf T_{dist}\,}
$$

Il termine $\boldsymbol\omega\times(J\boldsymbol\omega+\mathbf h_{rw})$ è l'**accoppiamento giroscopico**. Contiene due effetti:
- $\boldsymbol\omega\times J\boldsymbol\omega$: l'accoppiamento non lineare tra gli assi di un corpo rigido in rotazione. È la causa, per esempio, dell'instabilità della rotazione attorno all'asse intermedio (teorema della racchetta da tennis).
- $\boldsymbol\omega\times\mathbf h_{rw}$: la **rigidezza giroscopica** delle ruote in rotazione. Se il corpo ruota mentre le ruote hanno momento angolare, nasce una coppia su un asse perpendicolare.

Nel codice $\dot{\boldsymbol\omega}$ si ottiene con $J^{-1}$, precalcolata una sola volta perché J è costante.

*Codice:* `SatelliteEngine._derivatives()`.

---

## 8. Equazione 4 — Cinematica dei quaternioni

I quaternioni rappresentano l'assetto **senza singolarità**, a differenza degli angoli di Eulero che soffrono di *gimbal lock*. Si usano 4 parametri, legati dal vincolo $\lVert\mathbf q\rVert=1$. La cinematica, con $\boldsymbol\omega$ espressa in body, è:

$$
\dot{\mathbf q} = \frac12\,\mathbf q\otimes\begin{bmatrix}0\\ \boldsymbol\omega\end{bmatrix}
= \frac12\begin{bmatrix}
-q_1\omega_x - q_2\omega_y - q_3\omega_z\\
\ \ q_0\omega_x - q_3\omega_y + q_2\omega_z\\
\ \ q_3\omega_x + q_0\omega_y - q_1\omega_z\\
-q_2\omega_x + q_1\omega_y + q_0\omega_z
\end{bmatrix}
$$

Il prodotto di Hamilton $\mathbf p\otimes\mathbf q = [p_0q_0-\mathbf p\cdot\mathbf q,\ p_0\mathbf q+q_0\mathbf p+\mathbf p\times\mathbf q]$ è implementato in `quaternion.quat_mult()`.

**Normalizzazione.** L'equazione esatta conserva $\lVert\mathbf q\rVert$, ma l'integratore numerico no: l'errore di troncamento RK4 fa derivare la norma. Dopo ogni passo si proietta quindi il quaternione sulla sfera unitaria $S^3$ con $\mathbf q\leftarrow\mathbf q/\lVert\mathbf q\rVert$.

**Errore d'assetto.** Rispetto a un target $\mathbf q_t$:

$$
\mathbf q_e = \mathbf q_t^{*}\otimes\mathbf q,\qquad \theta_{err} = 2\arccos|q_{e,0}|
$$

Il segno viene scelto in modo che $q_{e,0}\ge0$. I quaternioni coprono SO(3) due volte ($\mathbf q$ e $-\mathbf q$ sono lo stesso assetto), e questa scelta garantisce che il controllore ruoti sempre lungo il cammino più breve (≤ 180°), evitando il fenomeno di *unwinding*.

---

## 9. Equazione 5 — Coppie di disturbo ambientali

$$
\mathbf T_{tot} = \mathbf T_{rw,azione} + \mathbf T_{gg} + \mathbf T_{srp} + \mathbf T_{mag}\ (+\ \mathbf T_{manuale})
$$

I disturbi dipendono dalla posizione in orbita, quindi serve un modello d'orbita. Si usa un'**orbita circolare kepleriana** a 500 km di quota e 51.6° di inclinazione, con periodo di circa 94.5 minuti:

$$
\mathbf r_I(t) = R_3(-\Omega_{RAAN})\,R_1(-i)\ r\,[\cos u,\ \sin u,\ 0]^T,\qquad u = u_0 + n t,\quad n=\sqrt{\mu/r^3}
$$

Tutti i disturbi vengono **ricalcolati a ogni stadio RK4**, perché dipendono dall'assetto $\mathbf q$ e dal tempo $t$.

### 9.1 Gradiente di gravità

La gravità non è uniforme sul corpo: la parte più vicina alla Terra è attratta un po' di più. Espandendo al primo ordine il potenziale $-\mu/|\mathbf r+\boldsymbol\rho|$ sulla distribuzione di massa si ottiene:

$$
\mathbf T_{gg} = \frac{3\mu}{r^3}\ \hat{\mathbf r}_B\times\left(J\,\hat{\mathbf r}_B\right),\qquad \hat{\mathbf r}_B = R(\mathbf q)^T\,\frac{\mathbf r_I}{r}
$$

La coppia si annulla quando la direzione radiale coincide con un asse principale d'inerzia (verificato nei test). Tende ad allineare l'asse di **minima** inerzia (qui $z_B$) con la verticale locale, principio sfruttato per la stabilizzazione passiva a gradiente di gravità. Ordine di grandezza: $10^{-9}$–$10^{-8}$ N·m.

### 9.2 Pressione di radiazione solare (SRP)

I fotoni solari trasportano quantità di moto. Il satellite è modellato con **6 facce piane**. Per ogni faccia illuminata ($\cos\theta=\hat{\mathbf n}\cdot\hat{\mathbf s}>0$):

$$
\mathbf F = -P_\odot\,A\cos\theta\left[(1-\rho_s)\,\hat{\mathbf s} + 2\left(\rho_s\cos\theta + \tfrac{\rho_d}{3}\right)\hat{\mathbf n}\right],\qquad
\mathbf T_{srp} = \sum_{facce}\left(\mathbf r_{cp}-\mathbf r_{cm}\right)\times\mathbf F
$$

- $P_\odot = 4.56\cdot10^{-6}$ N/m² è la pressione a 1 UA.
- $\rho_s$ e $\rho_d$ sono le frazioni di riflessione speculare e diffusa. La parte restante viene assorbita.
- $\hat{\mathbf s}$ è la direzione del Sole in body.

La coppia nasce dall'**offset tra centro di pressione e centro di massa** $\mathbf r_{cm}$ (qui [2, −1, 10] mm). Con il baricentro al centro geometrico le coppie delle facce opposte si annullerebbero per simmetria.

**Eclisse.** Si usa un modello d'ombra cilindrico: il satellite è in ombra se $\mathbf r\cdot\hat{\mathbf s}<0$ e la sua distanza dall'asse Terra-Sole è minore di $R_\oplus$. In eclisse $\mathbf T_{srp}=0$ (visibile in GUI). Ordine di grandezza: $10^{-9}$ N·m.

### 9.3 Coppia magnetica residua

Correnti nei circuiti e materiali magnetizzati danno al satellite un dipolo magnetico residuo $\mathbf m$ (qui ~0.01 A·m²). Questo dipolo interagisce con il campo terrestre, modellato come **dipolo** allineato con l'asse polare:

$$
\mathbf B_I(\mathbf r) = B_0\left(\frac{R_\oplus}{r}\right)^3\left[3(\hat{\mathbf m}_\oplus\cdot\hat{\mathbf r})\hat{\mathbf r}-\hat{\mathbf m}_\oplus\right],\qquad
\mathbf T_{mag} = \mathbf m\times\left(R(\mathbf q)^T\mathbf B_I\right)
$$

con $B_0 = 3.12\cdot10^{-5}$ T. A 500 km $|\mathbf B|\approx 25$–$50\ \mu$T e $|\mathbf T_{mag}|\sim10^{-7}$ N·m. È tipicamente il **disturbo dominante** per un CubeSat in LEO.

### 9.4 Coppia manuale

Una coppia costante in body, impostata dagli slider della GUI (±5 mN·m), oppure un impulso di 1 s. Serve per testare visivamente la risposta del controllore e la saturazione delle ruote. È volutamente molto più grande dei disturbi naturali.

*Codice:* `environment.OrbitalEnvironment`.

---

## 10. Integrazione numerica RK4

Il sistema $\dot{\mathbf x}=f(t,\mathbf x,\mathbf u)$ è integrato con **Runge-Kutta del 4° ordine a passo fisso** ($\Delta t = 0.05$ s):

$$
\begin{aligned}
\mathbf k_1 &= f(t,\ \mathbf x_n,\ \mathbf u_n)\\
\mathbf k_2 &= f(t+\tfrac{\Delta t}{2},\ \mathbf x_n+\tfrac{\Delta t}{2}\mathbf k_1,\ \mathbf u_n)\\
\mathbf k_3 &= f(t+\tfrac{\Delta t}{2},\ \mathbf x_n+\tfrac{\Delta t}{2}\mathbf k_2,\ \mathbf u_n)\\
\mathbf k_4 &= f(t+\Delta t,\ \mathbf x_n+\Delta t\,\mathbf k_3,\ \mathbf u_n)\\
\mathbf x_{n+1} &= \mathbf x_n + \tfrac{\Delta t}{6}(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4)
\end{aligned}
$$

Scelte progettuali:
- **Zero-order hold** sul comando $\mathbf u_n$: la coppia delle ruote resta costante per tutto il passo. È ciò che accade in un computer di bordo, che aggiorna il comando a frequenza fissa (qui 20 Hz), ed è la semantica naturale di `env.step(action)` in Gymnasium.
- **Passo fisso:** rende la simulazione deterministica e riproducibile, e l'errore locale è $O(\Delta t^5)$. Le costanti di tempo del sistema, cioè il periodo del controllore (~15 s) e il periodo orbitale, sono di vari ordini di grandezza maggiori di $\Delta t$.
- Dopo il passo si **normalizza** il quaternione (§8).

Prestazioni: circa 1.5 ms per passo in Python puro con NumPy. Il prodotto vettoriale è implementato a mano (`cross3`) perché `np.cross` ha un overhead che dominava il tempo di calcolo su vettori così piccoli.

---

## 11. Controllore PD sui quaternioni

Un controllore provvisorio in attesa dell'agente RL:

$$
\mathbf T_{cmd} = -J\left(K_p\,\mathbf q_{e,vec} + K_d\,\boldsymbol\omega\right) + \boldsymbol\omega\times(J\boldsymbol\omega+\mathbf h_{rw})
$$

- Il primo termine è un PD **normalizzato con l'inerzia**. Per piccoli angoli $\mathbf q_{e,vec}\approx\boldsymbol\theta/2$ e ogni asse diventa un oscillatore del secondo ordine $\ddot\theta+K_d\dot\theta+\tfrac{K_p}{2}\theta=0$. Si sceglie quindi $K_p=2\omega_n^2$ e $K_d=2\zeta\omega_n$ (default $\omega_n=0.4$ rad/s, $\zeta=0.9$).
- Il secondo termine **compensa l'accoppiamento giroscopico**, rendendo la dinamica ad anello chiuso quasi lineare.
- $\mathbf T_{cmd}$ viene poi allocato sulle ruote (§6.3) e limitato dalle saturazioni (§6.4), compreso il limite sulla variazione della coppia: il PD non può più far saltare la coppia da un passo all'altro.

**Stabilità.** Senza disturbi e senza saturazioni, la compensazione giroscopica cancella il termine non lineare e resta $\dot{\boldsymbol\omega} = -K_p\mathbf q_{e,vec} - K_d\boldsymbol\omega$. Si usa come funzione di Lyapunov

$$
V = 2K_p\,(1-q_{e,0}) + \tfrac12\,\boldsymbol\omega^T\boldsymbol\omega \;\ge 0
$$

Poiché $\dot q_{e,0} = -\tfrac12\mathbf q_{e,vec}^T\boldsymbol\omega$, la derivata vale

$$
\dot V = K_p\,\mathbf q_{e,vec}^T\boldsymbol\omega + \boldsymbol\omega^T(-K_p\mathbf q_{e,vec} - K_d\boldsymbol\omega) = -K_d\,\lVert\boldsymbol\omega\rVert^2 \le 0
$$

Per il principio di invarianza di LaSalle il sistema converge a $\boldsymbol\omega=0,\ \mathbf q_{e,vec}=0$. Non essendoci un termine integrale, con disturbi costanti rimane un piccolo errore a regime (~0.003°).

---

## 12. Verifica e validazione

`tests/test_physics.py` contiene 9 test (`python -m pytest -q`):

| Test | Proprietà fisica verificata |
|---|---|
| `test_inertia_matrix_3U` | J corrisponde alla formula analitica ed è simmetrica. |
| `test_torque_free_conservation` | Moto libero (tumbling di 100 s): $\mathbf H_I$ costante (errore < 1e-9) ed energia cinetica costante (errore relativo < 1e-7). $\lVert\mathbf q\rVert=1$. |
| `test_internal_torques_conserve_total_momentum` | Coppie casuali sulle ruote: il momento angolare **totale** inerziale si conserva, perché le ruote scambiano solo momento interno. |
| `test_wheel_saturation_is_exact_and_conservative` | Le ruote a coppia massima arrivano esattamente a 6000 RPM senza superarli, e il momento totale resta conservato. |
| `test_power_and_energy_idle` | A coppia nulla $P = N\cdot P_{statica}$ e $E = N P_{statica}\,t$. |
| `test_quaternion_kinematics_analytic` | Rotazione uniforme attorno a un asse principale: coincide con la soluzione analitica $\mathbf q(t)=[\cos\tfrac{\omega t}{2},0,0,\sin\tfrac{\omega t}{2}]$. |
| `test_gravity_gradient_zero_on_principal_axis` | $\mathbf T_{gg}=0$ quando il nadir è allineato con un asse principale. |
| `test_pd_controller_converges` | Da 90° e in presenza di disturbi: errore < 0.1° dopo 80 s. |
| `test_torque_rate_limit` | Con un comando a gradino (+T_max poi −T_max) la coppia applicata varia al massimo di $\dot T_{max}\Delta t$ per passo; il reset azzera la coppia. |

La conservazione di $\mathbf H$ è il test più importante. Verifica insieme la coerenza dei segni tra l'equazione 2 (ruote), l'equazione 3 (Eulero) e l'allocazione. Un errore di segno sulla reazione $-A\mathbf T_{rw}$ farebbe crescere H in modo sistematico.

---

## 13. Parametri di default

Sono tutti in `satsim/config.py` (dataclass modificabili).

| Parametro | Valore | Note |
|---|---|---|
| Massa, dimensioni | 3 kg, 10×10×30 cm | CubeSat 3U |
| Offset del baricentro | [2, −1, 10] mm | braccio della coppia SRP |
| $I_{rw}$ | 1.5·10⁻⁵ kg·m² | ruota di classe CubeSat |
| $\Omega_{max}$ | 6000 RPM | $h_{max}\approx 9.4$ mN·m·s per ruota |
| $T_{max}$ | 2 mN·m | |
| $k_1, k_2, P_s$ | 50 W/(N·m), 1.2, 0.15 W | modello di potenza |
| Orbita | 500 km, i = 51.6° | circolare |
| Dipolo residuo | [5, −3, 10] mA·m² | |
| $\rho_s, \rho_d$ | 0.1, 0.3 | |
| $\Delta t$ | 0.05 s | RK4 |
| PD | $\omega_n=0.4$ rad/s, $\zeta=0.9$ | |

---

## 14. Limiti del modello e sviluppi futuri

**Semplificazioni attuali**
- Orbita circolare kepleriana, senza J2 né resistenza atmosferica.
- Il Sole ha direzione inerziale fissa: su poche ore il moto apparente è di ~0.04°/h.
- Il campo magnetico è un dipolo allineato con l'asse polare, non il modello IGRF.
- La coppia aerodinamica non è modellata. A 500 km è dello stesso ordine di SRP e gradiente di gravità.
- J è rigida e costante, senza flessibilità né sloshing. L'inerzia trasversale delle ruote è inclusa in J.
- Le ruote sono ideali, senza attrito viscoso o di Coulomb, ritardi del motore o rumore degli encoder.
- Sensori perfetti: il controllore conosce lo stato vero, senza modelli di giroscopi o star tracker né filtro di Kalman.

**Roadmap**
1. `satsim/gym_env.py`: ambiente `gymnasium.Env` con `action_space = Box(-1, 1, (N,))` e osservazione [q_err, ω, Ω/Ω_max]. La reward penalizzerà errore d'assetto, velocità angolare, energia e vicinanza alla saturazione. Randomizzazione delle condizioni iniziali e dei parametri (*domain randomization*).
2. Addestramento PPO/SAC con Stable-Baselines3 e confronto quantitativo con il PD, su tempo di assestamento, energia ed errore a regime.
3. Attrito delle ruote, rumore dei sensori, desaturazione con magnetorquer.
4. Integrazione della policy addestrata nella GUI, selezionabile al posto del PD.

---

## 15. Registro di sviluppo

### v0.1.0 — 2026-09-23 — Prima versione
- Engine fisico disaccoppiato (`satsim/`) con stato $[\mathbf q,\boldsymbol\omega,\boldsymbol\Omega,E]$ e integratore RK4 a passo fisso (Δt = 0.05 s) con zero-order hold sul comando.
- Implementate le 5 equazioni: J del 3U con prodotti d'inerzia; dinamica delle RW con saturazione esatta in velocità e coppia, e modello di potenza $k_1|T|+k_2|T\Omega|+P_s$; Eulero con accoppiamento giroscopico; cinematica dei quaternioni con normalizzazione; disturbi di gradiente di gravità, SRP (6 facce + eclisse cilindrica) e dipolo magnetico.
- Due configurazioni di ruote: 3 ortogonali e 4 in piramide, con allocazione tramite pseudo-inversa.
- Controllore PD sui quaternioni con compensazione giroscopica.
- GUI in PySide6 + pyqtgraph.opengl: vista 3D (satellite, terne inerziale e body, assi delle ruote, Sole, nadir), grafici in tempo reale, telemetria con barre di saturazione, pannello del modello matematico con i valori live, coppie di disturbo manuali, pausa e velocità variabile.
- 8 test di verifica fisica: conservazione di H e dell'energia, soluzioni analitiche, saturazione, convergenza del PD.
- Ottimizzazione: prodotto vettoriale scritto a mano (da ~4 a ~1.5 ms per passo).

### 2026-10-01 — Limite sulla variazione della coppia e modalità confronto
- Nuovo parametro fisico `ReactionWheelParams.max_torque_rate` (8 mN·m/s): l'engine limita la variazione della coppia di ogni ruota a 0.4 mN·m per passo, per qualunque controllore. Il PD della GUI ora rispetta lo stesso limite dell'agente RL. Effetto sul PD (seed 42): picco di \|α\| da 6.93 a 6.75 °/s²; tempo per scendere sotto 1° ed errore finale invariati.
- `python main.py --compare`: confronto visivo PPO vs PD sullo stesso seed (`gui/compare_window.py`, logica in `rl/compare.py`).
- Test: `test_torque_rate_limit` (fisica), `test_comparison_matches_evaluate` (RL).

### 2026-10-01 — Restyling dell'interfaccia
- Nuovo `gui/theme.py`, condiviso dalle due finestre: tema scuro con pannelli a "card", colore d'accento, font di sistema, foglio di stile Qt (QSS) per pulsanti, campi, checkbox, slider e barre; stile uniforme dei grafici pyqtgraph.
- Icone vettoriali Material Design con **QtAwesome** al posto delle emoji. La dipendenza è opzionale: senza QtAwesome i pulsanti restano senza icona.
- Modalità confronto ridisegnata:
  - intestazione con seed, condizioni iniziali, avanzamento dell'episodio e stato;
  - barra comandi orizzontale, con la velocità come selettore a pulsanti;
  - card delle metriche con il valore migliore evidenziato in verde, al posto della tabella di testo.
