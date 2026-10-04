# satSim — ADCS simulator for a 3U CubeSat with reaction wheels

Physics simulator with 3D visualisation of the attitude dynamics of a 3U CubeSat controlled by reaction wheels (RW). The project is the basis for training a **Reinforcement Learning** agent (PPO/SAC) for attitude control (ADCS). The satellite can also be stabilised by a **quaternion PD controller**, which serves as the *baseline* for the agent.

The physics engine is **fully decoupled** from the GUI: the `satsim/` package imports nothing from Qt, and the Gymnasium environment is built on top of it.

---

## Contents

1. [Installation and launch](#1-installation-and-launch)
2. [Software architecture](#2-software-architecture)
3. [Physical principle](#3-physical-principle)
4. [Reference frames and conventions](#4-reference-frames-and-conventions)
5. [Equation 1 — Inertia matrix](#5-equation-1--inertia-matrix-j)
6. [Equation 2 — Wheel dynamics and power consumption](#6-equation-2--reaction-wheel-dynamics-and-power-consumption)
7. [Equation 3 — Euler equations with gyroscopic coupling](#7-equation-3--euler-equations-with-gyroscopic-coupling)
8. [Equation 4 — Quaternion kinematics](#8-equation-4--quaternion-kinematics)
9. [Equation 5 — Environmental disturbance torques](#9-equation-5--environmental-disturbance-torques)
10. [RK4 numerical integration](#10-rk4-numerical-integration)
11. [Quaternion PD controller](#11-quaternion-pd-controller)
12. [Verification and validation](#12-verification-and-validation)
13. [Default parameters](#13-default-parameters)
14. [Model limitations and future work](#14-model-limitations-and-future-work)
15. [Development log](#15-development-log)

---

## 1. Installation and launch

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy PySide6 pyqtgraph PyOpenGL qtawesome pytest
# or: pip install -r requirements.txt

python main.py                         # 4 wheels in a pyramid, real time
python main.py --wheels 3 --speed 5    # 3 orthogonal wheels, 5× real time
python main.py --dt 0.02 --seed 42     # different step, reproducible initial conditions
python main.py --compare --seed 101    # PPO (left) vs PD (right) comparison, see below

python -m pytest -q                    # physics verification tests
```

### Interface

| Area | Content |
|---|---|
| **3D view** (left) | 3U CubeSat (gold +Z face, purple +X face). Fixed **inertial/target** frame (`X_I, Y_I, Z_I`, dark colours). **Body** frame attached to the satellite (`x_B, y_B, z_B`, bright colours). Wheel spin axes (`RW1…RW4`). Sun direction (yellow) and nadir (cyan). The mouse rotates and zooms the camera. |
| **Controls** | Pause/Resume (also with `Space`), Reset with random tumbling, instant 45° perturbation, 5 mN·m torque impulse for 1 s, simulation speed (0.25×…20×), enabling of controller and disturbances, **manual disturbance torque sliders** on x/y/z (±5 mN·m). |
| **Plots** (centre) | 60 s moving window: quaternion, attitude error in degrees, ω, wheel speeds in RPM (with the ±6000 limits dashed), electrical power. |
| **Telemetry** (right, top) | Instantaneous values: q, q_err, error angle, RPY angles, ω, **wheel saturation in %** (green < 70 %, yellow < 95 %, red above), instantaneous power and accumulated energy (J and Wh), magnitude of the disturbance torques. |
| **Mathematical model** (right, bottom) | Matrix J and the 5 model equations, with the current numerical value of every term. |

**Suggested experiments**
- With the PD active, move the `T_z` slider to +2 mN·m: the wheels accumulate angular momentum until they saturate (red bars), and from that moment the satellite loses control. This is why real satellites perform *momentum dumping*.
- Disable the PD during a tumble: the motion becomes free Euler-Poinsot motion.

### PPO vs PD comparison mode

```bash
python main.py --compare                 # seed 0, model rl/pretrained/local_training_v4_seed1@3_best
python main.py --compare --seed 101 --model models/ppo_adcs_best
```

Also requires `gymnasium` and `stable-baselines3`; the normal GUI does not. Two satellites start from the **same initial condition** (same seed, same draw in `ClosedLoopSimulation.reset` and in the RL environment) and advance together, with the same disturbances and the same torque limit.

| Area | Content |
|---|---|
| **3D views** | Left: PPO agent. Right: PD. |
| **Controls** | Seed field + Start, random seed, pause (`Space`), speed (0.5×…10×), 5 mN·m impulse for 1 s with the same direction on both. |
| **Comparison** | PPO / PD cards: error, \|ω\|, max \|α\|, max wheel \|Ω\|, energy, cumulative reward, times to get below 1° and 0.01°. At 100 s the episode ends with a summary. |
| **Plots** | Overlaid curves (PPO blue, PD orange): error on a log scale (1° and 0.01° thresholds), \|ω\|, max\|α_i\| (2 °/s² threshold), max wheel \|Ω\| (6000 RPM limit), power. |

The numbers match `python -m rl.evaluate` on the same seed (checked by `tests/test_rl_env.py::test_comparison_matches_evaluate`).

---

## 2. Software architecture

```
satSim/
├── main.py                  # entry point (CLI + GUI start)
├── satsim/                  # ─── ENGINE (no Qt dependency) ───
│   ├── config.py            # parameter dataclasses + physical constants
│   ├── quaternion.py        # quaternion algebra, DCM, fast cross product
│   ├── environment.py       # orbit, Sun, eclipse, B field, T_gg / T_srp / T_mag
│   ├── physics_engine.py    # J, wheels, state, derivatives, RK4, telemetry
│   ├── controller.py        # quaternion PD (baseline for the RL agent)
│   └── simulation.py        # closed loop + manual torques + history
├── gui/                     # ─── VISUALISATION ───
│   ├── view3d.py            # OpenGL scene (pyqtgraph.opengl)
│   ├── compare_window.py    # PPO vs PD comparison window (--compare)
│   ├── theme.py             # shared graphical theme: colours, Qt style sheet, icons
│   ├── dashboard.py         # plots, numerical telemetry, model panel
│   └── main_window.py       # layout, controls, time loop
├── rl/                      # ─── REINFORCEMENT LEARNING (PPO, SB3) — see rl/README.md ───
│   ├── adcs_env.py          # Gymnasium environment (physics only, no GUI)
│   ├── train.py             # headless PPO training: python -m rl.train
│   ├── evaluate.py          # numerical comparison of a model with the PD
│   ├── compare.py           # step-by-step PPO vs PD comparison logic (no Qt)
│   └── pretrained/          # trained models (ppo_adcs_v8.zip, local_training_v4_seed1@3_best.zip)
└── tests/
    ├── test_physics.py      # verification: conservation and analytical solutions
    └── test_rl_env.py       # RL environment and GUI-free training
```

Dependencies go one way only: `gui → satsim`, never the other way round.

**Engine API (used by the Gymnasium environment)**

```python
from satsim import SatelliteEngine, SimParams

eng = SatelliteEngine(SimParams(dt=0.05))
tel = eng.reset(q0=[...], omega0=[...])       # -> Telemetry
tel = eng.step(tau_wheels, T_manual=None)     # tau_wheels: motor torques of the N wheels [N·m]
tau = eng.allocate(T_body_desired)            # allocation with the pseudo-inverse
```

The RL environment, its observation, action and reward are described in [rl/README.md](rl/README.md).

---

## 3. Physical principle

A reaction wheel is a flywheel driven by an electric motor fixed to the satellite structure. When the motor applies a torque $T_{rw}$ to the rotor, by **Newton's third law** the rotor applies an equal and opposite torque $-T_{rw}$ to the structure.

The satellite + wheels system is isolated, apart from the small environmental torques. The **total angular momentum is therefore conserved**:

$$
\mathbf{H} = \underbrace{J\,\boldsymbol\omega}_{\text{body}} + \underbrace{\mathbf h_{rw}}_{\text{wheels}}
\qquad\Longrightarrow\qquad
\left.\frac{d\mathbf H}{dt}\right|_{I} = \mathbf T_{ext}
$$

Spinning a wheel up in one direction makes the body rotate in the opposite direction. The satellite *exchanges* angular momentum with the wheels, without using propellant. The ADCS controller exploits this exchange to point the satellite.

External disturbance torques (gravity, Sun, magnetism), on the other hand, **add** angular momentum to the system. The wheels absorb it until they reach their maximum speed (**saturation**). From then on they can no longer control the affected axis. This dynamics, with its trade-offs between precision, energy and saturation margin, is what the RL agent has to learn to manage.

The simulator solves the system of nonlinear ordinary differential equations (ODEs)

$$
\dot{\mathbf x} = f(t, \mathbf x, \mathbf u), \qquad
\mathbf x = [\,\mathbf q,\ \boldsymbol\omega,\ \boldsymbol\Omega,\ E\,]\in\mathbb R^{7+N+1}
$$

where $\mathbf u$ are the motor torques of the $N$ wheels and $E$ is the electrical energy consumed.

---

## 4. Reference frames and conventions

| Symbol | Meaning |
|---|---|
| **I** | Inertial frame (ECI, Earth-centred). It coincides with the **target attitude** (identity quaternion). |
| **B** | Body frame, attached to the satellite and centred at the centre of mass. $z_B$ is the long axis of the 3U. |
| $\mathbf q=[q_0,q_1,q_2,q_3]$ | Unit quaternion, **scalar first**, **Hamilton** product. It rotates vectors from B to I: $\mathbf v_I = R(\mathbf q)\,\mathbf v_B$. |
| $\boldsymbol\omega$ | Angular velocity of B relative to I, **expressed in B** [rad/s]. |
| $\Omega_i$ | Spin rate of wheel $i$ about its own axis [rad/s]. |
| $A\in\mathbb R^{3\times N}$ | Distribution matrix: column $i$ is the spin axis of wheel $i$ in B. |

Rotation matrix obtained from the quaternion:

$$
R(\mathbf q)=\begin{bmatrix}
1-2(q_2^2+q_3^2) & 2(q_1q_2-q_0q_3) & 2(q_1q_3+q_0q_2)\\
2(q_1q_2+q_0q_3) & 1-2(q_1^2+q_3^2) & 2(q_2q_3-q_0q_1)\\
2(q_1q_3-q_0q_2) & 2(q_2q_3+q_0q_1) & 1-2(q_1^2+q_2^2)
\end{bmatrix}
$$

All quantities are in SI units.

---

## 5. Equation 1 — Inertia matrix J

The 3U CubeSat is modelled as a **homogeneous rectangular box** of mass $m = 3$ kg and sides $a\times b\times c = 0.1\times0.1\times0.3$ m. Its inertia tensor about the centre of mass, in the symmetry axes, is:

$$
J_{xx}=\frac{m}{12}(b^2+c^2),\qquad
J_{yy}=\frac{m}{12}(a^2+c^2),\qquad
J_{zz}=\frac{m}{12}(a^2+b^2)
$$

Numerically $J_{xx}=J_{yy}=0.025$ kg·m² and $J_{zz}=0.005$ kg·m². The satellite is therefore an **elongated** (*prolate*) body, with z as the axis of minimum inertia.

A real satellite is not homogeneous: batteries, boards and payload make it asymmetric. Small **products of inertia** $J_{xy}, J_{xz}, J_{yz}\sim10^{-5}$ kg·m² are therefore added:

$$
J=\begin{bmatrix}
0.025 & 2\cdot10^{-5} & -1\cdot10^{-5}\\
2\cdot10^{-5} & 0.025 & 1.5\cdot10^{-5}\\
-1\cdot10^{-5} & 1.5\cdot10^{-5} & 0.005
\end{bmatrix}\ \text{kg·m}^2
$$

As a result the body axes are not exactly principal axes: a coupling between the axes appears, which the controller has to handle. The code checks that J is symmetric and positive definite (eigenvalues > 0), a necessary condition for it to represent a physical body.

*Code:* `physics_engine.cubesat_inertia()`.

---

## 6. Equation 2 — Reaction wheel dynamics and power consumption

### 6.1 Dynamics

Each wheel is a symmetric rotor with axial inertia $I_{rw}$. The motor applies the torque $T_{rw,i}$ to it:

$$
\frac{d\Omega_i}{dt} = \frac{T_{rw,i}}{I_{rw}}
$$

The angular momentum of the wheel array, expressed in body axes, is

$$
\mathbf h_{rw} = A\,I_{rw}\,\boldsymbol\Omega
$$

The reaction torque transmitted to the body is $\mathbf T_{rw,action} = -A\,\mathbf T_{rw}$.

> **Note on accuracy.** $\Omega$ is treated as the **absolute** (inertial) axial speed of the rotor. With this choice $I_{rw}\dot\Omega = T_{rw}$ holds exactly for a symmetric rotor, and the total angular momentum $J\boldsymbol\omega + \mathbf h_{rw}$ is conserved exactly (§12). The relative speed measured by the encoder would be $\Omega_i - \mathbf a_i^T\boldsymbol\omega$. The difference is $\sim10^{-2}$ rad/s against $\sim10^{2}$ rad/s, so it is negligible for visualisation.

### 6.2 Configurations

- `orthogonal3`: three wheels aligned with $x_B, y_B, z_B$, so $A = I_3$.
- `pyramid4` (default): four wheels tilted by $\beta = 54.74°$ from $z_B$ and placed at azimuths of 45°, 135°, 225° and 315°:
  $$\mathbf a_i = [\sin\beta\cos\varphi_i,\ \sin\beta\sin\varphi_i,\ \cos\beta]^T$$
  This configuration is **redundant**: if one wheel fails, the other three still control all three axes.

### 6.3 Torque allocation

Given the desired body torque $\mathbf T_{cmd}$, the motor torques are

$$
\mathbf T_{rw} = -A^{+}\,\mathbf T_{cmd},\qquad A^{+}=A^T(AA^T)^{-1}
$$

$A^{+}$ is the Moore-Penrose pseudo-inverse. With 4 wheels it gives the **minimum-norm** solution, i.e. the most "economical" distribution among the wheels.

### 6.4 Saturations

1. **Torque:** $|T_{rw,i}|\le T_{max}$ (2 mN·m).
2. **Torque rate (rate limit):** $|T_{rw,i}(t+\Delta t)-T_{rw,i}(t)|\le \dot T_{max}\,\Delta t$, with $\dot T_{max}$ = `max_torque_rate` = 8 mN·m/s. Going from 0 to $T_{max}$ takes 0.25 s, from $-T_{max}$ to $+T_{max}$ 0.5 s. It models the motor driver, which cannot change the current (∝ torque) instantaneously: an abrupt torque jump would cause vibrations, current peaks and wear. The limit is applied by the engine (`ReactionWheelArray.rate_limit`) to any controller, PD or RL agent.
3. **Speed:** $|\Omega_i|\le\Omega_{max}$ (6000 RPM ≈ 628 rad/s).

The torque is constant during the step $\Delta t$ (see §10), so equation 2 is integrated **exactly**: $\Omega_i(t+\Delta t)=\Omega_i+T_{rw,i}\Delta t/I_{rw}$. Before integration the command is limited to

$$
\frac{(-\Omega_{max}-\Omega_i)\,I_{rw}}{\Delta t}\ \le\ T_{rw,i}\ \le\ \frac{(\Omega_{max}-\Omega_i)\,I_{rw}}{\Delta t}
$$

This way the wheel reaches *exactly* $\Omega_{max}$ without exceeding it. There is no need to cut the state afterwards (*clipping*), an operation that would violate the conservation of angular momentum. At saturation the wheel accepts torque only in the direction that slows it down.

### 6.5 Electrical power model

$$
P_i = k_1\,|T_{rw,i}| + k_2\,|T_{rw,i}\,\Omega_i| + P_{static}
$$

| Term | Physical origin |
|---|---|
| $k_1\lvert T\rvert$ | In a brushless motor the torque is proportional to the current ($T=k_t i$). Ohmic losses $R i^2$ and driver losses grow with the requested torque (approximated linearly here). |
| $k_2\lvert T\Omega\rvert$ | The mechanical power $T\Omega$ exchanged with the rotor, divided by the efficiency ($k_2 = 1/\eta \approx 1.2$). The absolute value is used because braking energy is assumed **not** to be recovered. |
| $P_{static}$ | Control electronics, encoder and bearings, always on. |

The energy $E=\int P\,dt$ is a **state variable** integrated with RK4 together with the dynamics, so it stays accurate even when $\Omega$ changes a lot within a step. It is one of the key quantities in the comparison between the RL agent and the PD.

*Code:* `physics_engine.ReactionWheelArray`.

---

## 7. Equation 3 — Euler equations with gyroscopic coupling

Start from the angular momentum theorem in an inertial frame, $\dot{\mathbf H}|_I=\mathbf T_{ext}$. Move the derivative to the rotating frame B with the **transport theorem**, $\dot{\mathbf H}|_I=\dot{\mathbf H}|_B+\boldsymbol\omega\times\mathbf H$. With $\mathbf H = J\boldsymbol\omega + \mathbf h_{rw}$ and $J$ constant in B this gives:

$$
J\dot{\boldsymbol\omega} + \dot{\mathbf h}_{rw} + \boldsymbol\omega\times(J\boldsymbol\omega+\mathbf h_{rw}) = \mathbf T_{dist}
$$

Substitute $\dot{\mathbf h}_{rw}=A\,I_{rw}\dot{\boldsymbol\Omega}=A\,\mathbf T_{rw}$ and move the wheel reaction to the right-hand side:

$$
\boxed{\,J\,\dot{\boldsymbol\omega} = \mathbf T_{tot} - \boldsymbol\omega\times\left(J\boldsymbol\omega + \mathbf h_{rw}\right),\qquad \mathbf T_{tot} = -A\mathbf T_{rw} + \mathbf T_{dist}\,}
$$

The term $\boldsymbol\omega\times(J\boldsymbol\omega+\mathbf h_{rw})$ is the **gyroscopic coupling**. It contains two effects:
- $\boldsymbol\omega\times J\boldsymbol\omega$: the nonlinear coupling between the axes of a rotating rigid body. It causes, for example, the instability of rotation about the intermediate axis (tennis racket theorem).
- $\boldsymbol\omega\times\mathbf h_{rw}$: the **gyroscopic stiffness** of the spinning wheels. If the body rotates while the wheels carry angular momentum, a torque appears about a perpendicular axis.

In the code $\dot{\boldsymbol\omega}$ is obtained with $J^{-1}$, computed only once because J is constant.

*Code:* `SatelliteEngine._derivatives()`.

---

## 8. Equation 4 — Quaternion kinematics

Quaternions represent the attitude **without singularities**, unlike Euler angles, which suffer from *gimbal lock*. They use 4 parameters, linked by the constraint $\lVert\mathbf q\rVert=1$. The kinematics, with $\boldsymbol\omega$ expressed in body axes, is:

$$
\dot{\mathbf q} = \frac12\,\mathbf q\otimes\begin{bmatrix}0\\ \boldsymbol\omega\end{bmatrix}
= \frac12\begin{bmatrix}
-q_1\omega_x - q_2\omega_y - q_3\omega_z\\
\ \ q_0\omega_x - q_3\omega_y + q_2\omega_z\\
\ \ q_3\omega_x + q_0\omega_y - q_1\omega_z\\
-q_2\omega_x + q_1\omega_y + q_0\omega_z
\end{bmatrix}
$$

The Hamilton product $\mathbf p\otimes\mathbf q = [p_0q_0-\mathbf p\cdot\mathbf q,\ p_0\mathbf q+q_0\mathbf p+\mathbf p\times\mathbf q]$ is implemented in `quaternion.quat_mult()`.

**Normalisation.** The exact equation preserves $\lVert\mathbf q\rVert$, but the numerical integrator does not: the RK4 truncation error makes the norm drift. After every step the quaternion is therefore projected onto the unit sphere $S^3$ with $\mathbf q\leftarrow\mathbf q/\lVert\mathbf q\rVert$.

**Attitude error.** Relative to a target $\mathbf q_t$:

$$
\mathbf q_e = \mathbf q_t^{*}\otimes\mathbf q,\qquad \theta_{err} = 2\arccos|q_{e,0}|
$$

The sign is chosen so that $q_{e,0}\ge0$. Quaternions cover SO(3) twice ($\mathbf q$ and $-\mathbf q$ are the same attitude), and this choice ensures that the controller always rotates along the shortest path (≤ 180°), avoiding *unwinding*.

---

## 9. Equation 5 — Environmental disturbance torques

$$
\mathbf T_{tot} = \mathbf T_{rw,action} + \mathbf T_{gg} + \mathbf T_{srp} + \mathbf T_{mag}\ (+\ \mathbf T_{manual})
$$

The disturbances depend on the position along the orbit, so an orbit model is needed. A **circular Keplerian orbit** at 500 km altitude and 51.6° inclination is used, with a period of about 94.5 minutes:

$$
\mathbf r_I(t) = R_3(-\Omega_{RAAN})\,R_1(-i)\ r\,[\cos u,\ \sin u,\ 0]^T,\qquad u = u_0 + n t,\quad n=\sqrt{\mu/r^3}
$$

All disturbances are **recomputed at every RK4 stage**, because they depend on the attitude $\mathbf q$ and on time $t$.

### 9.1 Gravity gradient

Gravity is not uniform over the body: the part closest to the Earth is pulled slightly more. Expanding the potential $-\mu/|\mathbf r+\boldsymbol\rho|$ to first order over the mass distribution gives:

$$
\mathbf T_{gg} = \frac{3\mu}{r^3}\ \hat{\mathbf r}_B\times\left(J\,\hat{\mathbf r}_B\right),\qquad \hat{\mathbf r}_B = R(\mathbf q)^T\,\frac{\mathbf r_I}{r}
$$

The torque vanishes when the radial direction coincides with a principal axis of inertia (checked in the tests). It tends to align the axis of **minimum** inertia (here $z_B$) with the local vertical, the principle behind passive gravity-gradient stabilisation. Order of magnitude: $10^{-9}$–$10^{-8}$ N·m.

### 9.2 Solar radiation pressure (SRP)

Solar photons carry momentum. The satellite is modelled with **6 flat faces**. For each lit face ($\cos\theta=\hat{\mathbf n}\cdot\hat{\mathbf s}>0$):

$$
\mathbf F = -P_\odot\,A\cos\theta\left[(1-\rho_s)\,\hat{\mathbf s} + 2\left(\rho_s\cos\theta + \tfrac{\rho_d}{3}\right)\hat{\mathbf n}\right],\qquad
\mathbf T_{srp} = \sum_{faces}\left(\mathbf r_{cp}-\mathbf r_{cm}\right)\times\mathbf F
$$

- $P_\odot = 4.56\cdot10^{-6}$ N/m² is the pressure at 1 AU.
- $\rho_s$ and $\rho_d$ are the specular and diffuse reflection fractions. The rest is absorbed.
- $\hat{\mathbf s}$ is the Sun direction in body axes.

The torque arises from the **offset between the centre of pressure and the centre of mass** $\mathbf r_{cm}$ (here [2, −1, 10] mm). With the centre of mass at the geometric centre the torques of opposite faces would cancel by symmetry.

**Eclipse.** A cylindrical shadow model is used: the satellite is in shadow if $\mathbf r\cdot\hat{\mathbf s}<0$ and its distance from the Earth-Sun axis is less than $R_\oplus$. In eclipse $\mathbf T_{srp}=0$ (shown in the GUI). Order of magnitude: $10^{-9}$ N·m.

### 9.3 Residual magnetic torque

Currents in the circuits and magnetised materials give the satellite a residual magnetic dipole $\mathbf m$ (here ~0.01 A·m²). This dipole interacts with the Earth's field, modelled as a **dipole** aligned with the polar axis:

$$
\mathbf B_I(\mathbf r) = B_0\left(\frac{R_\oplus}{r}\right)^3\left[3(\hat{\mathbf m}_\oplus\cdot\hat{\mathbf r})\hat{\mathbf r}-\hat{\mathbf m}_\oplus\right],\qquad
\mathbf T_{mag} = \mathbf m\times\left(R(\mathbf q)^T\mathbf B_I\right)
$$

with $B_0 = 3.12\cdot10^{-5}$ T. At 500 km $|\mathbf B|\approx 25$–$50\ \mu$T and $|\mathbf T_{mag}|\sim10^{-7}$ N·m. It is typically the **dominant disturbance** for a CubeSat in LEO.

### 9.4 Manual torque

A constant body torque, set with the GUI sliders (±5 mN·m), or a 1 s impulse. It is used to test visually the response of the controller and the saturation of the wheels. It is deliberately much larger than the natural disturbances.

*Code:* `environment.OrbitalEnvironment`.

---

## 10. RK4 numerical integration

The system $\dot{\mathbf x}=f(t,\mathbf x,\mathbf u)$ is integrated with **fixed-step 4th-order Runge-Kutta** ($\Delta t = 0.05$ s):

$$
\begin{aligned}
\mathbf k_1 &= f(t,\ \mathbf x_n,\ \mathbf u_n)\\
\mathbf k_2 &= f(t+\tfrac{\Delta t}{2},\ \mathbf x_n+\tfrac{\Delta t}{2}\mathbf k_1,\ \mathbf u_n)\\
\mathbf k_3 &= f(t+\tfrac{\Delta t}{2},\ \mathbf x_n+\tfrac{\Delta t}{2}\mathbf k_2,\ \mathbf u_n)\\
\mathbf k_4 &= f(t+\Delta t,\ \mathbf x_n+\Delta t\,\mathbf k_3,\ \mathbf u_n)\\
\mathbf x_{n+1} &= \mathbf x_n + \tfrac{\Delta t}{6}(\mathbf k_1+2\mathbf k_2+2\mathbf k_3+\mathbf k_4)
\end{aligned}
$$

Design choices:
- **Zero-order hold** on the command $\mathbf u_n$: the wheel torque stays constant for the whole step. This is what happens in an on-board computer, which updates the command at a fixed rate (here 20 Hz), and it is the natural semantics of `env.step(action)` in Gymnasium.
- **Fixed step:** it makes the simulation deterministic and reproducible, and the local error is $O(\Delta t^5)$. The time constants of the system, i.e. the period of the controller (~15 s) and the orbital period, are several orders of magnitude larger than $\Delta t$.
- After the step the quaternion is **normalised** (§8).

Performance: about 1.5 ms per step in pure Python with NumPy. The cross product is written by hand (`cross3`) because `np.cross` has an overhead that dominated the computation time on such small vectors.

---

## 11. Quaternion PD controller

The classical baseline controller for the RL agent:

$$
\mathbf T_{cmd} = -J\left(K_p\,\mathbf q_{e,vec} + K_d\,\boldsymbol\omega\right) + \boldsymbol\omega\times(J\boldsymbol\omega+\mathbf h_{rw})
$$

- The first term is an **inertia-normalised** PD. For small angles $\mathbf q_{e,vec}\approx\boldsymbol\theta/2$ and each axis becomes a second-order oscillator $\ddot\theta+K_d\dot\theta+\tfrac{K_p}{2}\theta=0$. Hence $K_p=2\omega_n^2$ and $K_d=2\zeta\omega_n$ (default $\omega_n=0.4$ rad/s, $\zeta=0.9$).
- The second term **compensates the gyroscopic coupling**, making the closed-loop dynamics almost linear.
- $\mathbf T_{cmd}$ is then allocated to the wheels (§6.3) and limited by the saturations (§6.4), including the torque rate limit: the PD can no longer make the torque jump from one step to the next.

**Stability.** Without disturbances or saturations, the gyroscopic compensation cancels the nonlinear term and what remains is $\dot{\boldsymbol\omega} = -K_p\mathbf q_{e,vec} - K_d\boldsymbol\omega$. Use as Lyapunov function

$$
V = 2K_p\,(1-q_{e,0}) + \tfrac12\,\boldsymbol\omega^T\boldsymbol\omega \;\ge 0
$$

Since $\dot q_{e,0} = -\tfrac12\mathbf q_{e,vec}^T\boldsymbol\omega$, its derivative is

$$
\dot V = K_p\,\mathbf q_{e,vec}^T\boldsymbol\omega + \boldsymbol\omega^T(-K_p\mathbf q_{e,vec} - K_d\boldsymbol\omega) = -K_d\,\lVert\boldsymbol\omega\rVert^2 \le 0
$$

By LaSalle's invariance principle the system converges to $\boldsymbol\omega=0,\ \mathbf q_{e,vec}=0$. Since there is no integral term, a small steady-state error (~0.003°) remains with constant disturbances.

---

## 12. Verification and validation

`tests/test_physics.py` contains 9 tests (`python -m pytest -q`):

| Test | Physical property checked |
|---|---|
| `test_inertia_matrix_3U` | J matches the analytical formula and is symmetric. |
| `test_torque_free_conservation` | Free motion (100 s of tumbling): constant $\mathbf H_I$ (error < 1e-9) and constant kinetic energy (relative error < 1e-7). $\lVert\mathbf q\rVert=1$. |
| `test_internal_torques_conserve_total_momentum` | Random wheel torques: the **total** inertial angular momentum is conserved, because the wheels only exchange internal momentum. |
| `test_wheel_saturation_is_exact_and_conservative` | Wheels at maximum torque reach exactly 6000 RPM without exceeding it, and the total momentum stays conserved. |
| `test_power_and_energy_idle` | At zero torque $P = N\cdot P_{static}$ and $E = N P_{static}\,t$. |
| `test_quaternion_kinematics_analytic` | Uniform rotation about a principal axis: matches the analytical solution $\mathbf q(t)=[\cos\tfrac{\omega t}{2},0,0,\sin\tfrac{\omega t}{2}]$. |
| `test_gravity_gradient_zero_on_principal_axis` | $\mathbf T_{gg}=0$ when nadir is aligned with a principal axis. |
| `test_pd_controller_converges` | From 90° and with disturbances: error < 0.1° after 80 s. |
| `test_torque_rate_limit` | With a step command (+T_max then −T_max) the applied torque changes by at most $\dot T_{max}\Delta t$ per step; reset zeroes the torque. |

The conservation of $\mathbf H$ is the most important test. It checks at once the sign consistency between equation 2 (wheels), equation 3 (Euler) and the allocation. A sign error in the reaction $-A\mathbf T_{rw}$ would make H grow systematically.

---

## 13. Default parameters

They are all in `satsim/config.py` (editable dataclasses).

| Parameter | Value | Notes |
|---|---|---|
| Mass, dimensions | 3 kg, 10×10×30 cm | 3U CubeSat |
| Centre-of-mass offset | [2, −1, 10] mm | lever arm of the SRP torque |
| $I_{rw}$ | 1.5·10⁻⁵ kg·m² | CubeSat-class wheel |
| $\Omega_{max}$ | 6000 RPM | $h_{max}\approx 9.4$ mN·m·s per wheel |
| $T_{max}$ | 2 mN·m | |
| $k_1, k_2, P_s$ | 50 W/(N·m), 1.2, 0.15 W | power model |
| Orbit | 500 km, i = 51.6° | circular |
| Residual dipole | [5, −3, 10] mA·m² | |
| $\rho_s, \rho_d$ | 0.1, 0.3 | |
| $\Delta t$ | 0.05 s | RK4 |
| PD | $\omega_n=0.4$ rad/s, $\zeta=0.9$ | |

---

## 14. Model limitations and future work

**Current simplifications**
- Circular Keplerian orbit, without J2 or atmospheric drag.
- The Sun has a fixed inertial direction: over a few hours its apparent motion is ~0.04°/h.
- The magnetic field is a dipole aligned with the polar axis, not the IGRF model.
- Aerodynamic torque is not modelled. At 500 km it is of the same order as SRP and gravity gradient.
- J is rigid and constant, without flexibility or sloshing. The transverse inertia of the wheels is included in J.
- The wheels are ideal, without viscous or Coulomb friction, motor delays or encoder noise.
- Perfect sensors: the controller knows the true state, with no gyroscope or star tracker models and no Kalman filter.

**Roadmap**
1. ~~Gymnasium environment and PPO training with Stable-Baselines3, compared quantitatively with the PD.~~ Done: see [rl/README.md](rl/README.md).
2. Randomisation of the initial conditions and of the parameters (*domain randomisation*).
3. Wheel friction, sensor noise, desaturation with magnetorquers.
4. Integration of the trained policy in the main GUI, selectable instead of the PD (today it is available in the comparison mode).

---

## 15. Development log

### v0.1.0 — 2026-09-23 — First version
- Decoupled physics engine (`satsim/`) with state $[\mathbf q,\boldsymbol\omega,\boldsymbol\Omega,E]$ and fixed-step RK4 integrator (Δt = 0.05 s) with zero-order hold on the command.
- The 5 equations implemented: J of the 3U with products of inertia; RW dynamics with exact speed and torque saturation, and power model $k_1|T|+k_2|T\Omega|+P_s$; Euler with gyroscopic coupling; quaternion kinematics with normalisation; gravity gradient, SRP (6 faces + cylindrical eclipse) and magnetic dipole disturbances.
- Two wheel configurations: 3 orthogonal and 4 in a pyramid, with pseudo-inverse allocation.
- Quaternion PD controller with gyroscopic compensation.
- GUI in PySide6 + pyqtgraph.opengl: 3D view (satellite, inertial and body frames, wheel axes, Sun, nadir), real-time plots, telemetry with saturation bars, mathematical model panel with live values, manual disturbance torques, pause and variable speed.
- 8 physics verification tests: conservation of H and energy, analytical solutions, saturation, PD convergence.
- Optimisation: hand-written cross product (from ~4 to ~1.5 ms per step).

### 2026-10-01 — Torque rate limit and comparison mode
- New physical parameter `ReactionWheelParams.max_torque_rate` (8 mN·m/s): the engine limits the torque change of each wheel to 0.4 mN·m per step, for any controller. The GUI PD now obeys the same limit as the RL agent. Effect on the PD (seed 42): peak \|α\| from 6.93 to 6.75 °/s²; time to get below 1° and final error unchanged.
- `python main.py --compare`: visual PPO vs PD comparison on the same seed (`gui/compare_window.py`, logic in `rl/compare.py`).
- Tests: `test_torque_rate_limit` (physics), `test_comparison_matches_evaluate` (RL).

### 2026-10-01 — Interface restyling
- New `gui/theme.py`, shared by the two windows: dark theme with "card" panels, accent colour, system font, Qt style sheet (QSS) for buttons, fields, checkboxes, sliders and bars; uniform pyqtgraph plot style.
- Material Design vector icons with **QtAwesome** instead of emojis. The dependency is optional: without QtAwesome the buttons simply have no icon.
- Comparison mode redesigned:
  - header with seed, initial conditions, episode progress and status;
  - horizontal toolbar, with speed as a segmented button selector;
  - metric cards with the better value highlighted in green, instead of the text table.

### 2026-10-04 — English translation
- The whole repository (code, comments, GUI strings, documentation) translated into English. This file was renamed from `DOCUMENTAZIONE_TECNICA.md` to `TECHNICAL_DOCUMENTATION.md`; the short presentation is in `README.md`.
