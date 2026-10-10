# satSim — Technical documentation

This page explains how the simulator works: what it models, how the code is organised and how to use the interface. The full mathematical derivations are in [`docs/fisica_satSim.tex`](docs/fisica_satSim.tex); the Reinforcement Learning part is in [`rl/README.md`](rl/README.md).

---

## Contents

1. [Installation and launch](#1-installation-and-launch)
2. [The interface](#2-the-interface)
3. [Software architecture](#3-software-architecture)
4. [How a reaction wheel turns a satellite](#4-how-a-reaction-wheel-turns-a-satellite)
5. [What the simulator models](#5-what-the-simulator-models)
6. [The PD controller](#6-the-pd-controller)
7. [Verification](#7-verification)
8. [Default parameters](#8-default-parameters)
9. [Limitations and future work](#9-limitations-and-future-work)

---

## 1. Installation and launch

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python main.py                         # 4 wheels in a pyramid, real time
python main.py --wheels 3 --speed 5    # 3 orthogonal wheels, 5× real time
python main.py --seed 42               # reproducible initial conditions
python main.py --compare --seed 105    # PPO agent (left) vs PD controller (right)

python -m pytest -q                    # automatic checks
```

## 2. The interface

### Main window (`python main.py`)

| Area | Content |
|---|---|
| **3D view** | The 3U CubeSat (gold +Z face, purple +X face), the fixed target frame, the satellite's own frame, the four wheel axes, the Sun (yellow) and the direction of the Earth (cyan). Drag to rotate, scroll to zoom. |
| **Controls** | Pause (`Space`), reset with a random tumble, sudden 45° perturbation, 1 s torque impulse, simulation speed (0.25×–20×), PD and disturbances on/off, sliders to push the satellite with an external torque. |
| **Plots** | Last 60 s of attitude, pointing error, rotation rate, wheel speeds (with the ±6000 RPM limits) and power. |
| **Telemetry** | Current values, wheel saturation bars (green / yellow / red), power and energy used, disturbance torques. |
| **Model panel** | The equations of the model with their live values. |

**Try this:** with the PD on, push the `T_z` slider to +2 mN·m. The wheels absorb the push until they hit 6000 RPM, then the satellite loses control. This is why real satellites periodically "unload" their wheels (*momentum dumping*).

### Comparison window (`python main.py --compare`)

Two satellites start from exactly the same situation and move side by side: the trained agent on the left, the PD on the right. Cards and plots compare pointing error, rotation rate, acceleration, wheel speed, energy and reward. After 100 s the episode ends with a summary. The numbers are the same as `python -m rl.evaluate` on that seed.

## 3. Software architecture

```
satSim/
├── main.py              # entry point
├── satsim/              # physics engine — no graphics, no RL
│   ├── config.py        # all parameters
│   ├── quaternion.py    # attitude maths
│   ├── environment.py   # orbit, Sun, eclipse, Earth's magnetic field, disturbances
│   ├── physics_engine.py# satellite + wheels dynamics, integration, telemetry
│   ├── controller.py    # PD controller
│   └── simulation.py    # closed loop used by the GUI
├── gui/                 # 3D view, plots, windows
├── rl/                  # RL environment, training, evaluation, trained models
├── tests/               # automatic checks
└── docs/                # full mathematical documentation (LaTeX)
```

- **The physics knows nothing about the GUI or RL.** The same engine runs in the 3D window and, much faster than real time, during training.
- Dependencies go one way only: `gui → satsim` and `rl → satsim`.
- Typical use of the engine:

```python
from satsim import SatelliteEngine, SimParams

eng = SatelliteEngine(SimParams())
tel = eng.reset()                    # initial state -> telemetry
tel = eng.step(wheel_torques)        # advance 0.05 s with the given motor torques
```

## 4. How a reaction wheel turns a satellite

A reaction wheel is a small flywheel driven by an electric motor. When the motor spins the wheel one way, the satellite turns the other way: action and reaction. No propellant is used. The satellite simply **exchanges rotation (angular momentum) with its wheels**.

Two consequences shape the whole problem:

- **Wheels only move momentum around.** External disturbances (gravity, sunlight, magnetism) slowly *add* momentum, and the wheels have to absorb it.
- **Wheels have a top speed (6000 RPM).** A wheel at its limit can no longer push in that direction, and control of that axis is lost until the wheel is unloaded.

The controller, PD or RL agent, has to point the satellite fast and accurately while keeping the wheels away from their limits and using as little energy as possible.

## 5. What the simulator models

| Part | What is simulated |
|---|---|
| **Satellite** | A rigid 3 kg box, 10 × 10 × 30 cm. Mass is slightly unevenly distributed, as in a real satellite, so the axes are weakly coupled. The long axis (z) is 5 times easier to rotate than the other two. |
| **Wheels** | 4 wheels in a pyramid (default) or 3 orthogonal wheels. With 4 wheels the system is redundant: if one fails, the other three still control all axes. |
| **Wheel limits** | Max torque 2 mN·m; torque can change by at most 0.4 mN·m per step (motor driver); max speed 6000 RPM, reached exactly without ever exceeding it; body acceleration limited to 10 °/s² per axis (safety filter applied to every controller). |
| **Power** | Each wheel uses power for its electronics (always on), for the motor current and for the mechanical work on the rotor. Braking energy is not recovered. The total energy used is tracked. |
| **Rotational dynamics** | The full nonlinear rigid-body equations, including the gyroscopic effects between body and spinning wheels. |
| **Attitude** | Represented with quaternions, which avoid the singularities of Euler angles (*gimbal lock*). The controller always turns the short way round. |
| **Orbit** | Circular orbit at 500 km, 51.6° inclination (like the ISS), period ~94 min. |
| **Disturbances** | Gravity gradient (gravity pulls the near end slightly harder), solar radiation pressure (with eclipses) and the satellite's residual magnetism in the Earth's field. They are tiny (10⁻⁹–10⁻⁷ N·m) but they accumulate over time. Manual torques from the GUI are much larger, for testing. |
| **Integration** | Fixed 0.05 s step (20 Hz, like an on-board computer) with a 4th-order Runge-Kutta method. Commands are held constant over each step. About 1.5 ms of computing per step. |

### Body acceleration limit

There is no universal safety threshold on angular acceleration: for a small rigid CubeSat the structural loads are negligible, and the real limits come from the mission (sensors, payload stability, flexible appendages). The simulator uses **10 °/s² per axis** as a design requirement: about the PD's peak and the most the wheels can give on the x/y axes. In practice it only trims the z axis, which is much lighter. The filter adjusts the wheel torques before they are applied, so it holds for the PD and for the RL agent alike.

## 6. The PD controller

The classical baseline the RL agent is compared with. It pushes the satellite towards the target in proportion to the pointing error (P) and brakes in proportion to the rotation rate (D), scaled by the satellite's inertia so that every axis behaves the same. It also cancels the gyroscopic effects. The requested torque is then shared among the wheels in the most economical way.

With the default gains it is close to critically damped: it settles without overshoot in about 10–15 s for a typical manoeuvre and keeps a small residual error (~0.003°). Its stability is proven with a Lyapunov argument (see `docs/fisica_satSim.tex`).

## 7. Verification

`tests/` contains automatic checks (`python -m pytest -q`). The most important physics checks:

| Check | Why it matters |
|---|---|
| Total angular momentum is conserved, with or without random wheel torques | Confirms that wheels, body dynamics and torque allocation are consistent. A single sign error would break it. |
| Free rotation keeps its kinetic energy | The integrator does not create or lose energy. |
| Uniform rotation matches the exact analytical solution | The attitude kinematics are correct. |
| Wheels saturate exactly at 6000 RPM | Limits are enforced without breaking conservation. |
| Torque rate and body acceleration limits are respected | The safety limits hold for any command. |
| The PD brings the satellite from 90° to < 0.1° | The baseline controller works with disturbances on. |

The RL tests check the environment, that training never loads the GUI, and that the saved models load on any Python version.

## 8. Default parameters

All in `satsim/config.py`.

| Parameter | Value |
|---|---|
| Satellite | 3U CubeSat, 3 kg, 10 × 10 × 30 cm |
| Wheels | 4 in a pyramid, max 2 mN·m and 6000 RPM each |
| Torque rate limit | 8 mN·m/s (0.4 mN·m per step) |
| Body acceleration limit | 10 °/s² per axis |
| Orbit | circular, 500 km, 51.6° |
| Time step | 0.05 s (20 Hz) |
| PD | natural frequency 0.4 rad/s, damping 0.9 |

## 9. Limitations and future work

**Current simplifications**
- Ideal wheels: no friction, no motor delay, no encoder noise.
- Perfect sensors: the controller knows the true attitude and rate. There are no gyroscope or star tracker models and no estimator.
- Simple environment: circular orbit, fixed Sun direction, dipole magnetic field, no aerodynamic drag.
- Rigid satellite: no flexible solar panels or fuel sloshing.

**Possible extensions**
- Randomised satellite parameters and disturbances during training, for a more robust agent.
- Sensor noise and a Kalman filter between the sensors and the controller.
- Wheel failure: control with only 3 of the 4 wheels.
- Magnetorquers to unload the wheels over long, multi-orbit episodes.
- Other baselines (LQR, MPC) and other RL algorithms.
