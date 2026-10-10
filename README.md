# satSim

**3D simulator of the attitude control of a CubeSat, with a Reinforcement Learning agent that learns to point it.**

A 3U CubeSat in space has to rotate and point precisely in a given direction. It does so with four reaction wheels: when a flywheel is spun up or slowed down, the satellite rotates the opposite way.

satSim simulates this system realistically and shows what happens in 3D. It also compares two ways of controlling it:
- a **classical controller** (PD), designed by hand;
- a **PPO agent**, a neural network trained by trial and error that learns by itself how to drive the wheels.

![PPO agent (left) vs PD controller (right), same initial condition](docs/images/compare.png)

## What you can do

- **Watch the satellite in 3D** as it rotates, with the reference frames, the wheel axes, the Sun and the Earth.
- **Follow the telemetry in real time**: pointing error, rotation rate, wheel speeds, power consumption.
- **Disturb the satellite**: sudden perturbations, external pushes, wheels driven to their speed limit.
- **Compare the agent and the classical controller** side by side, starting from the same situation.
- **Train new agents** and evaluate them against the classical controller.

## Results

Average over 10 test manoeuvres, starting 40–80° away from the target direction:

| | PPO agent | PD controller |
|---|---|---|
| Final pointing accuracy | **0.0004°** | 0.0031° |
| Time to get within 1° | **10.4 s** | 12.2 s |
| Energy used | 65.8 J | **61.3 J** |

The agent is about 8 times more precise and faster than the classical controller, with slightly higher consumption. The path that led to this result is told in the [RL development log](rl/README.md#9-rl-development-log).

## Quick start

Requires Python 3.10 or later.

```bash
git clone https://github.com/Maurizio19hub/satSim.git
cd satSim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py                        # 3D simulation with the classical controller
python -m pytest -q                   # automatic checks (physics + RL)
```

## Compare the agent with the classical controller

The best agent so far is included in the repository: `rl/pretrained/local_training_v4_seed1@3_best.zip`. It is the default model of the comparison, so no path is needed.

```bash
python main.py --compare              # 3D side by side: PPO agent (left) vs PD controller (right), seed 0
python main.py --compare --seed 105   # another initial condition (any integer seed)
python -m rl.evaluate rl/pretrained/local_training_v4_seed1@3_best   # numerical comparison on the 10 test manoeuvres
```

Another model can be loaded with `--model`, e.g. `python main.py --compare --model models/my_model_best`.

## Reproduce the training

The best agent was trained in two stages, 2 + 2 million steps (about 45 minutes each on a 4-core CPU):

```bash
# 1. Training from scratch, 2 M steps
python -m rl.train --subproc --seed 1 --out models/local_training_v4_seed1

# 2. Continue the same model for another 2 M steps, with a different seed
python -m rl.train --subproc --resume models/local_training_v4_seed1 --timesteps 2000000 \
                   --seed 3 --out models/local_training_v4_seed1@3

# 3. Evaluate the best model saved during training
python -m rl.evaluate models/local_training_v4_seed1@3_best
```

- During training the model is checked every 100 000 steps on separate validation manoeuvres, and the best one is saved as `<out>_best.zip`.
- Learning curves: `tensorboard --logdir runs/ppo_adcs`.
- Results can differ a little from run to run (different hardware, parallel environments): training is sensitive to the seed, see the [RL development log](rl/README.md#9-rl-development-log).

## How it is built

| Folder | Content |
|---|---|
| `satsim/` | The physics simulator: satellite dynamics, wheels, environmental disturbances, PD controller. |
| `gui/` | The 3D graphical interface and the comparison window. |
| `rl/` | The agent: training environment, training, evaluation and pre-trained models. |
| `tests/` | Automatic checks of the physics and of the RL environment. |

The simulator is independent of the interface: the agent trains without graphics and much faster than real time.

**Technologies:** Python, NumPy, PySide6 + pyqtgraph (3D graphics), Gymnasium and Stable-Baselines3 (Reinforcement Learning).

## Documentation

- [Technical documentation](TECHNICAL_DOCUMENTATION.md): physical model and equations, architecture, interface, parameters and verification.
- [Reinforcement Learning](rl/README.md): problem formulation, reward, training and the log of every version of the agent.
