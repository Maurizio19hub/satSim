"""
Verification tests of the physics engine (conservation laws and analytical solutions).

Run with:  python -m pytest -q
"""
import numpy as np
import pytest

from satsim import QuaternionPDController, SatelliteEngine, SimParams
from satsim.quaternion import quat_from_axis_angle, quat_to_dcm


def make_engine(disturbances=False, products=None, config="pyramid4", accel_limit=True):
    p = SimParams()
    p.wheels.configuration = config
    if not accel_limit:
        p.wheels.max_body_accel_deg = float("inf")
    d = p.disturbances
    d.enable_gravity_gradient = d.enable_srp = d.enable_magnetic = disturbances
    if products is not None:
        p.spacecraft.products_of_inertia = products
    return SatelliteEngine(p)


def kinetic_energy(e):
    w = e.omega
    return 0.5 * w @ e.J @ w


def test_inertia_matrix_3U():
    e = make_engine(products=(0, 0, 0))
    # m=3 kg, 10x10x30 cm → Jxx = Jyy = 3/12·(0.01+0.09) = 0.025, Jzz = 3/12·0.02 = 0.005
    assert np.allclose(np.diag(e.J), [0.025, 0.025, 0.005])
    assert np.allclose(e.J, e.J.T)


def test_torque_free_conservation():
    """No external torques and wheels at rest: inertial ‖H‖ and kinetic energy are constant."""
    e = make_engine()
    e.reset(omega0=[0.3, -0.2, 0.5])
    H0, K0 = e._telemetry(np.zeros(4), np.zeros(4), np.zeros(3)).H_inertial, kinetic_energy(e)
    for _ in range(2000):                         # 100 s of tumbling
        tel = e.step(np.zeros(4))
    assert np.allclose(tel.H_inertial, H0, rtol=0, atol=1e-9)
    assert abs(kinetic_energy(e) - K0) / K0 < 1e-7
    assert abs(np.linalg.norm(tel.q) - 1) < 1e-12


def test_internal_torques_conserve_total_momentum():
    """The wheels exchange momentum with the body: the total inertial H must stay constant."""
    e = make_engine()
    e.reset(omega0=[0.05, 0.02, -0.04])
    rng = np.random.default_rng(0)
    H0 = e.step(np.zeros(4)).H_inertial
    for _ in range(2000):
        tel = e.step(rng.uniform(-2e-3, 2e-3, 4))
    assert np.allclose(tel.H_inertial, H0, rtol=0, atol=1e-9)


def test_wheel_saturation_is_exact_and_conservative():
    e = make_engine(config="orthogonal3")
    H0 = e.step(np.zeros(3)).H_inertial
    for _ in range(4000):                         # maximum torque for 200 s → saturation
        tel = e.step(np.full(3, 1.0))
    assert np.all(tel.wheel_rpm <= 6000 + 1e-9)
    assert np.allclose(tel.wheel_rpm, 6000)
    assert np.allclose(tel.wheel_torque, 0)       # at saturation the wheel accepts no torque
    assert np.allclose(tel.H_inertial, H0, rtol=0, atol=1e-9)


def test_power_and_energy_idle():
    e = make_engine()
    for _ in range(200):
        tel = e.step(np.zeros(4))
    p_static = e.params.wheels.p_static
    assert tel.power_total == pytest.approx(4 * p_static)
    assert tel.energy == pytest.approx(4 * p_static * 200 * e.dt)


def test_quaternion_kinematics_analytic():
    """Uniform rotation about a principal axis: q(t) = [cos(ωt/2), 0, 0, sin(ωt/2)]."""
    e = make_engine(products=(0, 0, 0))
    w = 0.7
    e.reset(omega0=[0, 0, w])
    for _ in range(400):
        tel = e.step(np.zeros(4))
    th = w * tel.t
    assert np.allclose(tel.q, [np.cos(th / 2), 0, 0, np.sin(th / 2)], atol=1e-9)


def test_gravity_gradient_zero_on_principal_axis():
    e = make_engine(disturbances=True, products=(0, 0, 0))
    r = e.env.position(0.0)
    # Align z_B with the radial direction: T_gg must vanish
    z = np.array([0, 0, 1.0])
    rh = r / np.linalg.norm(r)
    axis = np.cross(z, rh)
    q = quat_from_axis_angle(axis, np.arccos(z @ rh))
    assert np.allclose(quat_to_dcm(q) @ z, rh)
    d = e.env.disturbance_torques(0.0, quat_to_dcm(q), e.J)
    assert np.linalg.norm(d["gg"]) < 1e-15


def test_pd_controller_converges():
    e = make_engine(disturbances=True)
    c = QuaternionPDController(e.J)
    e.reset(q0=quat_from_axis_angle([1, -2, 0.5], np.radians(90)), omega0=[0.02, -0.03, 0.05])
    for _ in range(int(80 / e.dt)):
        tel = e.step(e.allocate(c.compute(e.q, e.omega, e.h_rw, e.q_target)))
    assert tel.att_err_deg < 0.1
    assert np.linalg.norm(tel.omega) < 1e-3


def test_torque_rate_limit():
    """The applied torque cannot change by more than max_torque_rate·dt per step."""
    e = make_engine(accel_limit=False)       # equal torques = pure z torque, above the accel limit
    d_max = e.params.wheels.max_torque_rate * e.dt
    tau_max = e.params.wheels.max_torque
    prev = np.zeros(4)
    for k in range(30):
        cmd = np.full(4, tau_max if k < 15 else -tau_max)    # step to +T_max, then −T_max
        tel = e.step(cmd)
        assert np.all(np.abs(tel.wheel_torque - prev) <= d_max + 1e-15)
        prev = tel.wheel_torque
    np.testing.assert_allclose(tel.wheel_torque, -tau_max)   # after 15 steps it reaches −T_max
    e.reset()
    np.testing.assert_array_equal(e.tau_cmd, 0.0)             # reset zeroes the torque


def test_body_acceleration_limit():
    """With random torque commands the body acceleration never exceeds max_body_accel_deg."""
    e = make_engine(disturbances=True)
    a_max = e.params.wheels.max_body_accel_deg
    rng = np.random.default_rng(1)
    e.reset(omega0=[0.05, -0.03, 0.04])
    w_prev, peak = e.omega, 0.0
    for _ in range(2000):
        tel = e.step(rng.uniform(-2e-3, 2e-3, 4))
        peak = max(peak, np.degrees(np.abs((tel.omega - w_prev) / e.dt)).max())
        w_prev = tel.omega
    assert peak <= a_max * 1.01                  # 1 %: change of the gyroscopic term within a step
    assert peak > 0.9 * a_max                    # the limit is actually reached (z axis)


def test_body_acceleration_limit_keeps_momentum():
    """The filter only changes the commanded torque: total momentum is still conserved."""
    e = make_engine()
    e.reset(omega0=[0.05, 0.02, -0.04])
    rng = np.random.default_rng(0)
    H0 = e.step(np.zeros(4)).H_inertial
    for _ in range(2000):
        tel = e.step(rng.uniform(-2e-3, 2e-3, 4))
    assert np.allclose(tel.H_inertial, H0, rtol=0, atol=1e-9)
