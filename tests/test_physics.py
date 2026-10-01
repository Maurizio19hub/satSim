"""
Test di verifica dell'engine fisico (leggi di conservazione e soluzioni analitiche).

Esecuzione:  python -m pytest -q
"""
import numpy as np
import pytest

from satsim import QuaternionPDController, SatelliteEngine, SimParams
from satsim.quaternion import quat_from_axis_angle, quat_to_dcm


def make_engine(disturbances=False, products=None, config="pyramid4"):
    p = SimParams()
    p.wheels.configuration = config
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
    """Senza coppie esterne e ruote ferme: ‖H‖ inerziale ed energia cinetica costanti."""
    e = make_engine()
    e.reset(omega0=[0.3, -0.2, 0.5])
    H0, K0 = e._telemetry(np.zeros(4), np.zeros(4), np.zeros(3)).H_inertial, kinetic_energy(e)
    for _ in range(2000):                         # 100 s di tumbling
        tel = e.step(np.zeros(4))
    assert np.allclose(tel.H_inertial, H0, rtol=0, atol=1e-9)
    assert abs(kinetic_energy(e) - K0) / K0 < 1e-7
    assert abs(np.linalg.norm(tel.q) - 1) < 1e-12


def test_internal_torques_conserve_total_momentum():
    """Le ruote scambiano momento col corpo: H totale inerziale deve restare costante."""
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
    for _ in range(4000):                         # coppia massima per 200 s → saturazione
        tel = e.step(np.full(3, 1.0))
    assert np.all(tel.wheel_rpm <= 6000 + 1e-9)
    assert np.allclose(tel.wheel_rpm, 6000)
    assert np.allclose(tel.wheel_torque, 0)       # a saturazione la ruota non accetta coppia
    assert np.allclose(tel.H_inertial, H0, rtol=0, atol=1e-9)


def test_power_and_energy_idle():
    e = make_engine()
    for _ in range(200):
        tel = e.step(np.zeros(4))
    p_static = e.params.wheels.p_static
    assert tel.power_total == pytest.approx(4 * p_static)
    assert tel.energy == pytest.approx(4 * p_static * 200 * e.dt)


def test_quaternion_kinematics_analytic():
    """Rotazione uniforme attorno a un asse principale: q(t) = [cos(ωt/2), 0, 0, sin(ωt/2)]."""
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
    # Allinea z_B alla direzione radiale: T_gg deve annullarsi
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
    """La coppia applicata non può variare più di max_torque_rate·dt per passo."""
    e = make_engine()
    d_max = e.params.wheels.max_torque_rate * e.dt
    tau_max = e.params.wheels.max_torque
    prev = np.zeros(4)
    for k in range(30):
        cmd = np.full(4, tau_max if k < 15 else -tau_max)    # gradino +T_max, poi −T_max
        tel = e.step(cmd)
        assert np.all(np.abs(tel.wheel_torque - prev) <= d_max + 1e-15)
        prev = tel.wheel_torque
    np.testing.assert_allclose(tel.wheel_torque, -tau_max)   # dopo 15 passi arriva a −T_max
    e.reset()
    np.testing.assert_array_equal(e.tau_cmd, 0.0)             # il reset azzera la coppia
