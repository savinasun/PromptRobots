import socket
import threading
import time

import numpy as np
import pytest

from astra_yam.config import DEFAULT_GELLO_SOFTWARE, REFERENCE_HOME_JOINTS
from astra_yam.robot_interface import RobotError, ZmqYamRobot, ensure_gello_on_path
from astra_yam.sim import SimYamRobot, start_sim_zmq_thread


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _home():
    q0 = np.zeros(14)
    q0[0:6] = q0[7:13] = REFERENCE_HOME_JOINTS
    q0[6] = q0[13] = 1.0
    return q0


def _exercise(robot, q0):
    assert robot.num_dofs() == 14
    assert np.allclose(robot.get_joint_positions(), q0)
    q_new = q0.copy()
    q_new[6] = 0.3
    robot.command_joint_positions(q_new)
    assert np.allclose(robot.get_joint_positions(), q_new)
    assert "joint_positions" in robot.get_observations()


def test_client_against_builtin_sim_server():
    pytest.importorskip("zmq")
    sim = SimYamRobot(initial_q=_home())
    port = _free_port()
    thread, stop = start_sim_zmq_thread(sim, port=port)
    try:
        robot = ZmqYamRobot("127.0.0.1", port, timeout_ms=2000)
        _exercise(robot, _home())
        robot.close()
    finally:
        stop.set()
        thread.join(timeout=2)


def test_client_against_real_gello_server_class():
    """Serve the simulator with gello's own ZMQServerRobot to prove protocol compatibility."""
    pytest.importorskip("zmq")
    try:
        ensure_gello_on_path(DEFAULT_GELLO_SOFTWARE)
        from gello.zmq_core.robot_node import ZMQServerRobot
    except ImportError:
        pytest.skip("gello_software not available")
    sim = SimYamRobot(initial_q=_home())
    port = _free_port()
    server = ZMQServerRobot(sim, port=port, host="127.0.0.1")
    t = threading.Thread(target=server.serve, daemon=True)
    t.start()
    time.sleep(0.2)
    try:
        robot = ZmqYamRobot("127.0.0.1", port, timeout_ms=2000)
        _exercise(robot, _home())
        robot.close()
    finally:
        server.stop()
        t.join(timeout=3)


def test_dead_server_fails_fast_and_close_does_not_hang():
    pytest.importorskip("zmq")
    port = _free_port()
    t0 = time.perf_counter()
    with pytest.raises(RobotError, match="did not answer"):
        ZmqYamRobot("127.0.0.1", port, timeout_ms=300)
    assert time.perf_counter() - t0 < 3.0
