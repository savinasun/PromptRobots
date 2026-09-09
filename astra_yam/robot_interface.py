"""Robot backends: the real bimanual YAM via the gello ZMQ robot server, or a kinematic simulator.

`ZmqYamRobot` speaks the gello `robot_node` protocol (REQ/REP, pickled {"method", "args"} dicts) with its own
socket handling: LINGER=0 and hard timeouts, so a dead server raises `RobotError` immediately instead of
blocking (gello's ZMQClientRobot sleeps for exponential back-off and its context.term() can hang forever on an
undelivered request).
"""
from __future__ import annotations

import pickle
import sys
from typing import Any, Optional, Protocol

import numpy as np

from astra_yam.config import NUM_DOFS


class RobotError(RuntimeError):
    """Raised when the robot server does not answer or returns malformed data."""


class RobotBackend(Protocol):
    def num_dofs(self) -> int: ...
    def get_joint_positions(self) -> np.ndarray: ...
    def command_joint_positions(self, q14: np.ndarray) -> None: ...
    def close(self) -> None: ...


def ensure_gello_on_path(gello_software_path: Optional[str]) -> None:
    """Make `gello` (and its `third_party`) importable from the given gello_software checkout."""
    try:
        import gello  # noqa: F401
        return
    except ImportError:
        pass
    if gello_software_path and gello_software_path not in sys.path:
        sys.path.insert(0, gello_software_path)
    try:
        import gello  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"cannot import gello from '{gello_software_path}'. Set robot.gello_software_path (or "
            f"GELLO_SOFTWARE_PATH) to the skild-gello/gello_software checkout."
        ) from e


class ZmqYamRobot:
    """Client for the bimanual YAM robot server (experiments/launch_nodes.py --robot=bimanual_yam, port 6001)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 6001, timeout_ms: int = 2000,
                 gello_software_path: Optional[str] = None):
        import zmq

        self._zmq = zmq
        self._host, self._port, self._timeout_ms = host, int(port), int(timeout_ms)
        self._addr = f"tcp://{host}:{port}"
        self._ctx = zmq.Context()
        self._sock = None
        self._connect()
        n = self._call("num_dofs")
        if not isinstance(n, (int, np.integer)):
            self.close()
            raise RobotError(f"robot server at {self._addr} returned {n!r} for num_dofs")
        if int(n) != NUM_DOFS:
            self.close()
            raise RobotError(f"robot server reports {n} DoF, expected {NUM_DOFS} (bimanual YAM)")
        self._n = int(n)

    # ------------------------------------------------------------ transport
    def _connect(self) -> None:
        zmq = self._zmq
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.connect(self._addr)

    def _call(self, method: str, **args: Any) -> Any:
        if self._sock is None:
            raise RobotError("robot connection is closed")
        req = {"method": method}
        if args:
            req["args"] = args
        try:
            self._sock.send(pickle.dumps(req))
            reply = self._sock.recv()
        except self._zmq.Again:
            self._connect()  # reset the REQ state machine for the next call
            raise RobotError(f"robot server at {self._addr} did not answer '{method}' within {self._timeout_ms} ms "
                             f"- is launch_nodes.py running?")
        except self._zmq.ZMQError as e:
            self._connect()
            raise RobotError(f"ZMQ error talking to {self._addr}: {e}")
        result = pickle.loads(reply)
        if isinstance(result, dict) and "error" in result and len(result) == 1:
            raise RobotError(f"robot server error for '{method}': {result['error']}")
        return result

    # ------------------------------------------------------------- interface
    def num_dofs(self) -> int:
        return self._n

    def get_observations(self) -> dict:
        obs = self._call("get_observations")
        if not isinstance(obs, dict) or "joint_positions" not in obs:
            raise RobotError(f"malformed observations from robot server: {type(obs).__name__}")
        return obs

    def get_joint_positions(self) -> np.ndarray:
        q = np.asarray(self.get_observations()["joint_positions"], dtype=float).reshape(-1)
        if q.shape[0] != NUM_DOFS or not np.all(np.isfinite(q)):
            raise RobotError(f"malformed joint_positions from robot server: {q}")
        return q

    def command_joint_positions(self, q14: np.ndarray) -> None:
        q = np.asarray(q14, dtype=float).reshape(-1)
        if q.shape[0] != NUM_DOFS:
            raise ValueError(f"expected {NUM_DOFS} joint values, got {q.shape[0]}")
        if not np.all(np.isfinite(q)):
            raise ValueError("non-finite joint command refused")
        self._call("command_joint_state", joint_state=q)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

    def __repr__(self) -> str:
        return f"ZmqYamRobot({self._addr})"
