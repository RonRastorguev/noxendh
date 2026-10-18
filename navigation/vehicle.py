"""The vehicle interface: the ONLY way flight software talks to a flight controller (docs/SCOUT_NAV.md §2).

Implementations:
    flightsim.vehicle_sim.SimVehicle      PX4-like closed-loop model for offline tests
    (later) a PX4 adapter over uXRCE-DDS  converts NED <-> the z-up odom frame, nothing else

Rules:
- Setpoints are in odom (navigation.types.Setpoint) and must be streamed at >= 10 Hz; PX4 leaves offboard when they
  stop (COM_OF_LOSS_T), which is the Scout's independent backstop for a dead laptop, link or companion (DH-05).
- `request_*` calls are asynchronous requests; success is read back from `status()`, never assumed.
- `capabilities()` says how the FC estimates height etc.; callers read capabilities, never branch on vendor names.
"""
from __future__ import annotations

from typing import Protocol

from .types import OdomSample, Setpoint, VehicleStatus


class VehicleInterface(Protocol):
    def odometry(self) -> OdomSample | None:
        """Latest state estimate (None before the FC has one)."""

    def status(self) -> VehicleStatus:
        ...

    def send_setpoint(self, sp: Setpoint) -> None:
        ...

    def request_arm(self) -> None:
        ...

    def request_offboard(self) -> None:
        ...

    def request_land(self) -> None:
        """Hand the descent to the FC's own land mode (it lands on its rangefinder and disarms)."""

    def capabilities(self) -> dict:
        """e.g. {"height_ref": "range" | "baro" | "ev", "max_xy_vel_mps": 1.5, "max_z_vel_up_mps": 1.0,
        "max_z_vel_dn_mps": 0.7, "max_yaw_rate_dps": 60.0}."""
