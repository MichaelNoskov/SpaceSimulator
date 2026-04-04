from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Command:
    """
    Control inputs for one simulation tick/frame.

    Fields are optional (often one-shot). `Controller` applies them safely.
    """

    # Continuous controls
    throttle_0_1: Optional[float] = None
    engine_on: Optional[bool] = None

    # One-shot event requests
    request_heatshield_jettison: bool = False
    request_drogue: bool = False
    request_main: bool = False
    request_chute_jettison: bool = False

    # Target selection
    set_target_world: Optional[tuple[float, float]] = None

    # Telemetry CSV (toggle via UI)
    request_toggle_csv_logging: bool = False

