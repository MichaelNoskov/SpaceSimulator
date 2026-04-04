from __future__ import annotations

import numpy as np

from digital_twin.model import PhysicsModel
from .commands import Command


class Controller:
    """
    Applies UI commands to the physics model.

    Only here may user input mutate model state.
    """

    def __init__(self, model: PhysicsModel):
        self.model = model
        self._pending = Command()

    def queue(self, cmd: Command) -> None:
        """Merge a command into the pending buffer."""

        if cmd.throttle_0_1 is not None:
            self._pending.throttle_0_1 = float(cmd.throttle_0_1)
        if cmd.engine_on is not None:
            self._pending.engine_on = bool(cmd.engine_on)

        self._pending.request_heatshield_jettison |= bool(cmd.request_heatshield_jettison)
        self._pending.request_drogue |= bool(cmd.request_drogue)
        self._pending.request_main |= bool(cmd.request_main)
        self._pending.request_chute_jettison |= bool(cmd.request_chute_jettison)

        if cmd.set_target_world is not None:
            self._pending.set_target_world = (float(cmd.set_target_world[0]), float(cmd.set_target_world[1]))

        self._pending.request_toggle_csv_logging |= bool(cmd.request_toggle_csv_logging)

    def consume_and_apply(self) -> None:
        """Apply pending commands and clear the buffer."""

        cmd = self._pending
        self._pending = Command()
        self.apply(cmd)

    def apply(self, cmd: Command) -> None:
        # Target selection (pure data update)
        if cmd.set_target_world is not None:
            self.model.set_target_world(cmd.set_target_world[0], cmd.set_target_world[1])

        # Engine toggle (explicit)
        if cmd.engine_on is not None:
            self.model.set_engine(bool(cmd.engine_on))

        # Throttle continuous (only meaningful when engine on)
        if cmd.throttle_0_1 is not None:
            self.model.set_throttle(float(np.clip(cmd.throttle_0_1, 0.0, 1.0)))

        # One-shot event requests with validation rules.
        if cmd.request_heatshield_jettison:
            self.model.request_heatshield_jettison()

        if cmd.request_drogue:
            self.model.request_drogue()

        if cmd.request_main:
            self.model.request_main()

        if cmd.request_chute_jettison:
            self.model.request_chute_jettison()

        if cmd.request_toggle_csv_logging:
            self.model.set_csv_logging(not self.model.csv_logging_enabled)

    # UI helpers (lever availability)
    def can_heatshield_jettison(self) -> bool:
        return bool(self.model.can_heatshield_jettison)

    def can_drogue(self) -> bool:
        return bool(self.model.can_drogue)

    def can_main(self) -> bool:
        return bool(self.model.can_main)

    def can_chute_jettison(self) -> bool:
        return bool(self.model.can_chute_jettison)

