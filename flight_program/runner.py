from __future__ import annotations

import math
from typing import Any, Callable, Optional

from control.commands import Command
from control.controller import Controller
from digital_twin.model import PhysicsModel, SimResult

DEFAULT_SCRIPT = """def tick(sim, ap):
    # Safety gates are in sim.can_* (Mach, altitude, heatshield before drogue, science hold).
    if sim.can_heatshield_jettison:
        ap.request_heatshield_jettison()
    if sim.can_drogue:
        ap.request_drogue()
    if sim.can_main:
        ap.request_main()
    if sim.can_chute_jettison:
        ap.request_chute_jettison()
    if sim.altitude_m < sim.parachute_jettison_max_alt_m and sim.chute_jettisoned:
        ap.set_engine(True)
        target_v = -20.0 if sim.altitude_m > 200.0 else -4.0
        error = target_v - sim.vertical_speed_mps
        cmd = ap.clamp(0.02 * error, 0.0, 1.0)
        ap.set_throttle_slider(cmd)
"""

# Full identifiers for hint panel and Tab completion (longest first helps some UIs; sorted for display)
SIM_API_COMPLETIONS: tuple[str, ...] = (
    "sim.parachute_jettison_max_alt_m",
    "sim.altitude_m",
    "sim.atm_pressure_bar",
    "sim.can_chute_jettison",
    "sim.can_drogue",
    "sim.can_heatshield_jettison",
    "sim.can_main",
    "sim.chute_jettisoned",
    "sim.distance_to_target_m",
    "sim.drogue_deployed",
    "sim.engine_on",
    "sim.fuel_kg",
    "sim.g_load",
    "sim.heatshield_jettisoned",
    "sim.heatshield_skin_temp_c",
    "sim.horizontal_speed_mps",
    "sim.internal_temp_c",
    "sim.main_deployed",
    "sim.result",
    "sim.time_s",
    "sim.vertical_speed_mps",
)

AP_API_COMPLETIONS: tuple[str, ...] = (
    "ap.clamp(x, lo, hi)",
    "ap.request_chute_jettison()",
    "ap.request_drogue()",
    "ap.request_heatshield_jettison()",
    "ap.request_main()",
    "ap.set_engine(True)",
    "ap.set_throttle_slider(0.5)",
    "ap.sleep(10)",
    "sleep(10)",
)

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sum": sum,
    "any": any,
    "all": all,
    "True": True,
    "False": False,
    "None": None,
    "tuple": tuple,
    "list": list,
    "dict": dict,
    "set": set,
    "pow": pow,
    "divmod": divmod,
}


class FlightProgramSleep(BaseException):
    """Stops tick(); runner resumes after sim time reaches until_s."""

    __slots__ = ("until_s",)

    def __init__(self, until_s: float) -> None:
        self.until_s = float(until_s)


def _sleep_duration_int_s(seconds: Any) -> int:
    """Whole seconds of PhysicsModel.time_s; fractional values truncated like int()."""
    return max(0, int(seconds))


def _sleep_stub(_seconds: float) -> None:
    """Replaced before each real tick; must not run from user code directly."""
    raise FlightProgramSleep(-1.0)


def compile_flight_program(
    source: str,
) -> tuple[Optional[Callable[..., None]], Optional[str], Optional[dict[str, Any]]]:
    """
    Compile and exec user source in a restricted namespace.
    Returns (tick, error_message, globals_dict). On failure tick and globals are None.
    """
    g: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "math": math, "sleep": _sleep_stub}
    try:
        code = compile(source, "<flight_program>", "exec")
        exec(code, g, g)
    except SyntaxError as e:
        return None, f"SyntaxError: {e}", None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", None
    tick = g.get("tick")
    if tick is None or not callable(tick):
        return None, "Define tick(sim, ap)", None
    return tick, None, g


class SimView:
    """Read-only view of simulator state for flight scripts."""

    __slots__ = ("_m", "_stats")

    def __init__(self, model: PhysicsModel, stats: Optional[dict[str, Any]] = None) -> None:
        self._m = model
        self._stats = stats or {}

    @property
    def altitude_m(self) -> float:
        return float(self._m.altitude_m)

    @property
    def vertical_speed_mps(self) -> float:
        return float(self._m.vertical_speed_mps)

    @property
    def horizontal_speed_mps(self) -> float:
        return float(self._m.horizontal_speed_mps)

    @property
    def time_s(self) -> float:
        return float(self._m.time_s)

    @property
    def result(self) -> SimResult:
        return self._m.result

    @property
    def engine_on(self) -> bool:
        return bool(self._m.engine_on)

    @property
    def heatshield_jettisoned(self) -> bool:
        return bool(self._m.heatshield_jettisoned)

    @property
    def heatshield_skin_temp_c(self) -> float:
        return float(self._m.heatshield_skin_temp_c)

    @property
    def drogue_deployed(self) -> bool:
        return bool(self._m.drogue_deployed)

    @property
    def main_deployed(self) -> bool:
        return bool(self._m.main_deployed)

    @property
    def chute_jettisoned(self) -> bool:
        return bool(self._m.chute_jettisoned)

    @property
    def can_heatshield_jettison(self) -> bool:
        return bool(self._m.can_heatshield_jettison)

    @property
    def can_drogue(self) -> bool:
        return bool(self._m.can_drogue)

    @property
    def can_main(self) -> bool:
        return bool(self._m.can_main)

    @property
    def can_chute_jettison(self) -> bool:
        return bool(self._m.can_chute_jettison)

    @property
    def parachute_jettison_max_alt_m(self) -> float:
        return float(self._m.chute_jettison_max_alt_m)

    @property
    def fuel_kg(self) -> float:
        return float(self._m.fuel_kg)

    @property
    def g_load(self) -> float:
        return float(self._m.g_load)

    @property
    def atm_pressure_bar(self) -> float:
        return float(self._m.atm_pressure_bar)

    @property
    def internal_temp_c(self) -> float:
        return float(self._m.internal_temp_c)

    @property
    def distance_to_target_m(self) -> float:
        return float(self._m.distance_to_target_m)

    def stat(self, key: str, default: Any = 0.0) -> Any:
        return self._stats.get(key, default)


class AutopilotActions:
    """Actions a flight program may request each tick (queues Command, adjusts UI throttle)."""

    __slots__ = ("_ui", "_c")

    def __init__(self, ui: Any, c: Controller) -> None:
        self._ui = ui
        self._c = c

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, float(x))))

    def request_heatshield_jettison(self) -> None:
        self._c.queue(Command(request_heatshield_jettison=True))

    def request_drogue(self) -> None:
        self._c.queue(Command(request_drogue=True))

    def request_main(self) -> None:
        self._c.queue(Command(request_main=True))

    def request_chute_jettison(self) -> None:
        self._c.queue(Command(request_chute_jettison=True))

    def set_engine(self, on: bool) -> None:
        self._c.queue(Command(engine_on=bool(on)))

    def set_throttle_slider(self, value_0_1: float) -> None:
        self._ui.throttle_slider.value = float(self.clamp(value_0_1, 0.0, 1.0))

    def sleep(self, seconds: Any) -> None:
        t = float(self._c.model.time_s) + float(_sleep_duration_int_s(seconds))
        raise FlightProgramSleep(t)


class DryRunAutopilotActions:
    """Runs tick() without queuing commands or touching the UI (post-save validation)."""

    __slots__ = ()

    @staticmethod
    def clamp(x: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, float(x))))

    def request_heatshield_jettison(self) -> None:
        pass

    def request_drogue(self) -> None:
        pass

    def request_main(self) -> None:
        pass

    def request_chute_jettison(self) -> None:
        pass

    def set_engine(self, on: bool) -> None:
        pass

    def set_throttle_slider(self, value_0_1: float) -> None:
        pass

    def sleep(self, _seconds: Any) -> None:
        pass


def validate_flight_program_tick(
    tick: Callable[..., None],
    model: PhysicsModel,
    stats: Optional[dict[str, Any]],
    prog_globals: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """
    Execute one tick(sim, ap) with read-only sim and no-op ap.
    sleep() is a no-op during validation.
    Returns an error string if tick raises, else None.
    """
    sim = SimView(model, stats)
    ap = DryRunAutopilotActions()
    g = prog_globals
    old_sleep: Any = None
    if g is not None:
        old_sleep = g.get("sleep")
        g["sleep"] = lambda _s: None
    try:
        tick(sim, ap)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        if g is not None:
            g["sleep"] = old_sleep
    return None


class FlightProgramRunner:
    @staticmethod
    def run(ui: Any, c: Controller) -> None:
        tick = getattr(ui, "flight_program_tick", None)
        if tick is None:
            return
        su = getattr(ui, "flight_program_sleep_until_s", None)
        if su is not None:
            if float(c.model.time_s) < float(su):
                return
            ui.flight_program_sleep_until_s = None
        sim = SimView(c.model, getattr(ui, "stats", None))
        ap = AutopilotActions(ui, c)
        g = getattr(ui, "flight_program_globals", None)
        if g is not None:

            def sleep(seconds: Any) -> None:
                t = float(c.model.time_s) + float(_sleep_duration_int_s(seconds))
                raise FlightProgramSleep(t)

            g["sleep"] = sleep
        try:
            tick(sim, ap)
        except FlightProgramSleep as e:
            if e.until_s >= 0.0:
                ui.flight_program_sleep_until_s = e.until_s
        except Exception as e:
            ui._on_flight_program_runtime_error(f"{type(e).__name__}: {e}")
