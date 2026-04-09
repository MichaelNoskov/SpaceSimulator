from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PerMonitorV2
except:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # System

import pygame

from control.controller import Controller
from digital_twin.model import PhysicsModel, SimResult
from flight_program import DEFAULT_SCRIPT, compile_flight_program
from ui import UI
from render import Renderer


@dataclass(frozen=True)
class AppConfig:
    width: int = 2560
    height: int = 1440
    title: str = "Titan Landing Simulator"
    dt_phys: float = 1.0 / 60.0
    max_frame_time: float = 0.25
    fullscreen: bool = True


def _make_display(size: tuple[int, int], fullscreen: bool) -> pygame.Surface:
    if fullscreen:
        info = pygame.display.Info()
        flags = pygame.FULLSCREEN | pygame.SCALED
        return pygame.display.set_mode((info.current_w, info.current_h), flags)
    return pygame.display.set_mode(size)


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        # PyInstaller onefile: bundled data lives under sys._MEIPASS
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass is not None:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _migrate_ui_state(prev: UI, new: UI) -> None:
    new.lang = prev.lang
    new.auto_mode = prev.auto_mode
    new.time_scale_slider.value = prev.time_scale_slider.value
    new.throttle_slider.value = prev.throttle_slider.value
    new.sim_paused = prev.sim_paused
    new.show_help = prev.show_help
    new.esc_menu_open = prev.esc_menu_open
    new.mission_report_open = prev.mission_report_open
    new.flight_program_source_saved = prev.flight_program_source_saved
    tick, _err, g = compile_flight_program(new.flight_program_source_saved)
    if tick is None:
        tick, _, g = compile_flight_program(DEFAULT_SCRIPT)
    new.flight_program_tick = tick
    new.flight_program_globals = g
    new.flight_program_sleep_until_s = getattr(prev, "flight_program_sleep_until_s", None)
    new.flight_program_editor_open = prev.flight_program_editor_open
    new._fp_lines = list(getattr(prev, "_fp_lines", [""]))
    new._fp_cy = int(getattr(prev, "_fp_cy", 0))
    new._fp_cx = int(getattr(prev, "_fp_cx", 0))
    new._fp_scroll = int(getattr(prev, "_fp_scroll", 0))
    new._fp_hint_scroll = int(getattr(prev, "_fp_hint_scroll", 0))
    new._fp_sel_mark = getattr(prev, "_fp_sel_mark", None)
    new._fp_drag_mark = getattr(prev, "_fp_drag_mark", None)
    new._fp_drag_selecting = bool(getattr(prev, "_fp_drag_selecting", False))
    new._fp_internal_clipboard = getattr(prev, "_fp_internal_clipboard", "")
    new._fp_compile_error = getattr(prev, "_fp_compile_error", None)
    new._toast_custom = getattr(prev, "_toast_custom", None)
    new._toast_until_ms = getattr(prev, "_toast_until_ms", 0)
    new._toast_key = getattr(prev, "_toast_key", "lever_denied")
    new._fp_text_input_started = False
    if new.flight_program_editor_open:
        new._fp_start_text_input()


def main() -> int:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pygame.init()
    pygame.display.set_caption(AppConfig.title)

    config = AppConfig()
    fullscreen = config.fullscreen
    screen = _make_display((config.width, config.height), fullscreen=fullscreen)
    clock = pygame.time.Clock()

    model = PhysicsModel(resource_root=_resource_root())
    controller = Controller(model)
    ui = UI(rect=screen.get_rect())
    renderer = Renderer(screen_size=screen.get_size())
    ui.controller = controller  # type: ignore[attr-defined]

    accumulator = 0.0
    paused = False
    running = True

    while running:
        # tick(0): measure dt without forcing ~60 FPS. With time warp, heavy and light frames
        # alternate; tick(60) would add artificial ~16 ms stalls and stutter.
        frame_dt = clock.tick(0) / 1000.0
        frame_dt = min(frame_dt, config.max_frame_time)
        frame_dt = max(frame_dt, 1e-6)
        ts = ui.time_scale()
        accumulator += frame_dt * ts
        # Cap accumulated sim time per frame to avoid thousands of step() calls and FPS drops.
        max_accum = min(36.0, 9.0 + 0.022 * ts)
        accumulator = min(accumulator, max_accum)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.scancode == pygame.KSCAN_F11:
                fullscreen = not fullscreen
                screen = _make_display((config.width, config.height), fullscreen=fullscreen)
                prev_ui = ui
                prev_renderer = renderer
                ui = UI(rect=screen.get_rect())
                _migrate_ui_state(prev_ui, ui)
                renderer = Renderer(screen_size=screen.get_size())
                renderer.migrate_from(prev_renderer)
                ui.controller = controller  # type: ignore[attr-defined]
            elif event.type == pygame.KEYDOWN and ui.handle_keydown(event, controller):
                pass
            elif renderer.handle_orbit_event(event, model):
                pass
            else:
                ui.handle_event(event, controller)

        # Simulation pause is the Esc menu or the flight program editor.
        paused = ui.esc_menu_open or ui.flight_program_editor_open

        if ui.consume_quit_request():
            running = False

        if ui.consume_restart_request():
            model.reset()
            accumulator = 0.0

        if paused:
            ui.apply_continuous_controls(controller)
            controller.consume_and_apply()

        if not paused:
            # Large time warp accumulates many sim seconds per frame: coarser dt at altitude,
            # slightly larger dt near ground when behind, else only dt_phys and hundreds of
            # iterations. Final meters use small dt.
            max_phys_iters = 2000
            phys_iters = 0
            dt0 = config.dt_phys
            acc_eps = 1e-9
            while accumulator > acc_eps and phys_iters < max_phys_iters:
                ui.apply_continuous_controls(controller)
                controller.consume_and_apply()
                sim_running = model.result == SimResult.RUNNING
                backlog = accumulator > 0.12
                h = float(model.altitude_m)
                acc = float(accumulator)
                if sim_running and backlog:
                    if h > 3000.0:
                        dt = min(0.35, acc)
                    elif h > 400.0:
                        dt = min(0.14, acc)
                    elif h > 45.0:
                        dt = min(0.055, acc)
                    elif h > 12.0:
                        # Low but not final approach: without this, thousands of 1/60 steps.
                        dt = min(0.022, acc)
                    elif h > 2.5:
                        dt = min(0.012, acc)
                    else:
                        dt = min(dt0, acc)
                else:
                    dt = min(dt0, acc)
                if dt <= acc_eps:
                    break
                model.step(dt)
                accumulator -= dt
                phys_iters += 1
                if model.result != SimResult.RUNNING:
                    accumulator = 0.0
                    break

        ui.set_paused(paused)
        ui.sync_from_twin(controller, None)

        renderer.draw(screen, model, ui, controller, frame_dt=frame_dt)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
