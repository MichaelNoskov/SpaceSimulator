from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import numpy as np
import pygame
import math

from control.commands import Command
from control.controller import Controller
from digital_twin.model import SimResult


I18N = {
    "RU": {
        "title": "Симулятор посадки на Титан",
        "h": "Высота",
        "v_vert": "Верт. скорость",
        "v_hor": "Гор. скорость",
        "g": "Перегрузка",
        "t_ext": "Т наружная",
        "t_int": "Т внутр.",
        "p": "Давление",
        "fuel": "Топливо",
        "surface": "Поверхность",
        "dist": "До цели",
        "surface_land": "суша",
        "surface_lake": "жидкость",
        "bar_unit": "бар",
        "heatshield": "Сброс теплозащиты",
        "drogue": "Тормозной парашют",
        "main": "Основной парашют",
        "jettison": "Сброс парашюта",
        "engine": "Двигатель",
        "engine_on": "ВКЛ",
        "engine_off": "ВЫКЛ",
        "auto": "Авто",
        "manual": "Ручной",
        "start": "Старт",
        "restart": "Рестарт",
        "pause": "Пауза (и параметры)",
        "paused_banner": "ПАУЗА",
        "lang": "RU/EN",
        "target": "Клик по карте — цель",
        "success": "УСПЕХ",
        "failure": "ПРОВАЛ",
        "csv_log_on": "CSV LOG: ВКЛ",
        "csv_log_off": "CSV LOG: ВЫКЛ",
        "lever_denied": "Сейчас недоступно",
        "help_title": "Управление",
        "help_body": (
            "Клик по мини-карте — выбор точки посадки.\n"
            "Рычаги: зелёный индикатор — можно включить.\n"
            "Авто: последовательность парашютов и снижение у поверхности.\n"
            "Esc / Пробел — меню паузы (продолжить, авто/ручной, язык, рестарт, CSV, выход). R — рестарт, A — авто/ручной, F1 — эта справка.\n"
            "+/− — скорость времени, F11 — полный экран.\n"
            "После завершения полёта — «Досье миссии»: чертежи телеметрии и синтетический снимок площадки."
        ),
        "esc_resume": "Продолжить",
        "esc_quit": "Выйти",
        "esc_title": "Меню",
        "post_landing_plots": "Телеметрия полёта",
        "plot_h_km": "Высота, км",
        "plot_v_vert": "Верт. скорость, м/с",
        "plot_g": "Перегрузка, g",
        "mission_dossier_btn": "Досье миссии",
        "mission_close": "Закрыть",
        "mission_new_flight": "Новый полёт",
        "mission_blueprint_title": "Пакет документации посадки",
        "mission_photo_label": "Кадр площадки (синтез)",
        "mission_no_plots": "Недостаточно точек телеметрии",
        "mission_plot_legend": "События: HS — теплозащита, DR — тормоз, MN — основной, CJ — сброс, EN — двигатель (▼ выкл)",
        "mission_axis_t": "Время полёта",
    },
    "EN": {
        "title": "Titan Landing Simulator",
        "h": "Altitude",
        "v_vert": "Vertical spd",
        "v_hor": "Horizontal spd",
        "g": "G-load",
        "t_ext": "Temp ext",
        "t_int": "Temp int",
        "p": "Pressure",
        "fuel": "Fuel",
        "surface": "Surface",
        "dist": "To target",
        "surface_land": "land",
        "surface_lake": "liquid",
        "bar_unit": "bar",
        "heatshield": "Jettison heatshield",
        "drogue": "Drogue chute",
        "main": "Main chute",
        "jettison": "Jettison chute",
        "engine": "Engine",
        "engine_on": "ON",
        "engine_off": "OFF",
        "auto": "Auto",
        "manual": "Manual",
        "start": "Start",
        "restart": "Restart",
        "pause": "Pause",
        "paused_banner": "PAUSED",
        "lang": "RU/EN",
        "target": "Click map to set target",
        "success": "SUCCESS",
        "failure": "FAILURE",
        "csv_log_on": "CSV LOG: ON",
        "csv_log_off": "CSV LOG: OFF",
        "lever_denied": "Not available now",
        "help_title": "Controls",
        "help_body": (
            "Click the minimap to set a landing target.\n"
            "Levers: green lamp means the action is allowed.\n"
            "Auto: chute sequence and descent rate near the surface.\n"
            "Esc / Space — pause menu (resume, auto/manual, lang, restart, CSV, quit). R — restart, A — auto/manual, F1 — this help.\n"
            "+/− — time speed, F11 — fullscreen.\n"
            "After landing, open «Mission dossier» for telemetry drawings and a synthetic site snapshot."
        ),
        "esc_resume": "Resume",
        "esc_quit": "Quit",
        "esc_title": "Menu",
        "post_landing_plots": "Flight telemetry",
        "plot_h_km": "Altitude, km",
        "plot_v_vert": "Vertical speed, m/s",
        "plot_g": "G-load, g",
        "mission_dossier_btn": "Mission dossier",
        "mission_close": "Close",
        "mission_new_flight": "New flight",
        "mission_blueprint_title": "Landing documentation pack",
        "mission_photo_label": "Site frame (synthetic)",
        "mission_no_plots": "Not enough telemetry samples",
        "mission_plot_legend": "Events: HS heatshield, DR drogue, MN main, CJ jettison, EN engine (▼ off)",
        "mission_axis_t": "Mission time",
    },
}


class _MissionReportGeom(NamedTuple):
    panel: pygame.Rect
    close_btn: pygame.Rect
    restart_btn: pygame.Rect
    photo: pygame.Rect
    plot0: pygame.Rect
    plot1: pygame.Rect
    plot2: pygame.Rect


@dataclass
class Lever:
    rect: pygame.Rect
    label_key: str
    # Returns (is_on, can_toggle_on, can_toggle_off)
    state: Callable[[Controller], tuple[bool, bool, bool]]
    on_toggle_on: Callable[[Controller], bool]
    on_toggle_off: Callable[[Controller], bool]

    last_action_ok: bool = True
    anim: float = 0.0  # 0..1 where 1 means ON position (top)
    _last_ms: int = 0

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, text: str, c: Controller) -> None:
        is_on, can_on, can_off = self.state(c)
        enabled = can_off if is_on else can_on
        target = 1.0 if is_on else 0.0

        now_ms = pygame.time.get_ticks()
        if self._last_ms == 0:
            self._last_ms = now_ms
            self.anim = target
        dt = max(0.0, (now_ms - self._last_ms) / 1000.0)
        self._last_ms = now_ms

        # Animate towards target
        speed = 8.0  # 1/s
        self.anim += (target - self.anim) * float(np.clip(speed * dt, 0.0, 1.0))
        a = float(np.clip(self.anim, 0.0, 1.0))
        # smoothstep for nicer motion
        a = a * a * (3 - 2 * a)

        track = self.rect
        # base plate (more detailed, dial-like styling)
        pygame.draw.rect(surf, (18, 18, 22), track, border_radius=12)
        pygame.draw.rect(surf, (140, 140, 155), track, width=2, border_radius=12)
        pygame.draw.rect(surf, (60, 60, 70), track.inflate(-6, -6), width=1, border_radius=10)

        # lamp (ready indicator)
        lamp_c = (track.right - 16, track.top + 16)
        # Lamp indicates whether the lever can be actuated right now (conditions satisfied).
        lamp_col = (80, 220, 120) if enabled else (230, 80, 80)
        pygame.draw.circle(surf, (20, 20, 22), lamp_c, 9)
        pygame.draw.circle(surf, (90, 90, 105), lamp_c, 9, 2)
        pygame.draw.circle(surf, lamp_col, lamp_c, 6)
        pygame.draw.circle(surf, (10, 10, 10), lamp_c, 6, 1)

        # slot + lever (stylized)
        slot = track.inflate(-28, -30)
        slot.top += 18
        pygame.draw.rect(surf, (12, 12, 14), slot, border_radius=10)
        pygame.draw.rect(surf, (70, 70, 80), slot, width=2, border_radius=10)

        # ON/OFF marks
        on_txt = font.render("ON", True, (200, 200, 210))
        off_txt = font.render("OFF", True, (200, 200, 210))
        surf.blit(on_txt, (slot.right - on_txt.get_width() - 6, slot.top - 16))
        surf.blit(off_txt, (slot.right - off_txt.get_width() - 6, slot.bottom + 2))

        # lever pivot point and angle animation
        pivot = (slot.left + 18, slot.centery)
        ang = lerp(-0.55, 0.55, a)  # radians-ish range
        length = slot.width - 26
        tip = (pivot[0] + int(math.cos(ang) * length), pivot[1] - int(math.sin(ang) * length))
        # shaft
        shaft_col = (210, 210, 220) if enabled else (110, 110, 120)
        pygame.draw.line(surf, shaft_col, pivot, tip, 5)
        pygame.draw.line(surf, (40, 40, 45), pivot, tip, 1)
        # pivot cap
        pygame.draw.circle(surf, (25, 25, 28), pivot, 10)
        pygame.draw.circle(surf, (140, 140, 155), pivot, 10, 2)
        pygame.draw.circle(surf, (200, 200, 210), pivot, 3)

        # handle knob at tip
        knob_r = 9
        knob_col = (240, 140, 60) if enabled else (90, 90, 100)
        pygame.draw.circle(surf, knob_col, tip, knob_r)
        pygame.draw.circle(surf, (20, 20, 22), tip, knob_r, 2)
        # small highlight
        pygame.draw.circle(surf, (255, 210, 160), (tip[0] - 3, tip[1] - 3), 3)

        # label
        lbl = font.render(text, True, (235, 235, 235))
        surf.blit(lbl, (track.left + 10, track.top + 8))

    def handle_event(self, event: pygame.event.Event, c: Controller) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.rect.collidepoint(event.pos):
                return False
            is_on, can_on, can_off = self.state(c)
            if is_on:
                if not can_off:
                    self.last_action_ok = False
                    return True
                self.last_action_ok = bool(self.on_toggle_off(c))
                return True
            else:
                if not can_on:
                    self.last_action_ok = False
                    return True
                self.last_action_ok = bool(self.on_toggle_on(c))
                return True
        return False


@dataclass
class Slider:
    rect: pygame.Rect
    value: float = 0.0  # 0..1
    dragging: bool = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
            return True
        return False

    def _set_from_x(self, x: int) -> None:
        t = (x - self.rect.left) / max(1, self.rect.width)
        self.value = float(np.clip(t, 0.0, 1.0))

    def draw(self, surf: pygame.Surface) -> None:
        pygame.draw.rect(surf, (30, 30, 34), self.rect, border_radius=8)
        pygame.draw.rect(surf, (120, 120, 135), self.rect, width=2, border_radius=8)
        knob_x = int(self.rect.left + self.value * self.rect.width)
        knob = pygame.Rect(0, 0, 14, self.rect.height + 6)
        knob.center = (knob_x, self.rect.centery)
        pygame.draw.rect(surf, (240, 140, 60), knob, border_radius=6)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * float(t)


def slider_to_time_scale(v01: float) -> float:
    # Map slider 0..1 to 0.1..1000x (log-like)
    v01 = float(np.clip(v01, 0.0, 1.0))
    exp = lerp(-1.0, math.log10(1000.0), v01)  # 10^-1 .. 1000
    return float(10.0**exp)


class UI:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.lang = "RU"
        self.ui_scale = 2.0

        pygame.font.init()
        self.font = pygame.font.SysFont("DejaVu Sans", int(18 * self.ui_scale))
        self.font_small = pygame.font.SysFont("DejaVu Sans", int(14 * self.ui_scale))
        self.font_big = pygame.font.SysFont("DejaVu Sans", int(28 * self.ui_scale), bold=True)
        self.font_mono = pygame.font.SysFont("DejaVu Sans Mono", int(22 * self.ui_scale), bold=True)

        self._restart_requested = False

        self.auto_mode = False
        self._last_log_path: Optional[str] = None

        self.sim_paused: bool = False
        self.show_help: bool = False
        self.esc_menu_open: bool = False
        self.mission_report_open: bool = False
        self._quit_requested: bool = False
        self._toast_until_ms: int = 0
        self._toast_key: str = "lever_denied"
        self.stats = {}
        # Set by main; used by renderer-less overlay calls.
        self.controller: Optional[Controller] = None
        self._text_cache: dict[tuple, pygame.Surface] = {}

        w, h = rect.size
        # Draw everything as overlay on the scene (no separate mid/ctrl strips).
        self.visual_rect = pygame.Rect(0, 0, w, h)
        self.mid_rect = pygame.Rect(0, 0, 0, 0)
        self.ctrl_rect = pygame.Rect(0, 0, 0, 0)

        # --- layout constants ---
        # Extra margin so right/bottom overlays never hug screen edge.
        margin = int(44 * self.ui_scale)
        gap = int(12 * self.ui_scale)

        # --- top-left (visual): minimap + telemetry + time ---
        map_size = int(220 * self.ui_scale)
        self.map_rect = pygame.Rect(self.visual_rect.left + margin, self.visual_rect.top + margin, map_size, map_size)
        hint_gap = int(6 * self.ui_scale)
        self._map_stack_y0 = self.map_rect.bottom + hint_gap
        hint_line_h = int(15 * self.ui_scale)
        telemetry_h = int(52 * self.ui_scale)
        self.telemetry_rect = pygame.Rect(
            self.map_rect.left,
            self._map_stack_y0 + hint_line_h + int(4 * self.ui_scale),
            self.map_rect.width,
            telemetry_h,
        )
        self.time_scale_slider = Slider(
            pygame.Rect(
                self.map_rect.left,
                self.telemetry_rect.bottom + int(8 * self.ui_scale),
                self.map_rect.width,
                int(18 * self.ui_scale),
            ),
            value=0.5,
        )

        # --- top-right (visual): pause only; rest is in the pause menu ---
        btn_w = int(220 * self.ui_scale)
        btn_h = int(36 * self.ui_scale)
        top_right_x = self.visual_rect.right - margin - btn_w
        top_right_y = self.visual_rect.top + margin
        by_gap = int(10 * self.ui_scale)
        self.pause_rect = pygame.Rect(top_right_x, top_right_y + 0 * (btn_h + by_gap), btn_w, btn_h)
        self.mission_report_btn_rect = pygame.Rect(top_right_x, top_right_y + btn_h + by_gap, btn_w, btn_h)

        # --- bottom-right (visual): flight controls (levers + engine + throttle) ---
        flight_w = int(np.clip(self.visual_rect.width * 0.34, int(360 * self.ui_scale), int(520 * self.ui_scale)))
        # Fit levers vertically in available space.
        lever_rows = 2
        lever_h = int(64 * self.ui_scale)
        lever_h_min = int(48 * self.ui_scale)
        required_h = lever_rows * lever_h + (lever_rows - 1) * gap + int(10 * self.ui_scale) + btn_h + int(10 * self.ui_scale) + int(18 * self.ui_scale) + int(16 * self.ui_scale)
        avail_h = int(self.visual_rect.height * 0.28)
        if required_h > avail_h:
            # shrink lever height to fit
            lever_h = int(
                np.clip(
                    (avail_h - ((lever_rows - 1) * gap + int(10 * self.ui_scale) + btn_h + int(10 * self.ui_scale) + int(18 * self.ui_scale) + int(16 * self.ui_scale)))
                    / lever_rows,
                    lever_h_min,
                    int(64 * self.ui_scale),
                )
            )
        flight_h = lever_rows * lever_h + (lever_rows - 1) * gap + int(10 * self.ui_scale) + btn_h + int(10 * self.ui_scale) + int(18 * self.ui_scale) + int(16 * self.ui_scale)
        self.flight_rect = pygame.Rect(
            self.visual_rect.right - margin - flight_w,
            self.visual_rect.bottom - margin - flight_h,
            flight_w,
            flight_h,
        )

        # --- bottom-left (visual): instruments ---
        # Keep instruments comfortably tall (avoid squashing gauges).
        instr_h = int(250 * self.ui_scale)
        instr_w = int(np.clip(self.visual_rect.width * 0.44, int(560 * self.ui_scale), int(820 * self.ui_scale)))
        self.instr_rect = pygame.Rect(
            self.visual_rect.left + margin,
            self.visual_rect.bottom - margin - instr_h,
            instr_w,
            instr_h,
        )

        # lever_h computed above to ensure fit
        lever_w = int((self.flight_rect.width - gap) * 0.55)
        lever_w = int(np.clip(lever_w, int(210 * self.ui_scale), int(280 * self.ui_scale)))
        lever_col2_x = self.flight_rect.left + lever_w + gap
        lever_col1_x = self.flight_rect.left
        lever_y0 = self.flight_rect.top
        self.levers: list[Lever] = [
            Lever(
                pygame.Rect(lever_col1_x, lever_y0 + 0 * (lever_h + gap), lever_w, lever_h),
                "heatshield",
                state=lambda c: (
                    bool(c.model.heatshield_jettisoned),
                    c.can_heatshield_jettison(),
                    False,
                ),
                on_toggle_on=lambda c: (c.queue(Command(request_heatshield_jettison=True)) or True),
                on_toggle_off=lambda c: False,
            ),
            Lever(
                pygame.Rect(lever_col1_x, lever_y0 + 1 * (lever_h + gap), lever_w, lever_h),
                "drogue",
                state=lambda c: (
                    bool(c.model.drogue_deployed),
                    c.can_drogue(),
                    False,
                ),
                on_toggle_on=lambda c: (c.queue(Command(request_drogue=True)) or True),
                on_toggle_off=lambda c: False,
            ),
            Lever(
                pygame.Rect(lever_col2_x, lever_y0 + 0 * (lever_h + gap), lever_w, lever_h),
                "main",
                state=lambda c: (
                    bool(c.model.main_deployed),
                    c.can_main(),
                    False,
                ),
                on_toggle_on=lambda c: (c.queue(Command(request_main=True)) or True),
                on_toggle_off=lambda c: False,
            ),
            Lever(
                pygame.Rect(lever_col2_x, lever_y0 + 1 * (lever_h + gap), lever_w, lever_h),
                "jettison",
                state=lambda c: (
                    bool(c.model.chute_jettisoned),
                    c.can_chute_jettison(),
                    False,
                ),
                on_toggle_on=lambda c: (c.queue(Command(request_chute_jettison=True)) or True),
                on_toggle_off=lambda c: False,
            ),
        ]

        engine_w = self.flight_rect.width
        engine_y = lever_y0 + 2 * (lever_h + gap) + 10
        self.engine_toggle_rect = pygame.Rect(self.flight_rect.left, engine_y, engine_w, btn_h)
        self.throttle_slider = Slider(
            pygame.Rect(self.flight_rect.left, self.engine_toggle_rect.bottom + 10, engine_w - 46, 18),
            value=0.0,
        )

        # If for any reason flight content goes off-screen, nudge it upward.
        overflow = (self.flight_rect.bottom + margin) - self.visual_rect.bottom
        if overflow > 0:
            dy = overflow
            self.flight_rect.y -= dy
            for lv in self.levers:
                lv.rect.y -= dy
            self.engine_toggle_rect.y -= dy
            self.throttle_slider.rect.y -= dy

        # Also ensure it doesn't hug / go beyond the right edge.
        overflow_r = (self.flight_rect.right + margin) - self.visual_rect.right
        if overflow_r > 0:
            dx = overflow_r
            self.flight_rect.x -= dx
            for lv in self.levers:
                lv.rect.x -= dx
            self.engine_toggle_rect.x -= dx
            self.throttle_slider.rect.x -= dx

    def t(self, key: str) -> str:
        return I18N[self.lang][key]

    def _cached_render(self, font: pygame.font.Font, text: str, color: tuple[int, ...]) -> pygame.Surface:
        key = (id(font), text, color)
        surf = self._text_cache.get(key)
        if surf is not None:
            return surf
        surf = font.render(text, True, color)
        if len(self._text_cache) > 512:
            self._text_cache.clear()
        self._text_cache[key] = surf
        return surf

    def set_paused(self, paused: bool) -> None:
        self.sim_paused = bool(paused)

    def bump_time_slider(self, direction: int) -> None:
        """direction +1 / −1 nudges the time slider (logical scale)."""
        self.time_scale_slider.value = float(
            np.clip(self.time_scale_slider.value + 0.04 * float(direction), 0.0, 1.0)
        )

    def consume_quit_request(self) -> bool:
        v = self._quit_requested
        self._quit_requested = False
        return v

    def _toggle_esc_menu(self) -> None:
        """Simulation pause matches this menu (see main: paused = esc_menu_open)."""
        self.esc_menu_open = not self.esc_menu_open

    def _request_restart(self) -> None:
        """New flight from the start, unpaused (menu closes)."""
        self._restart_requested = True
        self.esc_menu_open = False
        self.mission_report_open = False

    def handle_keydown(self, event: pygame.event.Event, c: Controller) -> bool:
        if event.type != pygame.KEYDOWN:
            return False
        key = event.key
        if key == pygame.K_ESCAPE:
            if self.show_help:
                self.show_help = False
                return True
            if self.mission_report_open:
                self.mission_report_open = False
                return True
            self._toggle_esc_menu()
            return True
        if key == pygame.K_F1:
            self.show_help = not self.show_help
            return True
        if key in (pygame.K_SPACE, pygame.K_PAUSE):
            if self.mission_report_open:
                self.mission_report_open = False
                return True
            self._toggle_esc_menu()
            return True
        if key == pygame.K_r:
            self._request_restart()
            return True
        if key == pygame.K_a:
            self.auto_mode = not self.auto_mode
            return True
        if key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.bump_time_slider(1)
            return True
        if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.bump_time_slider(-1)
            return True
        return False

    def _surface_display(self, raw: str) -> str:
        if raw == "lake":
            return self.t("surface_lake")
        return self.t("surface_land")

    def _esc_menu_layout_full(
        self,
    ) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect, pygame.Rect]:
        """Order: resume, auto/manual, language, restart, CSV, quit."""
        cx = self.visual_rect.centerx
        bw, bh = int(240 * self.ui_scale), int(44 * self.ui_scale)
        gap = int(10 * self.ui_scale)
        n = 6
        total_h = n * bh + (n - 1) * gap
        y0 = self.visual_rect.centery - total_h // 2 + int(28 * self.ui_scale)
        rects: list[pygame.Rect] = []
        for i in range(n):
            r = pygame.Rect(0, 0, bw, bh)
            r.centerx = cx
            r.top = y0 + i * (bh + gap)
            rects.append(r)
        return (rects[0], rects[1], rects[2], rects[3], rects[4], rects[5])

    def handle_event(self, event: pygame.event.Event, c: Controller) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.show_help:
                self.show_help = False
                return
            if self.esc_menu_open:
                r_res, r_mode, r_lang, r_rst, r_log, r_quit = self._esc_menu_layout_full()
                if r_res.collidepoint(event.pos):
                    self.esc_menu_open = False
                    return
                if r_mode.collidepoint(event.pos):
                    self.auto_mode = not self.auto_mode
                    return
                if r_lang.collidepoint(event.pos):
                    self.lang = "EN" if self.lang == "RU" else "RU"
                    return
                if r_rst.collidepoint(event.pos):
                    self._request_restart()
                    return
                if r_log.collidepoint(event.pos):
                    c.queue(Command(request_toggle_csv_logging=True))
                    return
                if r_quit.collidepoint(event.pos):
                    self._quit_requested = True
                    self.esc_menu_open = False
                    return
                return

        if self.throttle_slider.handle_event(event):
            return
        if self.time_scale_slider.handle_event(event):
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if c.model.result != SimResult.RUNNING:
                if self.mission_report_open:
                    g = self._mission_report_geometry()
                    if g.close_btn.collidepoint(event.pos):
                        self.mission_report_open = False
                        return
                    if g.restart_btn.collidepoint(event.pos):
                        self._request_restart()
                        return
                    if not g.panel.collidepoint(event.pos):
                        self.mission_report_open = False
                        return
                    return
                if self.mission_report_btn_rect.collidepoint(event.pos):
                    self.mission_report_open = True
                    return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.map_rect.collidepoint(event.pos):
                mx, my = event.pos
                u = (mx - self.map_rect.left) / max(1, self.map_rect.width)
                v = (my - self.map_rect.top) / max(1, self.map_rect.height)
                # Infinite minimap: probe stays centered; click sets a world target relative to probe.
                h = float(c.model.altitude_m)
                # Must match _draw_minimap in render.py (Z axis: forward = up on map).
                mpp = float(np.clip(8.0 + 0.004 * h, 8.0, 420.0))
                mpp = float(round(mpp / 2.0) * 2.0) if mpp < 20.0 else float(round(mpp / 5.0) * 5.0)
                dx_m = (u - 0.5) * self.map_rect.width * mpp
                dz_m = (0.5 - v) * self.map_rect.height * mpp
                c.queue(Command(set_target_world=(c.model.pos_x_m + dx_m, c.model.pos_z_m + dz_m)))
                return

            if self.engine_toggle_rect.collidepoint(event.pos):
                c.queue(Command(engine_on=not c.model.engine_on))
                return

            if self.pause_rect.collidepoint(event.pos):
                self._toggle_esc_menu()
                return

        for lv in self.levers:
            if lv.handle_event(event, c):
                if not lv.last_action_ok:
                    self._toast_until_ms = pygame.time.get_ticks() + 1800
                    self._toast_key = "lever_denied"
                return

    def apply_continuous_controls(self, c: Controller) -> None:
        c.queue(Command(throttle_0_1=self.throttle_slider.value if c.model.engine_on else 0.0))

        if self.auto_mode and c.model.result == SimResult.RUNNING:
            if c.model.can_heatshield_jettison:
                c.queue(Command(request_heatshield_jettison=True))
            if c.model.can_drogue and (c.model.altitude_m > 180_000.0):
                c.queue(Command(request_drogue=True))
            if c.model.can_main and (c.model.altitude_m < 160_000.0):
                c.queue(Command(request_main=True))
            if c.model.can_chute_jettison and (c.model.altitude_m < 2_000.0):
                c.queue(Command(request_chute_jettison=True))

            if c.model.altitude_m < 2_000.0 and c.model.chute_jettisoned:
                c.queue(Command(engine_on=True))
                target_v = -20.0 if c.model.altitude_m > 200.0 else -4.0
                error = target_v - c.model.vertical_speed_mps
                cmd = float(np.clip(0.02 * error, 0.0, 1.0))
                self.throttle_slider.value = cmd

    def time_scale(self) -> float:
        return slider_to_time_scale(self.time_scale_slider.value)

    def sync_from_twin(self, c: Controller, _tel) -> None:
        # Telemetry is no longer used after getter-model refactor.
        v_h = float(c.model.horizontal_speed_mps)
        self.stats = {
            "t_s": float(c.model.time_s),
            "h_km": c.model.altitude_m / 1000.0,
            "v_vert": c.model.vertical_speed_mps,
            "v_hor": v_h,
            "g": float(c.model.g_load),
            "t_ext": float(c.model.atm_temp_ext_c),
            "t_int": float(c.model.internal_temp_c),
            "p": float(c.model.atm_pressure_bar),
            "fuel_pct": 100.0 * (float(c.model.fuel_kg) / 50.0),
            "surface": c.model.surface_type_under_probe.value,
            "dist_m": float(c.model.distance_to_target_m),
        }
        self._last_log_path = c.model.csv_log_path

    def _color_zone(self, t: float) -> tuple[int, int, int]:
        # t in [0..1] => green->yellow->red
        t = float(np.clip(t, 0.0, 1.0))
        if t < 0.5:
            u = t / 0.5
            return (int(80 + (230 - 80) * u), int(220 + (190 - 220) * u), int(120 + (90 - 120) * u))
        u = (t - 0.5) / 0.5
        return (int(230 + (230 - 230) * u), int(190 + (80 - 190) * u), int(90 + (80 - 90) * u))

    def _draw_dial(
        self,
        surf: pygame.Surface,
        center: tuple[int, int],
        radius: int,
        label: str,
        value: float,
        unit: str,
        vmin: float,
        vmax: float,
        zones: list[tuple[float, float, tuple[int, int, int]]],
        tick_step: float,
        danger_value: Optional[float] = None,
    ) -> None:
        cx, cy = center
        bg = (24, 24, 28)
        pygame.draw.circle(surf, bg, center, radius)
        pygame.draw.circle(surf, (120, 120, 135), center, radius, width=2)

        # zones arcs (map value range to angles)
        start_ang = math.radians(210)
        end_ang = math.radians(-30)
        span = end_ang - start_ang

        def ang_for(v: float) -> float:
            t = (v - vmin) / max(1e-9, (vmax - vmin))
            t = float(np.clip(t, 0.0, 1.0))
            return start_ang + span * t

        arc_rect = pygame.Rect(0, 0, radius * 2 - 10, radius * 2 - 10)
        arc_rect.center = center
        for z0, z1, col in zones:
            a0 = ang_for(z0)
            a1 = ang_for(z1)
            pygame.draw.arc(surf, col, arc_rect, min(a0, a1), max(a0, a1), 6)

        # ticks
        tick = vmin
        while tick <= vmax + 1e-6:
            a = ang_for(tick)
            x1 = cx + int(math.cos(a) * (radius - 8))
            y1 = cy + int(math.sin(a) * (radius - 8))
            x2 = cx + int(math.cos(a) * (radius - 16))
            y2 = cy + int(math.sin(a) * (radius - 16))
            pygame.draw.line(surf, (180, 180, 190), (x1, y1), (x2, y2), 2)
            tick += tick_step

        # needle
        vv = float(np.clip(value, vmin, vmax))
        a = ang_for(vv)
        nx = cx + int(math.cos(a) * (radius - 22))
        ny = cy + int(math.sin(a) * (radius - 22))
        pygame.draw.line(surf, (240, 140, 60), (cx, cy), (nx, ny), 3)
        pygame.draw.circle(surf, (240, 140, 60), center, 5)

        label_txt = self._cached_render(self.font_small, label, (220, 220, 230))
        surf.blit(label_txt, label_txt.get_rect(center=(cx, cy + radius - 22)))

        value_txt = self._cached_render(self.font, f"{value:.1f} {unit}", (240, 240, 245))
        surf.blit(value_txt, value_txt.get_rect(center=(cx, cy + 8)))

        if danger_value is not None and abs(value) > danger_value:
            warn = self.font_small.render("!", True, (230, 80, 80))
            surf.blit(warn, (cx + radius - 18, cy - radius + 10))

    def _draw_bar_gauge(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        value: float,
        unit: str,
        vmin: float,
        vmax: float,
        zones: list[tuple[float, float, tuple[int, int, int]]],
        ticks: list[float],
        value_fmt: str = "{:.1f}",
    ) -> None:
        pygame.draw.rect(surf, (24, 24, 28), rect, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), rect, width=2, border_radius=10)

        inner = rect.inflate(-12, -26)
        inner.top += 16

        # zone background segments
        for z0, z1, col in zones:
            t0 = (z0 - vmin) / max(1e-9, (vmax - vmin))
            t1 = (z1 - vmin) / max(1e-9, (vmax - vmin))
            x0 = inner.left + int(np.clip(t0, 0.0, 1.0) * inner.width)
            x1 = inner.left + int(np.clip(t1, 0.0, 1.0) * inner.width)
            seg = pygame.Rect(min(x0, x1), inner.top, abs(x1 - x0), inner.height)
            pygame.draw.rect(surf, (*col, 60), seg, border_radius=6)

        # value fill
        t = (value - vmin) / max(1e-9, (vmax - vmin))
        t = float(np.clip(t, 0.0, 1.0))
        fill = pygame.Rect(inner.left, inner.top, int(inner.width * t), inner.height)
        pygame.draw.rect(surf, (240, 140, 60), fill, border_radius=6)

        # ticks
        for tv in ticks:
            tt = (tv - vmin) / max(1e-9, (vmax - vmin))
            x = inner.left + int(np.clip(tt, 0.0, 1.0) * inner.width)
            pygame.draw.line(surf, (180, 180, 190), (x, inner.top - 2), (x, inner.bottom + 2), 1)

        lab = self._cached_render(self.font_small, label, (220, 220, 230))
        surf.blit(lab, (rect.left + 10, rect.top + 8))
        val = self._cached_render(self.font, f"{value_fmt.format(value)} {unit}", (240, 240, 245))
        surf.blit(val, (rect.right - val.get_width() - 10, rect.top + 5))

    def _draw_vertical_bar_gauge(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        value: float,
        unit: str,
        vmin: float,
        vmax: float,
        zones: list[tuple[float, float, tuple[int, int, int]]],
        ticks: list[float],
        value_fmt: str = "{:.1f}",
    ) -> None:
        pygame.draw.rect(surf, (24, 24, 28), rect, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), rect, width=2, border_radius=10)

        inner = rect.inflate(-18, -44)
        inner.top += 30

        # zones background (vertical)
        for z0, z1, col in zones:
            t0 = (z0 - vmin) / max(1e-9, (vmax - vmin))
            t1 = (z1 - vmin) / max(1e-9, (vmax - vmin))
            y0 = inner.bottom - int(np.clip(t0, 0.0, 1.0) * inner.height)
            y1 = inner.bottom - int(np.clip(t1, 0.0, 1.0) * inner.height)
            seg = pygame.Rect(inner.left, min(y0, y1), inner.width, abs(y1 - y0))
            pygame.draw.rect(surf, (*col, 60), seg, border_radius=6)

        # value fill
        t = (value - vmin) / max(1e-9, (vmax - vmin))
        t = float(np.clip(t, 0.0, 1.0))
        fill_h = int(inner.height * t)
        fill = pygame.Rect(inner.left, inner.bottom - fill_h, inner.width, fill_h)
        pygame.draw.rect(surf, (240, 140, 60), fill, border_radius=6)

        # ticks
        for tv in ticks:
            tt = (tv - vmin) / max(1e-9, (vmax - vmin))
            y = inner.bottom - int(np.clip(tt, 0.0, 1.0) * inner.height)
            pygame.draw.line(surf, (180, 180, 190), (inner.left - 2, y), (inner.right + 2, y), 1)

        lab = self._cached_render(self.font_small, label, (220, 220, 230))
        surf.blit(lab, (rect.left + 10, rect.top + 8))
        val = self._cached_render(self.font, f"{value_fmt.format(value)} {unit}", (240, 240, 245))
        surf.blit(val, (rect.left + 10, rect.bottom - val.get_height() - 8))

    def _h_bar_value(self, h_km: float) -> float:
        # Piecewise-linear mapping to stretch low altitudes:
        # 0..20km -> 0..0.55, 20..200km -> 0.55..0.85, 200..1270km -> 0.85..1.0
        h_km = float(max(0.0, h_km))
        if h_km <= 20.0:
            return 0.55 * (h_km / 20.0)
        if h_km <= 200.0:
            return 0.55 + 0.30 * ((h_km - 20.0) / 180.0)
        return 0.85 + 0.15 * (min(h_km, 1270.0) - 200.0) / 1070.0

    def _h_from_bar_value(self, t: float) -> float:
        # Inverse of _h_bar_value(). Input t is 0..1, output km.
        t = float(np.clip(t, 0.0, 1.0))
        if t <= 0.55:
            return 20.0 * (t / 0.55)
        if t <= 0.85:
            return 20.0 + 180.0 * ((t - 0.55) / 0.30)
        return 200.0 + 1070.0 * ((t - 0.85) / 0.15)

    def _titan_sky_color_by_alt(self, h_km: float) -> tuple[int, int, int]:
        # Approximate color progression similar to render._draw_sky palette.
        # 0..2km: milky orange, 2..30km: orange-brown, 30..200km: darker, 200km+: brown/purple.
        def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
            t = float(np.clip(t, 0.0, 1.0))
            return (
                int(a[0] * (1 - t) + b[0] * t),
                int(a[1] * (1 - t) + b[1] * t),
                int(a[2] * (1 - t) + b[2] * t),
            )

        if h_km <= 2.0:
            return blend((255, 175, 110), (235, 140, 70), h_km / 2.0)
        if h_km <= 30.0:
            return blend((235, 140, 70), (160, 90, 70), (h_km - 2.0) / 28.0)
        if h_km <= 200.0:
            return blend((160, 90, 70), (70, 45, 65), (h_km - 30.0) / 170.0)
        return (70, 45, 65)

    def _draw_altitude_tape(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        h_km: float,
    ) -> None:
        # Thick vertical tape with color gradient by altitude and a red marker triangle.
        pygame.draw.rect(surf, (24, 24, 28), rect, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), rect, width=2, border_radius=10)

        inner = rect.inflate(-18, -40)
        inner.top += 26

        # draw gradient segments
        steps = max(24, inner.height // 6)
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 1) / steps
            # map display position to km using inverse of stretched mapping
            km = self._h_from_bar_value(t1)
            col = self._titan_sky_color_by_alt(km)
            y1 = inner.bottom - int(t1 * inner.height)
            y0 = inner.bottom - int(t0 * inner.height)
            pygame.draw.rect(surf, col, pygame.Rect(inner.left, y1, inner.width, max(1, y0 - y1)))

        # marker position uses stretched mapping for better low-alt visibility
        t = self._h_bar_value(h_km)
        y = inner.bottom - int(np.clip(t, 0.0, 1.0) * inner.height)
        tri = [(inner.right + 2, y), (inner.right + 14, y - 8), (inner.right + 14, y + 8)]
        pygame.draw.polygon(surf, (230, 80, 80), tri)

        # ticks (basic)
        for km in [0.0, 2.0, 20.0, 200.0, 1270.0]:
            tt = self._h_bar_value(km)
            yy = inner.bottom - int(np.clip(tt, 0.0, 1.0) * inner.height)
            pygame.draw.line(surf, (180, 180, 190), (inner.left - 2, yy), (inner.right + 2, yy), 1)

        txt = self._cached_render(self.font, f"{h_km:.2f} km", (240, 240, 245))
        chip = pygame.Rect(0, 0, txt.get_width() + 14, txt.get_height() + 10)
        chip.center = inner.center
        pygame.draw.rect(surf, (18, 18, 22), chip, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), chip, width=2, border_radius=10)
        surf.blit(txt, txt.get_rect(center=chip.center))

        lab = self._cached_render(self.font_small, self.t("h"), (220, 220, 230))
        surf.blit(lab, (rect.left + 10, rect.top + 8))

    def _draw_thermometer(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        label: str,
        temp_c: float,
        vmin: float = -220.0,
        vmax: float = 50.0,
    ) -> None:
        pygame.draw.rect(surf, (24, 24, 28), rect, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), rect, width=2, border_radius=10)

        # Use more vertical space for the scale (longer tube).
        inner = rect.inflate(-int(14 * self.ui_scale), -int(40 * self.ui_scale))
        inner.top += int(24 * self.ui_scale)

        # tube + bulb
        tube_w = max(4, inner.width // 6)
        bulb_r = max(5, tube_w // 2 + int(4 * self.ui_scale))
        bulb_c = (inner.centerx, inner.bottom - bulb_r)

        # Make the tube end inside the bulb (continuous connection).
        # Keep overlap for continuity, but don't push the tube too deep into the bulb
        # (so the colored column reads higher).
        tube_h = max(10, (bulb_c[1] - inner.top) + int(0.10 * bulb_r))
        tube = pygame.Rect(0, 0, tube_w, tube_h)
        tube.midtop = (inner.centerx, inner.top)

        pygame.draw.rect(surf, (14, 14, 18), tube, border_radius=7)
        pygame.draw.circle(surf, (14, 14, 18), bulb_c, bulb_r)
        pygame.draw.rect(surf, (80, 80, 95), tube, width=2, border_radius=7)
        pygame.draw.circle(surf, (80, 80, 95), bulb_c, bulb_r, 2)

        # fill
        t = (float(temp_c) - vmin) / max(1e-9, (vmax - vmin))
        t = float(np.clip(t, 0.0, 1.0))
        pad = 2
        fill_h = int((tube.height - 2 * pad - 2) * t)
        fill = pygame.Rect(tube.left + pad, tube.bottom - (pad + 1) - fill_h, max(1, tube.width - 2 * pad), fill_h)

        # color by temperature
        if temp_c < -120:
            col = (80, 170, 255)   # cold
        elif temp_c < -30:
            col = (120, 230, 190)  # ok
        else:
            col = (255, 120, 70)   # hot
        pygame.draw.rect(surf, col, fill, border_radius=6)
        pygame.draw.circle(surf, col, bulb_c, max(3, bulb_r - int(3 * self.ui_scale)))

        # ticks
        for tv in [-200, -180, -150, -120, -90, -60, -30, 0]:
            tt = (tv - vmin) / max(1e-9, (vmax - vmin))
            yy = tube.bottom - (pad + 1) - int(np.clip(tt, 0.0, 1.0) * (tube.height - 2 * pad - 2))
            pygame.draw.line(surf, (180, 180, 190), (tube.right + 4, yy), (tube.right + 8, yy), 1)

        lab = self._cached_render(self.font_small, label, (220, 220, 230))
        surf.blit(lab, (rect.left + 10, rect.top + 8))
        val = self._cached_render(self.font, f"{temp_c:.1f} C", (240, 240, 245))
        surf.blit(val, (rect.left + 10, rect.bottom - val.get_height() - 8))

    def draw_overlay(self, surf: pygame.Surface, c: Controller) -> None:
        # Overlay UI panels (semi-transparent black for readability).
        def panel_bg(r: pygame.Rect) -> None:
            bg = pygame.Surface(r.size, pygame.SRCALPHA)
            bg.fill((0, 0, 0, 110))
            surf.blit(bg, r.topleft)
            pygame.draw.rect(surf, (45, 45, 55), r, width=2, border_radius=10)

        def fmt_hms(t_s: float) -> str:
            t = max(0, int(t_s))
            h = t // 3600
            m = (t % 3600) // 60
            s = t % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        v_vert = float(self.stats.get("v_vert", 0.0))
        v_hor = float(self.stats.get("v_hor", 0.0))
        h_km = float(self.stats.get("h_km", 0.0))
        fuel = float(self.stats.get("fuel_pct", 0.0))
        t_ext = float(self.stats.get("t_ext", 0.0))
        t_int = float(self.stats.get("t_int", 0.0))
        g_load = float(self.stats.get("g", 0.0))

        # --- top-center: mission time (NASA-like HH:MM:SS) ---
        t_s = float(self.stats.get("t_s", 0.0))
        time_txt = self._cached_render(self.font_mono, fmt_hms(t_s), (240, 240, 245))
        chip = pygame.Rect(0, 0, time_txt.get_width() + 22, time_txt.get_height() + 12)
        chip.midtop = (self.visual_rect.centerx, self.visual_rect.top + 10)
        pygame.draw.rect(surf, (10, 10, 12), chip, border_radius=12)
        pygame.draw.rect(surf, (120, 120, 135), chip, width=2, border_radius=12)
        surf.blit(time_txt, time_txt.get_rect(center=chip.center))

        # --- instruments overlay ---
        panel = self.instr_rect

        # Avoid shrinking instruments by height; keep a stable readable radius.
        dial_r = int(72 * self.ui_scale)
        if panel.height < 2 * dial_r + 46:
            dial_r = max(52, (panel.height - 46) // 2)
        dial_y = panel.top + dial_r + int(8 * self.ui_scale)

        gap = int(12 * self.ui_scale)
        tape_w = int(120 * self.ui_scale)
        fuel_w = int(92 * self.ui_scale)
        # If width is tight, shrink side gauges a bit.
        min_side = 64
        if tape_w + fuel_w + 2 * gap + 4 * dial_r + 28 > panel.width:
            overflow = (tape_w + fuel_w + 2 * gap + 4 * dial_r + 28) - panel.width
            shrink_each = int(np.ceil(overflow / 2))
            tape_w = max(min_side, tape_w - shrink_each)
            fuel_w = max(min_side, fuel_w - shrink_each)

        # Altitude tape left of v_vert; fuel immediately right of round dials.
        tape_rect = pygame.Rect(panel.left + 8, panel.top + 2, tape_w, panel.height - 4)
        dial_area_left = tape_rect.right + gap
        dial1_c = (dial_area_left + dial_r, dial_y)
        dial2_c = (dial1_c[0] + 2 * dial_r + 28, dial_y)
        fuel_rect = pygame.Rect(dial2_c[0] + dial_r + gap, panel.top + 2, fuel_w, panel.height - 4)
        if fuel_rect.right > panel.right - 8:
            fuel_rect.right = panel.right - 8
            dial2_c = (fuel_rect.left - gap - dial_r, dial2_c[1])
            dial1_c = (dial2_c[0] - (2 * dial_r + 28), dial1_c[1])

        # Thermometers under the two round dials (raise them; position by center so bulb is accounted for).
        therm_w = int(np.clip(int(70 * self.ui_scale), int(60 * self.ui_scale), int(90 * self.ui_scale)))
        therm_h = int(np.clip(int(150 * self.ui_scale), int(110 * self.ui_scale), int(160 * self.ui_scale)))
        therm1 = pygame.Rect(0, 0, therm_w, therm_h)
        therm2 = pygame.Rect(0, 0, therm_w, therm_h)

        # Place center a bit below dial bottom (higher than previous midtop layout).
        center_y = int(dial1_c[1] + dial_r + (therm_h * 0.36))
        therm1.center = (dial1_c[0], center_y)
        therm2.center = (dial2_c[0], center_y)

        # Digital G display between thermometers.
        g_rect = pygame.Rect(0, 0, int(160 * self.ui_scale), int(56 * self.ui_scale))
        g_rect.center = (
            (dial1_c[0] + dial2_c[0]) // 2,
            int(dial1_c[1] - dial_r - (18 * self.ui_scale)),
        )

        # Shift the main instruments block upward (tape + dials + fuel).
        # Slightly less than 1/3 to avoid pushing too high.
        shift_main = int(panel.height / 4)
        tape_rect.y -= shift_main
        fuel_rect.y -= shift_main
        dial1_c = (dial1_c[0], dial1_c[1] - shift_main)
        dial2_c = (dial2_c[0], dial2_c[1] - shift_main)
        d1_bb = pygame.Rect(0, 0, 2 * dial_r + 6, 2 * dial_r + 6)
        d1_bb.center = dial1_c
        d2_bb = pygame.Rect(0, 0, 2 * dial_r + 6, 2 * dial_r + 6)
        d2_bb.center = dial2_c

        # Move g-display up by same amount + half thermometer height (separate block).
        g_rect.centery -= shift_main + (therm_h // 2)

        # Fit background to instruments content with even padding (draw it BEFORE instruments).
        pad = int(10 * self.ui_scale)
        # Main background for primary instruments (avoid covering thermometers).
        main_bg = tape_rect.union(fuel_rect).union(d1_bb).union(d2_bb)
        panel_bg(main_bg.inflate(2 * pad, 2 * pad))
        # Separate lower background for thermometers only (keeps it lower, avoids overlap).
        thermo_bg = therm1.union(therm2)
        panel_bg(thermo_bg.inflate(2 * pad, 2 * pad))

        # v_vert zones (abs speed)
        vv_abs = abs(v_vert)
        vv_zones = [
            (0.0, 10.0, (80, 220, 120)),
            (10.0, 20.0, (230, 190, 90)),
            (20.0, 80.0, (230, 80, 80)),
        ]
        self._draw_dial(
            surf,
            dial1_c,
            dial_r,
            self.t("v_vert"),
            vv_abs,
            "m/s",
            vmin=0.0,
            vmax=80.0,
            zones=vv_zones,
            tick_step=10.0,
            danger_value=20.0,
        )

        # v_hor zones
        vh_abs = abs(v_hor)
        vh_zones = [
            (0.0, 3.0, (80, 220, 120)),
            (3.0, 10.0, (230, 190, 90)),
            (10.0, 40.0, (230, 80, 80)),
        ]
        self._draw_dial(
            surf,
            dial2_c,
            dial_r,
            self.t("v_hor"),
            vh_abs,
            "m/s",
            vmin=0.0,
            vmax=40.0,
            zones=vh_zones,
            tick_step=5.0,
            danger_value=10.0,
        )

        # Side gauges: altitude (left) and fuel (right)
        self._draw_altitude_tape(surf, tape_rect, h_km=h_km)

        fuel_zones = [
            (0.0, 10.0, (230, 80, 80)),
            (10.0, 30.0, (230, 190, 90)),
            (30.0, 100.0, (80, 220, 120)),
        ]
        self._draw_vertical_bar_gauge(
            surf,
            fuel_rect,
            self.t("fuel"),
            value=float(np.clip(fuel, 0.0, 100.0)),
            unit="%",
            vmin=0.0,
            vmax=100.0,
            zones=fuel_zones,
            ticks=[0.0, 10.0, 30.0, 50.0, 100.0],
            value_fmt="{:.1f}",
        )

        # Temperatures (thermometers)
        self._draw_thermometer(surf, therm1, self.t("t_ext"), t_ext)
        self._draw_thermometer(surf, therm2, self.t("t_int"), t_int)

        # Digital G-load display (replaces text block)
        pygame.draw.rect(surf, (10, 10, 12), g_rect, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), g_rect, width=2, border_radius=10)
        pygame.draw.rect(surf, (40, 40, 45), g_rect.inflate(-6, -6), width=1, border_radius=9)
        g_label = self._cached_render(self.font_small, self.t("g"), (220, 220, 230))
        surf.blit(g_label, (g_rect.left + int(10 * self.ui_scale), g_rect.top + int(6 * self.ui_scale)))
        g_txt = self._cached_render(self.font_mono, f"{g_load:05.2f} g", (120, 230, 190))
        surf.blit(g_txt, g_txt.get_rect(center=(g_rect.centerx, g_rect.centery + int(8 * self.ui_scale))))

        if c.model.result == SimResult.SUCCESS:
            msg = self._cached_render(self.font_big, self.t("success"), (80, 220, 120))
            surf.blit(msg, (self.visual_rect.right - msg.get_width() - 16, self.visual_rect.top + 16))
        elif bool(c.model.failed):
            msg = self._cached_render(self.font_big, self.t("failure"), (230, 80, 80))
            surf.blit(msg, (self.visual_rect.right - msg.get_width() - 16, self.visual_rect.top + 16))
            reason = str(c.model.failure_reason)
            if reason:
                reason_txt = self._cached_render(self.font_small, reason, (230, 190, 90))
                surf.blit(
                    reason_txt,
                    (self.visual_rect.right - reason_txt.get_width() - 16, self.visual_rect.top + 50),
                )

        self._draw_controls(surf, c)
        self._draw_mission_dossier_button(surf, c)
        self._draw_modal_overlays(surf, c)

    def _draw_telemetry_panel(self, surf: pygame.Surface) -> None:
        tr = self.telemetry_rect
        bg = pygame.Surface(tr.size, pygame.SRCALPHA)
        bg.fill((0, 0, 0, 110))
        surf.blit(bg, tr.topleft)
        pygame.draw.rect(surf, (45, 45, 55), tr, width=2, border_radius=10)
        dist = float(self.stats.get("dist_m", 0.0))
        raw = str(self.stats.get("surface", "land"))
        p = float(self.stats.get("p", 0.0))
        if dist >= 1000.0:
            dtxt = f"{self.t('dist')}: {dist / 1000.0:.2f} km"
        else:
            dtxt = f"{self.t('dist')}: {dist:.0f} m"
        lines = (
            dtxt,
            f"{self.t('surface')}: {self._surface_display(raw)}",
            f"{self.t('p')}: {p:.3f} {self.t('bar_unit')}",
        )
        y = tr.top + int(8 * self.ui_scale)
        for line in lines:
            t = self._cached_render(self.font_small, line, (220, 220, 230))
            surf.blit(t, (tr.left + int(10 * self.ui_scale), y))
            y += t.get_height() + int(2 * self.ui_scale)

    def _draw_modal_overlays(self, surf: pygame.Surface, c: Controller) -> None:
        now = pygame.time.get_ticks()
        if self.sim_paused and not self.esc_menu_open:
            chip_txt = self._cached_render(self.font_big, self.t("paused_banner"), (230, 190, 90))
            chip = pygame.Rect(0, 0, chip_txt.get_width() + 28, chip_txt.get_height() + 16)
            chip.center = (self.visual_rect.centerx, int(self.visual_rect.centery * 0.42))
            dim = pygame.Surface(chip.size, pygame.SRCALPHA)
            dim.fill((0, 0, 0, 170))
            surf.blit(dim, chip.topleft)
            pygame.draw.rect(surf, (140, 120, 80), chip, width=2, border_radius=12)
            surf.blit(chip_txt, chip_txt.get_rect(center=chip.center))

        if self.show_help:
            self._draw_help_panel(surf)

        if self.esc_menu_open:
            self._draw_esc_menu(surf, c)

        if now < self._toast_until_ms:
            toast = self._cached_render(self.font_small, self.t(self._toast_key), (255, 200, 120))
            bx = toast.get_width() + 24
            by = toast.get_height() + 16
            bg = pygame.Surface((bx, by), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 200))
            r = bg.get_rect(midbottom=(self.visual_rect.centerx, self.visual_rect.bottom - int(80 * self.ui_scale)))
            surf.blit(bg, r.topleft)
            pygame.draw.rect(surf, (120, 100, 70), r, width=2, border_radius=10)
            surf.blit(toast, toast.get_rect(center=r.center))

    def _draw_help_panel(self, surf: pygame.Surface) -> None:
        pad = int(18 * self.ui_scale)
        title = self._cached_render(self.font_big, self.t("help_title"), (240, 240, 245))
        body_lines = self.t("help_body").split("\n")
        body_surfs = [self._cached_render(self.font_small, ln, (210, 210, 220)) for ln in body_lines]
        bw = max(title.get_width(), max((s.get_width() for s in body_surfs), default=0)) + 2 * pad
        bh = title.get_height() + int(12 * self.ui_scale) + sum(s.get_height() + 4 for s in body_surfs) + 2 * pad
        panel = pygame.Rect(0, 0, min(bw, self.visual_rect.width - 2 * pad), bh)
        panel.center = self.visual_rect.center
        dim = pygame.Surface(self.visual_rect.size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        surf.blit(dim, (0, 0))
        bg = pygame.Surface(panel.size, pygame.SRCALPHA)
        bg.fill((12, 12, 16, 230))
        surf.blit(bg, panel.topleft)
        pygame.draw.rect(surf, (100, 100, 120), panel, width=2, border_radius=12)
        y = panel.top + pad
        surf.blit(title, (panel.left + pad, y))
        y += title.get_height() + int(10 * self.ui_scale)
        for s in body_surfs:
            surf.blit(s, (panel.left + pad, y))
            y += s.get_height() + int(4 * self.ui_scale)
        hint = self._cached_render(self.font_small, "Esc — закрыть" if self.lang == "RU" else "Esc — close", (150, 150, 160))
        surf.blit(hint, (panel.left + pad, panel.bottom - pad - hint.get_height()))

    def _draw_esc_menu(self, surf: pygame.Surface, c: Controller) -> None:
        dim = pygame.Surface(self.visual_rect.size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        surf.blit(dim, (0, 0))
        r_res, r_mode, r_lang, r_rst, r_log, r_quit = self._esc_menu_layout_full()
        title = self._cached_render(self.font_big, self.t("esc_title"), (240, 240, 245))
        cx = self.visual_rect.centerx
        surf.blit(title, title.get_rect(center=(cx, r_res.top - int(36 * self.ui_scale))))

        def pill(rect: pygame.Rect, label: str, hot: bool) -> None:
            bgc = (50, 55, 65) if hot else (35, 36, 42)
            pygame.draw.rect(surf, bgc, rect, border_radius=10)
            pygame.draw.rect(surf, (160, 165, 180), rect, width=2, border_radius=10)
            t = self._cached_render(self.font_small, label, (235, 235, 240))
            surf.blit(t, t.get_rect(center=rect.center))

        log_on = c.model.csv_logging_enabled
        pill(r_res, self.t("esc_resume"), True)
        pill(r_mode, self.t("auto") if self.auto_mode else self.t("manual"), True)
        pill(r_lang, self.t("lang"), True)
        pill(r_rst, self.t("restart"), True)
        pill(r_log, self.t("csv_log_on") if log_on else self.t("csv_log_off"), log_on)
        pill(r_quit, self.t("esc_quit"), True)

    def _draw_controls(self, surf: pygame.Surface, c: Controller) -> None:
        # --- top-left: minimap + time (in visual area) ---
        pygame.draw.rect(surf, (24, 24, 28), self.map_rect, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), self.map_rect, width=2, border_radius=10)

        hint_s = self._cached_render(self.font_small, self.t("target"), (190, 190, 200))
        surf.blit(hint_s, (self.map_rect.left, self._map_stack_y0))
        self._draw_telemetry_panel(surf)

        self.time_scale_slider.draw(surf)
        ts = self.time_scale()
        ts_label = f"TIME: {ts:.2f}x" if ts < 20 else f"TIME: {ts:.0f}x"
        ts_txt = self._cached_render(self.font_small, ts_label, (220, 220, 230))
        surf.blit(ts_txt, (self.time_scale_slider.rect.left, self.time_scale_slider.rect.bottom + 6))

        # --- bottom-right: flight controls background (fit-to-content like instruments) ---
        thr_txt = self._cached_render(self.font_small, f"{int(self.throttle_slider.value * 100):d}%", (220, 220, 230))
        thr_bb = thr_txt.get_rect(topleft=(self.throttle_slider.rect.right + 8, self.throttle_slider.rect.top - 2))
        content = self.engine_toggle_rect.union(self.throttle_slider.rect).union(thr_bb)
        for lv in self.levers:
            content = content.union(lv.rect)
        pad = 10
        flight_group = content.inflate(2 * pad, 2 * pad)
        bg2 = pygame.Surface(flight_group.size, pygame.SRCALPHA)
        bg2.fill((0, 0, 0, 110))
        surf.blit(bg2, flight_group.topleft)
        pygame.draw.rect(surf, (45, 45, 55), flight_group, width=2, border_radius=10)

        # --- bottom-right: flight controls (in visual area) ---
        for lv in self.levers:
            lv.draw(surf, self.font_small, self.t(lv.label_key), c)

        eng_bg = (40, 40, 45)
        pygame.draw.rect(surf, eng_bg, self.engine_toggle_rect, border_radius=8)
        pygame.draw.rect(surf, (180, 180, 190), self.engine_toggle_rect, width=2, border_radius=8)
        eng_state = self.t("engine_on") if c.model.engine_on else self.t("engine_off")
        eng_txt = self._cached_render(self.font_small, f"{self.t('engine')}: {eng_state}", (235, 235, 235))
        surf.blit(eng_txt, eng_txt.get_rect(center=self.engine_toggle_rect.center))

        self.throttle_slider.draw(surf)
        surf.blit(thr_txt, thr_bb.topleft)

    def draw_pause_control(self, surf: pygame.Surface) -> None:
        sim_group = self.pause_rect.inflate(14, 14)
        bg = pygame.Surface(sim_group.size, pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        surf.blit(bg, sim_group.topleft)
        pygame.draw.rect(surf, (45, 45, 55), sim_group, width=2, border_radius=10)
        pr = self.pause_rect
        bgc = (40, 40, 45)
        pygame.draw.rect(surf, bgc, pr, border_radius=8)
        pygame.draw.rect(surf, (180, 180, 190), pr, width=2, border_radius=8)
        txt = self._cached_render(self.font_small, self.t("pause"), (235, 235, 235))
        surf.blit(txt, txt.get_rect(center=pr.center))

    def _draw_plot_panel(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        xs: list[float],
        ys: list[float],
        title: str,
        color: tuple[int, int, int],
    ) -> None:
        pygame.draw.rect(surf, (18, 18, 22), rect, border_radius=8)
        pygame.draw.rect(surf, (90, 90, 102), rect, width=1, border_radius=8)
        tt = self._cached_render(self.font_small, title, (200, 200, 210))
        surf.blit(tt, (rect.left + 6, rect.top + 4))
        inner = rect.inflate(-12, -24)
        inner.y += 14
        n = min(len(xs), len(ys))
        if n < 2:
            return
        ymin = float(min(ys[:n]))
        ymax = float(max(ys[:n]))
        if abs(ymax - ymin) < 1e-9:
            ymin -= 1.0
            ymax += 1.0
        span_y = ymax - ymin
        pad_y = max(span_y * 0.08, 1e-6)
        ymin -= pad_y
        ymax += pad_y
        x0 = float(xs[0])
        x1 = float(xs[n - 1])
        if abs(x1 - x0) < 1e-9:
            return
        pts: list[tuple[int, int]] = []
        for i in range(n):
            tx = (float(xs[i]) - x0) / (x1 - x0)
            ty = (float(ys[i]) - ymin) / (ymax - ymin)
            px = int(inner.left + tx * inner.width)
            py = int(inner.bottom - ty * inner.height)
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(surf, color, False, pts, 2)

    def _mission_report_geometry(self) -> _MissionReportGeom:
        vr = self.visual_rect
        scale = self.ui_scale
        pw = int(np.clip(vr.width * 0.88, 520, vr.width - 24))
        ph = int(np.clip(vr.height * 0.84, 400, vr.height - 24))
        panel = pygame.Rect(0, 0, pw, ph)
        panel.center = vr.center

        pad = max(12, int(14 * scale))
        inner = panel.inflate(-2 * pad, -2 * pad)
        title_h = int(32 * scale) + int(20 * scale)
        foot_h = int(48 * scale)
        content_top = inner.top + title_h + 6
        content_bot = inner.bottom - foot_h
        split = inner.left + int(inner.width * 0.36)
        gap_m = int(10 * scale)
        photo_w = max(60, split - inner.left - gap_m - 8)
        photo = pygame.Rect(inner.left + 4, content_top, photo_w, content_bot - content_top)
        px = split + gap_m
        plot_w = inner.right - px - 4
        avail_h = content_bot - content_top
        plot_gap = int(8 * scale)
        plot_h = max(40, (avail_h - 2 * plot_gap) // 3)
        p0 = pygame.Rect(px, content_top, plot_w, plot_h)
        p1 = pygame.Rect(px, content_top + plot_h + plot_gap, plot_w, plot_h)
        p2 = pygame.Rect(px, content_top + 2 * (plot_h + plot_gap), plot_w, plot_h)
        bw = int(np.clip(168 * scale, 96, 220))
        bh = int(36 * scale)
        restart_btn = pygame.Rect(inner.right - bw - 4, inner.bottom - bh - 6, bw, bh)
        close_btn = pygame.Rect(restart_btn.left - bw - 12, restart_btn.top, bw, bh)
        return _MissionReportGeom(panel, close_btn, restart_btn, photo, p0, p1, p2)

    def _draw_mission_dossier_button(self, surf: pygame.Surface, c: Controller) -> None:
        if c.model.result == SimResult.RUNNING or self.mission_report_open:
            return
        r = self.mission_report_btn_rect
        bg = pygame.Surface(r.size, pygame.SRCALPHA)
        bg.fill((0, 0, 0, 130))
        surf.blit(bg, r.topleft)
        pygame.draw.rect(surf, (75, 70, 55), r, width=2, border_radius=8)
        pygame.draw.rect(surf, (42, 38, 32), r, border_radius=8)
        t = self._cached_render(self.font_small, self.t("mission_dossier_btn"), (235, 228, 200))
        surf.blit(t, t.get_rect(center=r.center))

    def _format_mission_elapsed(self, t_s: float) -> str:
        t_s = max(0.0, float(t_s))
        h = int(t_s // 3600)
        m = int((t_s % 3600) // 60)
        s = int(t_s % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _telemetry_user_events(self, hist: list[dict]) -> list[tuple[float, tuple[int, int, int], int]]:
        """User actions: (t_rel_s, rgb, tri_dir) tri_dir 1 = marker at top (▼), -1 = at bottom (▲ off)."""
        col_hs = (200, 95, 40)
        col_dr = (40, 150, 185)
        col_mn = (45, 130, 65)
        col_cj = (155, 55, 155)
        col_en_on = (215, 165, 40)
        col_en_off = (115, 75, 45)
        out: list[tuple[float, tuple[int, int, int], int]] = []
        if len(hist) < 2:
            return out

        def _g(row: dict, k: str) -> float:
            return float(row.get(k, 0.0))

        t0 = float(hist[0]["t_s"])
        for i in range(1, len(hist)):
            prev, cur = hist[i - 1], hist[i]
            tr = float(cur["t_s"]) - t0
            a, b = _g(prev, "hs"), _g(cur, "hs")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_hs, 1))
            a, b = _g(prev, "dr"), _g(cur, "dr")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_dr, 1))
            a, b = _g(prev, "mn"), _g(cur, "mn")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_mn, 1))
            a, b = _g(prev, "cj"), _g(cur, "cj")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_cj, 1))
            a, b = _g(prev, "eng"), _g(cur, "eng")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_en_on, 1))
            elif a > 0.5 and b < 0.5:
                out.append((tr, col_en_off, -1))
        return out

    def _draw_plot_blueprint(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        xs: list[float],
        ys: list[float],
        title: str,
        color: tuple[int, int, int],
        events: Optional[list[tuple[float, tuple[int, int, int], int]]] = None,
    ) -> None:
        events = events or []
        paper = (228, 224, 212)
        grid_maj = (200, 196, 180)
        grid_min = (216, 212, 200)
        ink = (28, 42, 68)
        pygame.draw.rect(surf, paper, rect, border_radius=4)
        pygame.draw.rect(surf, ink, rect, width=2, border_radius=4)
        inner = rect.inflate(-10, -22)
        inner.y += 12
        axis_h = max(44, int(50 * self.ui_scale))
        plot_inner = pygame.Rect(inner.left, inner.top, inner.width, max(12, inner.height - axis_h))

        step = max(8, int(12 * self.ui_scale))
        gx = plot_inner.left
        while gx <= plot_inner.right:
            pygame.draw.line(
                surf,
                grid_min if (gx - plot_inner.left) % (step * 2) else grid_maj,
                (gx, plot_inner.top),
                (gx, plot_inner.bottom),
                1,
            )
            gx += step
        gy = plot_inner.top
        while gy <= plot_inner.bottom:
            pygame.draw.line(
                surf,
                grid_min if (gy - plot_inner.top) % (step * 2) else grid_maj,
                (plot_inner.left, gy),
                (plot_inner.right, gy),
                1,
            )
            gy += step
        cap = self._cached_render(self.font_small, title.upper(), ink)
        surf.blit(cap, (rect.left + 6, rect.top + 3))
        n = min(len(xs), len(ys))
        if n < 2:
            return
        ymin = float(min(ys[:n]))
        ymax = float(max(ys[:n]))
        if abs(ymax - ymin) < 1e-9:
            ymin -= 1.0
            ymax += 1.0
        span_y = ymax - ymin
        pad_y = max(span_y * 0.08, 1e-6)
        ymin -= pad_y
        ymax += pad_y
        xl0 = float(xs[0])
        xl1 = float(xs[n - 1])
        if abs(xl1 - xl0) < 1e-9:
            xl0 -= 0.5
            xl1 += 0.5

        def _x_to_px(t_rel: float) -> int:
            return int(plot_inner.left + (float(t_rel) - xl0) / (xl1 - xl0) * plot_inner.width)

        pts: list[tuple[int, int]] = []
        for i in range(n):
            tx = (float(xs[i]) - xl0) / (xl1 - xl0)
            ty = (float(ys[i]) - ymin) / (ymax - ymin)
            px = int(plot_inner.left + tx * plot_inner.width)
            py = int(plot_inner.bottom - ty * plot_inner.height)
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(surf, color, False, pts, 2)
            pygame.draw.lines(surf, ink, False, pts, 1)

        ax_y = plot_inner.bottom
        pygame.draw.line(surf, ink, (plot_inner.left, ax_y), (plot_inner.right, ax_y), 2)
        nt = 6 if (xl1 - xl0) > 3600.0 else 5
        tick_y = ax_y + 5
        for k in range(nt + 1):
            t_rel = xl0 + (xl1 - xl0) * (k / float(nt))
            px = _x_to_px(t_rel)
            pygame.draw.line(surf, ink, (px, ax_y), (px, ax_y + 4), 2)
            tl = self._cached_render(self.font_small, self._format_mission_elapsed(t_rel), ink)
            lx = int(np.clip(px - tl.get_width() // 2, plot_inner.left, max(plot_inner.left, plot_inner.right - tl.get_width())))
            surf.blit(tl, (lx, tick_y))

        tri_s = max(5, int(6 * self.ui_scale))
        for t_rel, evc, tri_dir in events:
            if t_rel < xl0 - 1e-6 or t_rel > xl1 + 1e-6:
                continue
            px = _x_to_px(t_rel)
            if px < plot_inner.left - 2 or px > plot_inner.right + 2:
                continue
            pygame.draw.line(surf, evc, (px, plot_inner.top), (px, plot_inner.bottom), 2)
            if tri_dir >= 0:
                pts_t = [(px, plot_inner.top), (px - tri_s, plot_inner.top + tri_s + 2), (px + tri_s, plot_inner.top + tri_s + 2)]
            else:
                pts_t = [
                    (px, plot_inner.bottom),
                    (px - tri_s, plot_inner.bottom - tri_s - 2),
                    (px + tri_s, plot_inner.bottom - tri_s - 2),
                ]
            pygame.draw.polygon(surf, evc, pts_t)
            pygame.draw.polygon(surf, ink, pts_t, 1)

    def _draw_landing_site_snapshot(self, surf: pygame.Surface, frame: pygame.Rect, c: Controller) -> None:
        if frame.width < 12 or frame.height < 12:
            return
        w, h = frame.w - 10, frame.h - 10
        if w < 4 or h < 4:
            return
        tmp = pygame.Surface((w, h))
        sx = float(c.model.pos_x_m)
        sz = float(c.model.pos_z_m)
        seed = int(abs(sx * 0.013) + abs(sz * 0.017)) & 0xFFFF
        phase = (seed % 628) * 0.01

        hsky = int(h * 0.58)
        for y in range(hsky):
            t = y / max(1, hsky - 1)
            col = (int(72 + 55 * t), int(48 + 38 * t), int(28 + 22 * t))
            pygame.draw.line(tmp, col, (0, y), (w, y))
        for y in range(hsky, h):
            t = (y - hsky) / max(1, h - hsky - 1)
            col = (int(52 + 35 * t), int(40 + 28 * t), int(28 + 18 * t))
            pygame.draw.line(tmp, col, (0, y), (w, y))

        hy = hsky
        dune_pts: list[tuple[int, int]] = [(0, hy)]
        for x in range(0, w + 4, 3):
            dy = int(7 * math.sin(x * 0.042 + phase) + 4 * math.sin(x * 0.1 + phase * 1.7))
            dune_pts.append((x, hy + dy))
        dune_pts.append((w, h))
        dune_pts.append((0, h))
        pygame.draw.polygon(tmp, (36, 30, 24), dune_pts)
        pygame.draw.lines(tmp, (24, 20, 16), False, dune_pts[: w // 3 + 8], 2)

        raw = str(self.stats.get("surface", "land"))
        if raw == "lake":
            haz = pygame.Surface((w, hsky), pygame.SRCALPHA)
            haz.fill((35, 65, 92, 70))
            tmp.blit(haz, (0, 0))

        cx = w // 2 + int(14 * math.sin(phase))
        rim_y = int(h * 0.48)
        vtx_y = int(h * 0.64)
        rw = int(w * 0.41)
        dish_pts: list[tuple[int, int]] = []
        for i in range(45):
            u = i / 44.0 * 2.0 - 1.0
            x = int(cx + u * rw)
            y = int(rim_y + (1.0 - u * u) * (vtx_y - rim_y))
            dish_pts.append((x, y))
        dish_pts.append((cx + rw + 10, vtx_y + 22))
        dish_pts.append((cx - rw - 10, vtx_y + 22))
        pygame.draw.polygon(tmp, (58, 54, 50), dish_pts)
        pygame.draw.polygon(tmp, (92, 88, 80), dish_pts, 2)
        hi = [dish_pts[i] for i in range(0, len(dish_pts) - 2, 7)]
        if len(hi) > 2:
            pygame.draw.lines(tmp, (130, 126, 118), False, hi, 1)

        lx = cx + int(rw * 0.32)
        ly = rim_y - 14
        pygame.draw.rect(tmp, (40, 44, 48), pygame.Rect(lx, ly, 30, 20))
        pygame.draw.polygon(tmp, (48, 52, 56), [(lx + 6, ly), (lx + 24, ly - 12), (lx + 24, ly)])
        fx, fy = cx, int(rim_y + (vtx_y - rim_y) * 0.42)
        pygame.draw.line(tmp, (32, 36, 40), (lx + 15, ly + 6), (fx, fy), 2)

        for t in range(2):
            pygame.draw.rect(tmp, (18, 16, 14), pygame.Rect(t, t, w - 2 * t, h - 2 * t), 1)

        surf.blit(tmp, (frame.left + 5, frame.top + 5))

    def draw_mission_report_modal(self, surf: pygame.Surface, c: Controller) -> None:
        if c.model.result == SimResult.RUNNING or not self.mission_report_open:
            return
        g = self._mission_report_geometry()
        dim = pygame.Surface(self.visual_rect.size, pygame.SRCALPHA)
        dim.fill((20, 18, 14, 200))
        surf.blit(dim, (0, 0))

        panel = g.panel
        parchment = (244, 238, 220)
        pygame.draw.rect(surf, parchment, panel, border_radius=6)
        pygame.draw.rect(surf, (32, 28, 22), panel, width=4, border_radius=6)
        pygame.draw.rect(surf, (18, 16, 12), panel.inflate(-8, -8), width=1, border_radius=4)

        inner = panel.inflate(-int(22 * self.ui_scale), -int(22 * self.ui_scale))
        title = self._cached_render(self.font, self.t("mission_blueprint_title"), (32, 28, 22))
        surf.blit(title, (inner.left, inner.top - int(4 * self.ui_scale)))
        leg = self._cached_render(self.font_small, self.t("mission_plot_legend"), (55, 48, 42))
        surf.blit(leg, (inner.left, inner.top + int(30 * self.ui_scale)))

        photo_frame = g.photo.inflate(-4, -4)
        pf = pygame.Rect(photo_frame.left, photo_frame.top + int(18 * self.ui_scale), photo_frame.width, photo_frame.height - int(18 * self.ui_scale))
        pygame.draw.rect(surf, (210, 205, 190), pf, border_radius=4)
        pygame.draw.rect(surf, (40, 36, 30), pf, width=2, border_radius=4)
        lbl = self._cached_render(self.font_small, self.t("mission_photo_label"), (55, 48, 40))
        surf.blit(lbl, (photo_frame.left + 4, photo_frame.top + 2))
        self._draw_landing_site_snapshot(surf, pf, c)

        hist = c.model.telemetry_history
        plots = (g.plot0, g.plot1, g.plot2)
        colors = ((40, 85, 140), (50, 110, 70), (130, 70, 45))
        if len(hist) >= 2:
            t0 = float(hist[0]["t_s"])
            xs = [float(p["t_s"]) - t0 for p in hist]
            user_events = self._telemetry_user_events(hist)
            series = (
                ([float(p["altitude_m"]) / 1000.0 for p in hist], self.t("plot_h_km"), colors[0]),
                ([float(p["v_vert_mps"]) for p in hist], self.t("plot_v_vert"), colors[1]),
                ([float(p["g_load"]) for p in hist], self.t("plot_g"), colors[2]),
            )
            for i, pr in enumerate(plots):
                ys, ptitle, col = series[i]
                self._draw_plot_blueprint(surf, pr, xs, ys, ptitle, col, user_events)
        else:
            empty = self._cached_render(self.font_small, self.t("mission_no_plots"), (80, 60, 45))
            surf.blit(empty, (g.plot0.left + 8, g.plot0.top + 8))

        def pill_btn(r: pygame.Rect, label: str, hot: bool) -> None:
            bc = (52, 48, 42) if hot else (38, 34, 30)
            pygame.draw.rect(surf, bc, r, border_radius=8)
            pygame.draw.rect(surf, (28, 24, 20), r, width=2, border_radius=8)
            t = self._cached_render(self.font_small, label, (238, 232, 210))
            surf.blit(t, t.get_rect(center=r.center))

        pill_btn(g.close_btn, self.t("mission_close"), True)
        pill_btn(g.restart_btn, self.t("mission_new_flight"), True)

    def consume_restart_request(self) -> bool:
        v = self._restart_requested
        self._restart_requested = False
        return v

