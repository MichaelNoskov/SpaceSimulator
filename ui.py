from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import math
import re

import numpy as np
import pygame

from control.commands import Command
from control.controller import Controller
from digital_twin.config import BodyConfig, DigitalTwinConfig
from digital_twin.model import PhysicsModel, SimResult
from flight_program import (
    AP_API_COMPLETIONS,
    DEFAULT_SCRIPT,
    HUYGENS_SCRIPT,
    FlightProgramRunner,
    SIM_API_COMPLETIONS,
    compile_flight_program,
    validate_flight_program_tick,
)
from flight_program.highlighter import iter_flight_program_tokens

_ALT_H_KM_TOP = BodyConfig().entry_start_altitude_m * 1e-3

I18N = {
    "RU": {
        "title": "Симулятор посадки на Титан",
        "h": "Высота",
        "v_vert": "Верт. скорость",
        "v_hor": "Гор. скорость",
        "g": "Перегрузка",
        "t_ext": "Т наружная",
        "t_int": "Т внутр.",
        "t_hs_skin": "Обшивка",
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
        "fail_overload": "Превышена допустимая перегрузка",
        "fail_t_int_min": "Внутренняя температура ниже минимума",
        "fail_t_int_max": "Внутренняя температура выше максимума",
        "fail_hs_thermal": "Термический предел теплозащиты",
        "fail_hard_landing": "Жёсткая посадка",
        "fail_wrong_site": "Неверная площадка посадки",
        "fail_fuel": "Топливо исчерпано над поверхностью",
        "fail_terrain": "Столкновение с рельефом",
        "csv_log_on": "CSV LOG: ВКЛ",
        "csv_log_off": "CSV LOG: ВЫКЛ",
        "lever_denied": "Сейчас недоступно",
        "help_title": "Управление",
        "help_body": (
            "При запуске и после «Рестарт» — параметры миссии (прокрутка списка): вход, массы, A_ref/Cd, площади парашютов, двигатель.\n"
            "Клик по мини-карте — выбор точки посадки.\n"
            "Рычаги: зелёный индикатор — можно включить.\n"
            "Авто: логика из «Программирования полёта» (Python, tick(sim, ap)); по умолчанию — шаблон ТЗ, в редакторе можно подставить шаблон Huygens.\n"
            "Esc / Пробел — меню паузы (продолжить, авто/ручной, язык, рестарт, CSV, программа полёта, выход). R — рестарт, A — авто/ручной, F1 — эта справка.\n"
            "+/− — скорость времени, F11 — полный экран.\n"
            "После завершения полёта — «Досье миссии»: чертежи телеметрии и вид площадки посадки."
        ),
        "esc_resume": "Продолжить",
        "esc_quit": "Выйти",
        "esc_flight_program": "Программирование полёта",
        "esc_title": "Меню",
        "fp_title": "Программа автопилота",
        "fp_save": "Сохранить",
        "fp_cancel": "Отменить",
        "fp_reset_default": "Шаблон: ТЗ",
        "fp_reset_huygens": "Шаблон: Huygens",
        "fp_hint": "Определите tick(sim, ap). sim — показания, ap — команды. Внизу: два шаблона (ТЗ и Huygens) — подставить в редактор без сохранения.",
        "fp_compile_error": "Ошибка программы",
        "fp_runtime_error": "Ошибка в полёте, авто выкл.",
        "fp_validate_error": "Проверка tick()",
        "fp_hints_title": "Подсказки API",
        "fp_header_sim": "sim (чтение)",
        "fp_header_ap": "ap (команды)",
        "fp_doc_ref": "См. FLIGHT_PROGRAM_RU.md",
        "post_landing_plots": "Телеметрия полёта",
        "plot_h_km": "Высота, км",
        "plot_v_vert": "Верт. скорость, м/с",
        "plot_v_vert_abs": "|Верт. скорость|, м/с",
        "plot_g": "Перегрузка, g",
        "mission_dossier_btn": "Досье миссии",
        "mission_close": "Закрыть",
        "mission_new_flight": "Новый полёт",
        "mission_blueprint_title": "Пакет документации посадки",
        "mission_photo_label": "Площадка посадки",
        "mission_no_plots": "Недостаточно точек телеметрии",
        "mission_plot_legend": (
            "Вертикаль: HS теплозащита, DR тормозной, MN основной, CJ сброс, EN двигатель (▼ выкл), "
            "TGT цель. Подписи у линий — команды, дошедшие до модели (ручной и авто режим). "
            "График времени: колёсико — сдвиг, Ctrl+колёсико — масштаб, [ ] — увеличить/уменьшить фрагмент, ползунок."
        ),
        "mission_timeline_hint": "Время полёта (прокрутка / масштаб)",
        "mission_plot_time_range": "Фрагмент: {a} — {b}  ·  миссия {total}",
        "mission_plot_stats": "min {mn} · max {mx} · конец {end}",
        "mission_x_axis_label": "время от старта",
        "mission_axis_t": "Время полёта",
        "plot_evt_hs": "HS · теплозащита",
        "plot_evt_dr": "DR · тормозной",
        "plot_evt_mn": "MN · основной",
        "plot_evt_cj": "CJ · сброс",
        "plot_evt_en_on": "EN · вкл",
        "plot_evt_en_off": "EN · выкл",
        "plot_evt_tgt": "TGT · цель",
        "ms_title": "Параметры миссии",
        "ms_alt": "Высота входа (км MSL)",
        "ms_speed": "Скорость входа (км/с)",
        "ms_fuel": "Топливо (кг)",
        "ms_start": "Начать миссию",
        "ms_defaults": "По умолчанию",
        "ms_hint": "",
        "ms_bad": "Проверьте числа и диапазоны (см. подсказку в консоли).",
        "esc_mission_setup": "Параметры старта",
        "ms_field_alt_km": "Высота входа, км MSL",
        "ms_field_speed_kms": "Скорость входа, км/с",
        "ms_field_fuel": "Топливо, кг",
        "ms_field_dry": "Сухая масса, кг",
        "ms_field_hs": "Масса теплозащиты, кг",
        "ms_field_a_ref": "Площадь миделя A_ref, м²",
        "ms_field_cd0": "Cd базовый (корпус)",
        "ms_field_drogue_a": "Тормозной парашют: площадь, м²",
        "ms_field_drogue_cd": "Тормозной парашют: Cd",
        "ms_field_main_a": "Основной парашют: площадь, м²",
        "ms_field_main_cd": "Основной парашют: Cd",
        "ms_field_t_max": "Двигатель: T_max, Н",
        "ms_field_isp": "Двигатель: Isp, с",
    },
    "EN": {
        "title": "Titan Landing Simulator",
        "h": "Altitude",
        "v_vert": "Vertical spd",
        "v_hor": "Horizontal spd",
        "g": "G-load",
        "t_ext": "Temp ext",
        "t_int": "Temp int",
        "t_hs_skin": "Heatshield",
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
        "fail_overload": "G-load limit exceeded",
        "fail_t_int_min": "Internal temperature below minimum",
        "fail_t_int_max": "Internal temperature above maximum",
        "fail_hs_thermal": "Heatshield thermal failure",
        "fail_hard_landing": "Hard landing",
        "fail_wrong_site": "Wrong landing site",
        "fail_fuel": "Fuel exhausted above ground",
        "fail_terrain": "Terrain collision",
        "csv_log_on": "CSV LOG: ON",
        "csv_log_off": "CSV LOG: OFF",
        "lever_denied": "Not available now",
        "help_title": "Controls",
        "help_body": (
            "On launch and after «Restart», mission parameters (scroll the list): entry, masses, A_ref/Cd, chute areas, engine.\n"
            "Click the minimap to set a landing target.\n"
            "Levers: green lamp means the action is allowed.\n"
            "Auto: «Flight program» (Python, tick(sim, ap)); default template matches the simulator spec; editor can load a Huygens-style timeline template.\n"
            "Esc / Space — pause menu (resume, auto/manual, lang, restart, CSV, flight program, quit). R — restart, A — auto/manual, F1 — this help.\n"
            "+/− — time speed, F11 — fullscreen.\n"
            "After landing, open «Mission dossier» for telemetry drawings and a landing site view."
        ),
        "esc_resume": "Resume",
        "esc_quit": "Quit",
        "esc_flight_program": "Flight program",
        "esc_title": "Menu",
        "fp_title": "Autopilot program",
        "fp_save": "Save",
        "fp_cancel": "Cancel",
        "fp_reset_default": "Template: spec",
        "fp_reset_huygens": "Template: Huygens",
        "fp_hint": "Define tick(sim, ap). sim reads instruments; ap sends commands. Footer: two templates (spec vs Huygens) load into the editor without saving.",
        "fp_compile_error": "Program error",
        "fp_runtime_error": "Runtime error, auto off.",
        "fp_validate_error": "tick() check failed",
        "fp_hints_title": "API hints",
        "fp_header_sim": "sim (read)",
        "fp_header_ap": "ap (actions)",
        "fp_doc_ref": "See FLIGHT_PROGRAM_EN.md",
        "post_landing_plots": "Flight telemetry",
        "plot_h_km": "Altitude, km",
        "plot_v_vert": "Vertical speed, m/s",
        "plot_v_vert_abs": "|Vertical speed|, m/s",
        "plot_g": "G-load, g",
        "mission_dossier_btn": "Mission dossier",
        "mission_close": "Close",
        "mission_new_flight": "New flight",
        "mission_blueprint_title": "Landing documentation pack",
        "mission_photo_label": "Landing site",
        "mission_no_plots": "Not enough telemetry samples",
        "mission_plot_legend": (
            "Markers: HS heatshield, DR drogue, MN main, CJ jettison, EN engine (▼ off), "
            "TGT map target. Labels are commands applied to the model (manual and auto). "
            "Timeline: wheel = pan, Ctrl+wheel = zoom, [ ] = zoom in/out, scrollbar."
        ),
        "mission_timeline_hint": "Mission time (scroll / zoom)",
        "mission_plot_time_range": "View: {a} – {b}  ·  mission {total}",
        "mission_plot_stats": "min {mn} · max {mx} · end {end}",
        "mission_x_axis_label": "time from launch",
        "mission_axis_t": "Mission time",
        "plot_evt_hs": "HS · heatshield",
        "plot_evt_dr": "DR · drogue",
        "plot_evt_mn": "MN · main",
        "plot_evt_cj": "CJ · jettison",
        "plot_evt_en_on": "EN · on",
        "plot_evt_en_off": "EN · off",
        "plot_evt_tgt": "TGT · target",
        "ms_title": "Mission parameters",
        "ms_alt": "Entry altitude (km MSL)",
        "ms_speed": "Entry speed (km/s)",
        "ms_fuel": "Fuel (kg)",
        "ms_start": "Start mission",
        "ms_defaults": "Restore defaults",
        "ms_hint": "",
        "ms_bad": "Invalid numbers or out-of-range values.",
        "esc_mission_setup": "Start parameters",
        "ms_field_alt_km": "Entry altitude, km MSL",
        "ms_field_speed_kms": "Entry speed, km/s",
        "ms_field_fuel": "Fuel, kg",
        "ms_field_dry": "Dry mass, kg",
        "ms_field_hs": "Heatshield mass, kg",
        "ms_field_a_ref": "Reference area A_ref, m²",
        "ms_field_cd0": "Base Cd (hull)",
        "ms_field_drogue_a": "Drogue chute: area, m²",
        "ms_field_drogue_cd": "Drogue chute: Cd",
        "ms_field_main_a": "Main chute: area, m²",
        "ms_field_main_cd": "Main chute: Cd",
        "ms_field_t_max": "Engine: T_max, N",
        "ms_field_isp": "Engine: Isp, s",
    },
}


_MS_FIELD_ORDER = (
    "alt_km",
    "speed_kms",
    "fuel",
    "dry_kg",
    "hs_kg",
    "a_ref",
    "cd0",
    "drogue_a",
    "drogue_cd",
    "main_a",
    "main_cd",
    "t_max",
    "isp",
)

_MS_FIELD_LABEL_KEYS = {
    "alt_km": "ms_field_alt_km",
    "speed_kms": "ms_field_speed_kms",
    "fuel": "ms_field_fuel",
    "dry_kg": "ms_field_dry",
    "hs_kg": "ms_field_hs",
    "a_ref": "ms_field_a_ref",
    "cd0": "ms_field_cd0",
    "drogue_a": "ms_field_drogue_a",
    "drogue_cd": "ms_field_drogue_cd",
    "main_a": "ms_field_main_a",
    "main_cd": "ms_field_main_cd",
    "t_max": "ms_field_t_max",
    "isp": "ms_field_isp",
}


def _default_mission_field_text() -> dict[str, str]:
    bc = BodyConfig()
    dc = DigitalTwinConfig()
    return {
        "alt_km": f"{bc.entry_start_altitude_m * 1e-3:g}",
        "speed_kms": f"{bc.entry_speed_mps * 1e-3:g}",
        "fuel": "50",
        "dry_kg": "270",
        "hs_kg": "30",
        "a_ref": "2.5",
        "cd0": "1.2",
        "drogue_a": f"{dc.drogue_area_m2:g}",
        "drogue_cd": f"{dc.drogue_cd:g}",
        "main_a": f"{dc.main_chute_area_m2:g}",
        "main_cd": f"{dc.main_chute_cd:g}",
        "t_max": f"{dc.engine.t_max_n:g}",
        "isp": f"{dc.engine.isp_s:g}",
    }


class _MissionReportGeom(NamedTuple):
    panel: pygame.Rect
    close_btn: pygame.Rect
    restart_btn: pygame.Rect
    photo: pygame.Rect
    plot0: pygame.Rect
    plot1: pygame.Rect
    plot2: pygame.Rect
    plot_scroll: pygame.Rect
    plots_union: pygame.Rect


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


_FP_SYNTAX_COLORS: dict[str, tuple[int, int, int]] = {
    "keyword": (188, 140, 230),
    "string": (130, 200, 150),
    "comment": (110, 115, 125),
    "number": (240, 175, 100),
    "api_root": (110, 200, 235),
    "name": (218, 220, 228),
    "op": (200, 205, 215),
    "ws": (218, 220, 228),
}


class _FlightProgramEditorGeom(NamedTuple):
    panel: pygame.Rect
    text_area: pygame.Rect
    hint_panel: pygame.Rect
    template_spec_btn: pygame.Rect
    template_huygens_btn: pygame.Rect
    save_btn: pygame.Rect
    cancel_btn: pygame.Rect


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
        self._fp_font = pygame.font.SysFont("DejaVu Sans Mono", int(15 * self.ui_scale))

        self._restart_requested = False

        self.auto_mode = False
        self._last_log_path: Optional[str] = None

        self.sim_paused: bool = False
        self.show_help: bool = False
        self.esc_menu_open: bool = False
        self.flight_program_editor_open: bool = False
        self.mission_report_open: bool = False
        self.mission_setup_open: bool = True
        self._ms_field_text: dict[str, str] = _default_mission_field_text()
        self._ms_scroll_px: float = 0.0
        self._ms_scroll_drag: bool = False
        self._ms_scroll_grab_y: float = 0.0
        self._ms_focus_idx: int = 0
        # Mission dossier: visible time window on plots (same coords as xs = t - t0).
        self._mission_view_lo: float = 0.0
        self._mission_view_hi: float = 1.0
        self._mission_timeline_reset: bool = True
        self._mission_scroll_drag: bool = False
        self._mission_scroll_grab: Optional[float] = None
        self._quit_requested: bool = False
        self._toast_until_ms: int = 0
        self._toast_key: str = "lever_denied"
        self._toast_custom: Optional[str] = None
        self.stats = {}
        # Set by main; used by renderer-less overlay calls.
        self.controller: Optional[Controller] = None
        self._text_cache: dict[tuple, pygame.Surface] = {}
        self._failure_hud_hover_rect: Optional[pygame.Rect] = None

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

        self.flight_program_source_saved: str = DEFAULT_SCRIPT
        tick0, _e0, g0 = compile_flight_program(DEFAULT_SCRIPT)
        self.flight_program_tick: Optional[Callable[..., None]] = tick0
        self.flight_program_globals: Optional[dict] = g0
        self.flight_program_sleep_until_s: Optional[float] = None
        self._fp_lines: list[str] = [""]
        self._fp_cy: int = 0
        self._fp_cx: int = 0
        self._fp_scroll: int = 0
        self._fp_compile_error: Optional[str] = None
        self._fp_text_input_started: bool = False
        self._fp_hint_scroll: int = 0
        self._fp_sel_mark: Optional[tuple[int, int]] = None
        self._fp_drag_mark: Optional[tuple[int, int]] = None
        self._fp_drag_selecting: bool = False
        self._fp_internal_clipboard: str = ""

    def t(self, key: str) -> str:
        return I18N[self.lang][key]

    def _t_failure_reason(self, category_id: str) -> str:
        """Translate model failure category id (e.g. ``overload``)."""
        k = f"fail_{category_id}"
        tab = I18N[self.lang]
        if k in tab:
            return str(tab[k])
        return category_id

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
        self.flight_program_editor_open = False
        self._fp_stop_text_input()
        self.mission_report_open = False
        self.flight_program_sleep_until_s = None
        self.mission_setup_open = True
        self._ms_focus_idx = 0
        self._ms_scroll_px = 0.0
        self._ms_scroll_drag = False

    def sync_mission_setup_from_model(self, model: PhysicsModel) -> None:
        """Refresh mission setup text fields from `PhysicsModel` mission getters."""
        self._ms_field_text = {
            "alt_km": f"{float(model.mission_entry_start_altitude_m) * 1e-3:g}",
            "speed_kms": f"{float(model.mission_entry_speed_mps) * 1e-3:g}",
            "fuel": f"{float(model.mission_fuel_kg):g}",
            "dry_kg": f"{float(model.mission_dry_mass_kg):g}",
            "hs_kg": f"{float(model.mission_heatshield_mass_kg):g}",
            "a_ref": f"{float(model.mission_a_ref_m2):g}",
            "cd0": f"{float(model.mission_cd_base):g}",
            "drogue_a": f"{float(model.mission_drogue_area_m2):g}",
            "drogue_cd": f"{float(model.mission_drogue_cd):g}",
            "main_a": f"{float(model.mission_main_chute_area_m2):g}",
            "main_cd": f"{float(model.mission_main_chute_cd):g}",
            "t_max": f"{float(model.mission_engine_t_max_n):g}",
            "isp": f"{float(model.mission_engine_isp_s):g}",
        }
        self._ms_scroll_px = 0.0

    def _mission_setup_restore_body_defaults(self) -> None:
        self._ms_field_text = _default_mission_field_text()
        self._ms_scroll_px = 0.0

    def _mission_setup_try_apply(self, c: Controller) -> None:
        def pf(key: str) -> float:
            return float(self._ms_field_text[key].strip().replace(",", "."))

        try:
            alt_km = pf("alt_km")
            spd_kms = pf("speed_kms")
            fuel = pf("fuel")
            dry = pf("dry_kg")
            hs = pf("hs_kg")
            a_ref = pf("a_ref")
            cd0 = pf("cd0")
            da = pf("drogue_a")
            dcd = pf("drogue_cd")
            ma = pf("main_a")
            mcd = pf("main_cd")
            tmax = pf("t_max")
            isp = pf("isp")
        except (ValueError, KeyError):
            self._toast_until_ms = pygame.time.get_ticks() + 4000
            self._toast_custom = self.t("ms_bad")
            return
        try:
            c.model.set_mission_parameters(
                entry_start_altitude_m=alt_km * 1000.0,
                entry_speed_mps=spd_kms * 1000.0,
                fuel_kg=fuel,
                dry_mass_kg=dry,
                heatshield_mass_kg=hs,
                a_ref_m2=a_ref,
                cd_base=cd0,
                drogue_area_m2=da,
                drogue_cd=dcd,
                main_chute_area_m2=ma,
                main_chute_cd=mcd,
                engine_t_max_n=tmax,
                engine_isp_s=isp,
            )
        except ValueError as ex:
            self._toast_until_ms = pygame.time.get_ticks() + 5000
            self._toast_custom = str(ex)[:220]
            return
        c.model.reset()
        self.mission_setup_open = False
        self._toast_custom = None

    def _mission_setup_layout(self) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect, int, int, int, int, int, pygame.Rect, pygame.Rect, pygame.Rect, float]:
        """
        panel, viewport, scroll_track, row_h, gap, label_w, inp_w, pad,
        def_btn, start_btn, lang, max_scroll_px.
        """
        pad = int(16 * self.ui_scale)
        row_h = int(34 * self.ui_scale)
        gap = int(8 * self.ui_scale)
        title_h = int(34 * self.ui_scale)
        hint_h = int(22 * self.ui_scale)
        btn_h = int(38 * self.ui_scale)
        btn_gap = int(10 * self.ui_scale)
        bw = int(min(720 * self.ui_scale, max(420.0, self.visual_rect.width - 80)))
        viewport_h = int(min(320 * self.ui_scale, self.visual_rect.height * 0.42))
        scroll_w = int(12 * self.ui_scale)
        bh = pad + title_h + viewport_h + hint_h + btn_gap + btn_h + pad
        panel = pygame.Rect(0, 0, int(bw), int(bh))
        panel.center = self.visual_rect.center
        inner_w = panel.width - 2 * pad - scroll_w - 6
        label_w = int(inner_w * 0.52)
        inp_w = inner_w - label_w - int(8 * self.ui_scale)
        v_top = panel.top + pad + title_h + int(4 * self.ui_scale)
        viewport = pygame.Rect(panel.left + pad, v_top, inner_w, viewport_h)
        scroll_track = pygame.Rect(viewport.right + 4, viewport.top, scroll_w, viewport.height)
        n = len(_MS_FIELD_ORDER)
        content_h = n * row_h + max(0, n - 1) * gap
        max_scroll = max(0.0, float(content_h - viewport_h))
        btn_y = panel.bottom - pad - btn_h
        btn_w = int(200 * self.ui_scale)
        def_btn = pygame.Rect(panel.left + pad, btn_y, btn_w, btn_h)
        start_btn = pygame.Rect(panel.right - pad - btn_w, btn_y, btn_w, btn_h)
        lang = pygame.Rect(panel.right - pad - int(72 * self.ui_scale), panel.top + pad, int(72 * self.ui_scale), int(28 * self.ui_scale))
        return (
            panel,
            viewport,
            scroll_track,
            row_h,
            gap,
            label_w,
            inp_w,
            pad,
            def_btn,
            start_btn,
            lang,
            max_scroll,
        )

    def _mission_setup_clamp_scroll(self, max_scroll: float) -> None:
        self._ms_scroll_px = float(np.clip(self._ms_scroll_px, 0.0, max(0.0, max_scroll)))

    def _mission_setup_row_index_at(self, pos: tuple[int, int], layout: tuple) -> Optional[int]:
        panel, viewport, scroll_track, row_h, gap, label_w, inp_w, pad, def_btn, start_btn, lang, max_scroll = layout
        if not viewport.collidepoint(pos):
            return None
        sc = float(np.clip(self._ms_scroll_px, 0.0, max_scroll))
        x, y = pos
        for i in range(len(_MS_FIELD_ORDER)):
            ry = viewport.top + i * (row_h + gap) - sc
            rr = pygame.Rect(viewport.left, ry, viewport.width, row_h)
            if rr.collidepoint(x, y):
                return i
        return None

    def _mission_setup_append_char(self, ch: str) -> None:
        if ch not in "0123456789.,":
            return
        ch = "." if ch == "," else ch
        if self._ms_focus_idx < 0 or self._ms_focus_idx >= len(_MS_FIELD_ORDER):
            return
        key = _MS_FIELD_ORDER[self._ms_focus_idx]
        cur = self._ms_field_text.get(key, "")
        if ch == "." and "." in cur:
            return
        self._ms_field_text[key] = cur + ch

    def handle_keydown(self, event: pygame.event.Event, c: Controller) -> bool:
        if event.type != pygame.KEYDOWN:
            return False
        sc = event.scancode
        if self.mission_setup_open:
            if sc == pygame.KSCAN_ESCAPE and self.show_help:
                self.show_help = False
                return True
            if sc == pygame.KSCAN_F1:
                self.show_help = not self.show_help
                return True
            if sc == pygame.KSCAN_TAB:
                self._ms_focus_idx = (self._ms_focus_idx + 1) % len(_MS_FIELD_ORDER)
                return True
            if sc in (pygame.KSCAN_RETURN, pygame.KSCAN_KP_ENTER):
                self._mission_setup_try_apply(c)
                return True
            if sc == pygame.KSCAN_BACKSPACE:
                fk = _MS_FIELD_ORDER[self._ms_focus_idx]
                cur = self._ms_field_text.get(fk, "")
                self._ms_field_text[fk] = cur[:-1]
                return True
            u = event.unicode
            if u:
                self._mission_setup_append_char(u)
                return True
            return True
        if self.flight_program_editor_open:
            if sc == pygame.KSCAN_ESCAPE:
                self._flight_program_cancel()
                return True
            self._flight_program_handle_keydown(event, c)
            return True
        if sc == pygame.KSCAN_ESCAPE:
            if self.show_help:
                self.show_help = False
                return True
            if self.mission_report_open:
                self.mission_report_open = False
                return True
            self._toggle_esc_menu()
            return True
        if sc == pygame.KSCAN_F1:
            self.show_help = not self.show_help
            return True
        if sc in (pygame.KSCAN_SPACE, pygame.KSCAN_PAUSE):
            if self.mission_report_open:
                self.mission_report_open = False
                return True
            self._toggle_esc_menu()
            return True
        if self.mission_report_open and c.model.result != SimResult.RUNNING:
            hist = c.model.telemetry_history
            if len(hist) >= 2:
                t_total = max(1e-9, float(hist[-1]["t_s"]) - float(hist[0]["t_s"]))
                span = max(1e-9, float(self._mission_view_hi) - float(self._mission_view_lo))
                center = 0.5 * (float(self._mission_view_lo) + float(self._mission_view_hi))
                if sc == pygame.KSCAN_LEFTBRACKET:
                    new_span = max(t_total * 1e-4, min(t_total, span * 0.75))
                    self._mission_view_lo = center - 0.5 * new_span
                    self._mission_view_hi = center + 0.5 * new_span
                    self._mission_timeline_clamp(t_total)
                    return True
                if sc == pygame.KSCAN_RIGHTBRACKET:
                    new_span = min(t_total, max(t_total * 1e-4, span * (4.0 / 3.0)))
                    self._mission_view_lo = center - 0.5 * new_span
                    self._mission_view_hi = center + 0.5 * new_span
                    self._mission_timeline_clamp(t_total)
                    return True
        if sc == pygame.KSCAN_R:
            self._request_restart()
            return True
        if sc == pygame.KSCAN_A:
            self.auto_mode = not self.auto_mode
            return True
        if sc in (pygame.KSCAN_EQUALS, pygame.KSCAN_KP_PLUS, pygame.KSCAN_KP_EQUALS):
            self.bump_time_slider(1)
            return True
        if sc in (pygame.KSCAN_MINUS, pygame.KSCAN_KP_MINUS):
            self.bump_time_slider(-1)
            return True
        return False

    def _surface_display(self, raw: str) -> str:
        if raw == "lake":
            return self.t("surface_lake")
        return self.t("surface_land")

    def _esc_menu_layout_full(
        self,
    ) -> tuple[
        pygame.Rect,
        pygame.Rect,
        pygame.Rect,
        pygame.Rect,
        pygame.Rect,
        pygame.Rect,
        pygame.Rect,
        pygame.Rect,
    ]:
        """Order: resume, auto/manual, language, restart, mission params, CSV, flight program, quit."""
        cx = self.visual_rect.centerx
        bw, bh = int(240 * self.ui_scale), int(44 * self.ui_scale)
        gap = int(10 * self.ui_scale)
        n = 8
        total_h = n * bh + (n - 1) * gap
        y0 = self.visual_rect.centery - total_h // 2 + int(28 * self.ui_scale)
        rects: list[pygame.Rect] = []
        for i in range(n):
            r = pygame.Rect(0, 0, bw, bh)
            r.centerx = cx
            r.top = y0 + i * (bh + gap)
            rects.append(r)
        return (rects[0], rects[1], rects[2], rects[3], rects[4], rects[5], rects[6], rects[7])

    def handle_event(self, event: pygame.event.Event, c: Controller) -> None:
        if self.flight_program_editor_open:
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self._fp_sel_mark is not None and (self._fp_cy, self._fp_cx) == self._fp_sel_mark:
                    self._fp_sel_mark = None
                self._fp_drag_selecting = False
                self._fp_drag_mark = None
            if event.type == pygame.TEXTINPUT:
                if pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
                    return
                self._flight_program_insert_text(event.text)
                return
            if event.type == pygame.MOUSEMOTION:
                if self._fp_drag_selecting and event.buttons[0]:
                    g = self._flight_program_editor_geometry()
                    ta = g.text_area
                    p = self._fp_clamp_pos_to_text_area(event.pos, ta)
                    self._flight_program_cursor_from_click(p, ta)
                    if self._fp_drag_mark is not None and (self._fp_cy, self._fp_cx) != self._fp_drag_mark:
                        self._fp_sel_mark = self._fp_drag_mark
                return
            if event.type == pygame.MOUSEWHEEL:
                g = self._flight_program_editor_geometry()
                mp = pygame.mouse.get_pos()
                lh = self._fp_line_height()
                hint_lh = self._fp_hint_row_height()
                dy = int(getattr(event, "y", 0))
                if g.hint_panel.collidepoint(mp):
                    mx = self._fp_hint_max_scroll(g.hint_panel)
                    self._fp_hint_scroll = int(np.clip(self._fp_hint_scroll - dy, 0, mx))
                elif g.text_area.collidepoint(mp):
                    vis = max(1, g.text_area.height // max(1, lh))
                    mx = max(0, len(self._fp_lines) - vis)
                    self._fp_scroll = int(np.clip(self._fp_scroll - dy, 0, mx))
                return
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                g = self._flight_program_editor_geometry()
                if g.template_spec_btn.collidepoint(event.pos):
                    self._flight_program_load_template("spec")
                    return
                if g.template_huygens_btn.collidepoint(event.pos):
                    self._flight_program_load_template("huygens")
                    return
                if g.save_btn.collidepoint(event.pos):
                    self._flight_program_save(c)
                    return
                if g.cancel_btn.collidepoint(event.pos):
                    self._flight_program_cancel()
                    return
                if g.hint_panel.collidepoint(event.pos):
                    pick = self._fp_hint_pick_line(event.pos, g.hint_panel)
                    if pick:
                        self._fp_insert_at_cursor(pick)
                    return
                if g.text_area.collidepoint(event.pos):
                    shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                    old_c = (self._fp_cy, self._fp_cx)
                    self._flight_program_cursor_from_click(event.pos, g.text_area)
                    if shift:
                        if self._fp_sel_mark is None:
                            self._fp_sel_mark = old_c
                    else:
                        self._fp_sel_mark = None
                        self._fp_drag_mark = (self._fp_cy, self._fp_cx)
                        self._fp_drag_selecting = True
                    return
                if g.panel.collidepoint(event.pos):
                    return
                return
            return

        if self.mission_setup_open:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.show_help:
                self.show_help = False
                return
            lay = self._mission_setup_layout()
            (
                panel,
                viewport,
                scroll_track,
                row_h,
                gap,
                label_w,
                inp_w,
                pad,
                def_btn,
                start_btn,
                lang_r,
                max_scroll,
            ) = lay
            viewport_h = float(viewport.height)
            content_h = max_scroll + viewport_h

            if event.type == pygame.MOUSEWHEEL:
                mp = pygame.mouse.get_pos()
                dy = float(getattr(event, "y", 0))
                if viewport.collidepoint(mp) or scroll_track.collidepoint(mp):
                    self._ms_scroll_px -= dy * float(row_h) * 0.65
                    self._mission_setup_clamp_scroll(max_scroll)
                return

            if event.type == pygame.MOUSEMOTION and self._ms_scroll_drag and event.buttons[0]:
                tr = scroll_track
                thumb_h = max(int(18 * self.ui_scale), int(viewport_h * viewport_h / max(content_h, 1.0)))
                travel = max(1.0, float(tr.height - thumb_h))
                new_y = float(event.pos[1]) - float(self._ms_scroll_grab_y)
                frac = float(np.clip((new_y - tr.top) / travel, 0.0, 1.0))
                self._ms_scroll_px = frac * max_scroll
                self._mission_setup_clamp_scroll(max_scroll)
                return

            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._ms_scroll_drag = False
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                idx = self._mission_setup_row_index_at(event.pos, lay)
                if idx is not None:
                    self._ms_focus_idx = idx
                    return
                if max_scroll > 1e-6:
                    tr = scroll_track
                    sc = float(np.clip(self._ms_scroll_px, 0.0, max_scroll))
                    thumb_h = max(int(18 * self.ui_scale), int(viewport_h * viewport_h / max(content_h, 1.0)))
                    travel = max(1.0, float(tr.height - thumb_h))
                    thumb_y = tr.top + (sc / max_scroll) * travel if max_scroll > 1e-6 else tr.top
                    thumb = pygame.Rect(tr.left, int(thumb_y), tr.width, thumb_h)
                    if thumb.collidepoint(event.pos):
                        self._ms_scroll_drag = True
                        self._ms_scroll_grab_y = float(event.pos[1]) - float(thumb_y)
                        return
                    if tr.collidepoint(event.pos) and not thumb.collidepoint(event.pos):
                        my = float(event.pos[1] - tr.top) - 0.5 * thumb_h
                        frac = float(np.clip(my / travel, 0.0, 1.0))
                        self._ms_scroll_px = frac * max_scroll
                        self._mission_setup_clamp_scroll(max_scroll)
                        return
                if def_btn.collidepoint(event.pos):
                    self._mission_setup_restore_body_defaults()
                    return
                if start_btn.collidepoint(event.pos):
                    self._mission_setup_try_apply(c)
                    return
                if lang_r.collidepoint(event.pos):
                    self.lang = "EN" if self.lang == "RU" else "RU"
                    return
                if panel.collidepoint(event.pos):
                    return
                return
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.show_help:
                self.show_help = False
                return
            if self.esc_menu_open:
                r_res, r_mode, r_lang, r_rst, r_ms, r_log, r_fp, r_quit = self._esc_menu_layout_full()
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
                if r_ms.collidepoint(event.pos):
                    self.mission_setup_open = True
                    self.esc_menu_open = False
                    self.sync_mission_setup_from_model(c.model)
                    self._ms_focus_idx = 0
                    return
                if r_log.collidepoint(event.pos):
                    c.queue(Command(request_toggle_csv_logging=True))
                    return
                if r_fp.collidepoint(event.pos):
                    self._flight_program_open_editor()
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

        if self.mission_report_open and c.model.result != SimResult.RUNNING:
            hist = c.model.telemetry_history
            if len(hist) >= 2:
                t0 = float(hist[0]["t_s"])
                t_total = max(1e-9, float(hist[-1]["t_s"]) - t0)
                g = self._mission_report_geometry()
                plots_band = g.plots_union.union(g.plot_scroll)
                mp = pygame.mouse.get_pos()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if g.plot_scroll.collidepoint(event.pos):
                        thumb = self._mission_timeline_thumb_rect(g.plot_scroll, t_total)
                        span = max(1e-9, float(self._mission_view_hi) - float(self._mission_view_lo))
                        rem = max(1e-9, t_total - span)
                        room = max(1, g.plot_scroll.width - thumb.width - 4)
                        if thumb.collidepoint(event.pos):
                            self._mission_scroll_drag = True
                            self._mission_scroll_grab = float(event.pos[0] - thumb.left)
                            return
                        else:
                            u = float(event.pos[0] - g.plot_scroll.left - 2) / max(1, g.plot_scroll.width - 4)
                            u = float(np.clip(u, 0.0, 1.0))
                            self._mission_view_lo = u * (t_total - span)
                            self._mission_view_hi = self._mission_view_lo + span
                            self._mission_timeline_clamp(t_total)
                        return
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._mission_scroll_drag = False
                    self._mission_scroll_grab = None
                if event.type == pygame.MOUSEMOTION and self._mission_scroll_drag and self._mission_scroll_grab is not None:
                    tr = g.plot_scroll
                    thumb = self._mission_timeline_thumb_rect(tr, t_total)
                    span = max(1e-9, float(self._mission_view_hi) - float(self._mission_view_lo))
                    rem = max(1e-9, t_total - span)
                    room = max(1, tr.width - thumb.width - 4)
                    thumb_left = float(event.pos[0]) - self._mission_scroll_grab
                    u = (thumb_left - tr.left - 2) / max(1e-9, float(room))
                    u = float(np.clip(u, 0.0, 1.0))
                    self._mission_view_lo = u * rem
                    self._mission_view_hi = self._mission_view_lo + span
                    self._mission_timeline_clamp(t_total)
                    return
                if event.type == pygame.MOUSEWHEEL and plots_band.collidepoint(mp):
                    dy = int(getattr(event, "y", 0))
                    span = max(1e-9, float(self._mission_view_hi) - float(self._mission_view_lo))
                    center = 0.5 * (float(self._mission_view_lo) + float(self._mission_view_hi))
                    if pygame.key.get_mods() & (pygame.KMOD_CTRL | pygame.KMOD_META):
                        z = 1.12 if dy > 0 else 1.0 / 1.12
                        new_span = max(t_total * 1e-4, min(t_total, span * z))
                        self._mission_view_lo = center - 0.5 * new_span
                        self._mission_view_hi = center + 0.5 * new_span
                    else:
                        pan = 0.06 * span * (-1.0 if dy > 0 else 1.0)
                        self._mission_view_lo += pan
                        self._mission_view_hi += pan
                    self._mission_timeline_clamp(t_total)
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
                    self._mission_timeline_reset = True
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
                    self._toast_custom = None
                return

    def _fp_stop_text_input(self) -> None:
        if self._fp_text_input_started:
            try:
                pygame.key.stop_text_input()
            except Exception:
                pass
            self._fp_text_input_started = False
        try:
            pygame.key.set_repeat(0)
        except Exception:
            pass

    def _fp_start_text_input(self) -> None:
        if not self._fp_text_input_started:
            try:
                pygame.key.start_text_input()
            except Exception:
                pass
            self._fp_text_input_started = True
        try:
            pygame.key.set_repeat(400, 42)
        except Exception:
            pass

    def _on_flight_program_runtime_error(self, msg: str) -> None:
        self.auto_mode = False
        self._toast_until_ms = pygame.time.get_ticks() + 5000
        self._toast_key = "fp_runtime_error"
        self._toast_custom = msg[:220]

    def _flight_program_editor_geometry(self) -> _FlightProgramEditorGeom:
        pad = int(14 * self.ui_scale)
        pw = min(int(self.visual_rect.width * 0.92), int(1120 * self.ui_scale))
        ph = min(int(self.visual_rect.height * 0.88), int(720 * self.ui_scale))
        panel = pygame.Rect(0, 0, pw, ph)
        panel.center = self.visual_rect.center
        btn_w, btn_h = int(132 * self.ui_scale), int(38 * self.ui_scale)
        gap = int(10 * self.ui_scale)
        footer_h = btn_h + pad * 2 + int(26 * self.ui_scale)
        top_y = panel.top + int(56 * self.ui_scale)
        body_h = max(int(120 * self.ui_scale), panel.height - int(56 * self.ui_scale) - footer_h)
        inner_w = panel.width - 2 * pad
        hint_w = min(int(300 * self.ui_scale), max(int(200 * self.ui_scale), int(inner_w * 0.34)))
        split_gap = int(8 * self.ui_scale)
        text_w = max(int(200 * self.ui_scale), inner_w - split_gap - hint_w)
        text_area = pygame.Rect(panel.left + pad, top_y, text_w, body_h)
        hint_panel = pygame.Rect(text_area.right + split_gap, top_y, hint_w, body_h)
        tw = min(int(168 * self.ui_scale), int(inner_w * 0.22))
        tw = max(tw, int(108 * self.ui_scale))
        template_spec_btn = pygame.Rect(panel.left + pad, panel.bottom - pad - btn_h, tw, btn_h)
        template_huygens_btn = pygame.Rect(template_spec_btn.right + gap, template_spec_btn.top, tw, btn_h)
        save_btn = pygame.Rect(panel.right - pad - 2 * btn_w - gap, panel.bottom - pad - btn_h, btn_w, btn_h)
        cancel_btn = pygame.Rect(panel.right - pad - btn_w, panel.bottom - pad - btn_h, btn_w, btn_h)
        return _FlightProgramEditorGeom(
            panel, text_area, hint_panel, template_spec_btn, template_huygens_btn, save_btn, cancel_btn
        )

    def _flight_program_open_editor(self) -> None:
        self.esc_menu_open = False
        self._fp_lines = self.flight_program_source_saved.split("\n")
        if not self._fp_lines:
            self._fp_lines = [""]
        self._fp_cy = 0
        self._fp_cx = 0
        self._fp_scroll = 0
        self._fp_hint_scroll = 0
        self._fp_compile_error = None
        self._fp_sel_mark = None
        self._fp_drag_mark = None
        self._fp_drag_selecting = False
        self.flight_program_editor_open = True
        self._fp_start_text_input()

    def _flight_program_cancel(self) -> None:
        self._fp_lines = self.flight_program_source_saved.split("\n")
        if not self._fp_lines:
            self._fp_lines = [""]
        self._fp_cy = 0
        self._fp_cx = 0
        self._fp_scroll = 0
        self._fp_sel_mark = None
        self._fp_drag_mark = None
        self._fp_drag_selecting = False
        self._fp_compile_error = None
        self.flight_program_editor_open = False
        self._fp_stop_text_input()
        self.esc_menu_open = True

    def _flight_program_save(self, c: Controller) -> None:
        src = "\n".join(self._fp_lines)
        tick, err, g = compile_flight_program(src)
        if err is not None:
            self._fp_compile_error = err
            self._toast_until_ms = pygame.time.get_ticks() + 4500
            self._toast_key = "fp_compile_error"
            self._toast_custom = err[:220]
            return
        verr = validate_flight_program_tick(tick, c.model, self.stats, g)
        if verr is not None:
            self._fp_compile_error = f"{self.t('fp_validate_error')}: {verr}"
            self._toast_until_ms = pygame.time.get_ticks() + 5000
            self._toast_key = "fp_validate_error"
            self._toast_custom = verr[:220]
            return
        self.flight_program_source_saved = src
        self.flight_program_tick = tick
        self.flight_program_globals = g
        self.flight_program_sleep_until_s = None
        self._fp_compile_error = None
        self._toast_custom = None
        self.flight_program_editor_open = False
        self._fp_stop_text_input()
        self.esc_menu_open = True

    def _fp_expand(self, s: str) -> str:
        return s.replace("\t", "    ")

    def _fp_text_width_prefix(self, line: str, idx: int) -> int:
        return self._fp_font.size(self._fp_expand(line[:idx]))[0]

    def _fp_line_height(self) -> int:
        return self._fp_font.get_height() + int(4 * self.ui_scale)

    def _fp_ensure_cursor_in_view(self, text_area: pygame.Rect) -> None:
        lh = self._fp_line_height()
        vis = max(1, text_area.height // max(1, lh))
        self._fp_cy = int(np.clip(self._fp_cy, 0, max(0, len(self._fp_lines) - 1)))
        line = self._fp_lines[self._fp_cy]
        self._fp_cx = int(np.clip(self._fp_cx, 0, len(line)))
        if self._fp_cy < self._fp_scroll:
            self._fp_scroll = self._fp_cy
        if self._fp_cy >= self._fp_scroll + vis:
            self._fp_scroll = self._fp_cy - vis + 1
        self._fp_scroll = int(np.clip(self._fp_scroll, 0, max(0, len(self._fp_lines) - 1)))

    def _fp_col_at_pixel(self, line: str, x_px: int, text_left: int) -> int:
        rel = x_px - text_left
        if rel <= 0:
            return 0
        lo, hi = 0, len(line)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._fp_text_width_prefix(line, mid) <= rel:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _fp_hint_entries(self) -> list[tuple[str, bool]]:
        rows: list[tuple[str, bool]] = [(self.t("fp_header_sim"), True)]
        rows.extend((s, False) for s in SIM_API_COMPLETIONS)
        rows.append((self.t("fp_header_ap"), True))
        rows.extend((a, False) for a in AP_API_COMPLETIONS)
        return rows

    def _fp_hint_row_height(self) -> int:
        return self.font_small.get_height() + int(4 * self.ui_scale)

    def _fp_hint_max_scroll(self, hint_panel: pygame.Rect) -> int:
        entries = len(self._fp_hint_entries())
        list_top = int(48 * self.ui_scale)
        vis = max(1, (hint_panel.height - list_top) // max(1, self._fp_hint_row_height()))
        return max(0, entries - vis)

    def _fp_hint_pick_line(self, pos: tuple[int, int], hint_panel: pygame.Rect) -> Optional[str]:
        entries = self._fp_hint_entries()
        row_h = self._fp_hint_row_height()
        list_top = hint_panel.top + int(48 * self.ui_scale)
        if pos[1] < list_top or pos[1] >= hint_panel.bottom - int(4 * self.ui_scale):
            return None
        rel = pos[1] - list_top + self._fp_hint_scroll * row_h
        idx = int(rel // max(1, row_h))
        if 0 <= idx < len(entries):
            text, is_hdr = entries[idx]
            if not is_hdr:
                return text
        return None

    def _fp_try_tab_complete(self) -> bool:
        if self._fp_has_selection():
            self._fp_delete_selection()
        line = self._fp_lines[self._fp_cy]
        bef = line[: self._fp_cx]
        mo = re.search(r"((?:sim|ap)\.[\w]*)$", bef)
        if not mo:
            return False
        prefix_full = mo.group(1)
        pool = list(SIM_API_COMPLETIONS) + list(AP_API_COMPLETIONS)
        cands = [x for x in pool if x.startswith(prefix_full)]
        if not cands:
            return False

        def lcp_str(strs: list[str]) -> str:
            p = strs[0]
            for s in strs[1:]:
                while p and not s.startswith(p):
                    p = p[:-1]
            return p

        lcp = lcp_str(cands)
        if len(lcp) <= len(prefix_full):
            return False
        start = mo.start(1)
        new_bef = bef[:start] + lcp
        self._fp_lines[self._fp_cy] = new_bef + line[self._fp_cx :]
        self._fp_cx = len(new_bef)
        self._fp_sel_mark = None
        g = self._flight_program_editor_geometry()
        self._fp_ensure_cursor_in_view(g.text_area)
        return True

    def _fp_insert_at_cursor(self, insertion: str) -> None:
        if self._fp_has_selection():
            self._fp_delete_selection()
        line = self._fp_lines[self._fp_cy]
        self._fp_lines[self._fp_cy] = line[: self._fp_cx] + insertion + line[self._fp_cx :]
        self._fp_cx += len(insertion)
        self._fp_sel_mark = None
        g = self._flight_program_editor_geometry()
        self._fp_ensure_cursor_in_view(g.text_area)

    def _fp_selection_carets_ordered(self) -> Optional[tuple[int, int, int, int]]:
        """Half-open range between two carets: (sy, sx, ey, ex), ex exclusive on last line."""
        if self._fp_sel_mark is None:
            return None
        a, b = self._fp_sel_mark, (self._fp_cy, self._fp_cx)
        if a == b:
            return None
        if a <= b:
            return (a[0], a[1], b[0], b[1])
        return (b[0], b[1], a[0], a[1])

    def _fp_has_selection(self) -> bool:
        return self._fp_selection_carets_ordered() is not None

    def _fp_delete_selection(self) -> None:
        b = self._fp_selection_carets_ordered()
        if not b:
            return
        sy, sx, ey, ex = b
        if sy == ey:
            ln = self._fp_lines[sy]
            self._fp_lines[sy] = ln[:sx] + ln[ex:]
        else:
            merged = self._fp_lines[sy][:sx] + self._fp_lines[ey][ex:]
            self._fp_lines[sy : ey + 1] = [merged]
        self._fp_cy, self._fp_cx = sy, sx
        self._fp_sel_mark = None
        g = self._flight_program_editor_geometry()
        self._fp_ensure_cursor_in_view(g.text_area)

    def _fp_get_selection_text(self) -> str:
        b = self._fp_selection_carets_ordered()
        if not b:
            return ""
        sy, sx, ey, ex = b
        if sy == ey:
            return self._fp_lines[sy][sx:ex]
        parts = [self._fp_lines[sy][sx:]]
        for i in range(sy + 1, ey):
            parts.append(self._fp_lines[i])
        parts.append(self._fp_lines[ey][:ex])
        return "\n".join(parts)

    def _fp_clipboard_set(self, text: str) -> None:
        try:
            if not pygame.scrap.get_init():
                pygame.scrap.init()
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
        except Exception:
            pass
        self._fp_internal_clipboard = text

    def _fp_clipboard_get(self) -> str:
        try:
            if pygame.scrap.get_init():
                raw = pygame.scrap.get(pygame.SCRAP_TEXT)
                if raw:
                    if isinstance(raw, bytes):
                        return raw.decode("utf-8", errors="replace")
                    return str(raw)
        except Exception:
            pass
        return self._fp_internal_clipboard

    def _flight_program_load_template(self, which: str) -> None:
        """Load built-in script into the editor (does not save). `which`: 'spec' | 'huygens'."""
        src = DEFAULT_SCRIPT if which == "spec" else HUYGENS_SCRIPT
        self._fp_lines = src.split("\n")
        if not self._fp_lines:
            self._fp_lines = [""]
        self._fp_cy = 0
        self._fp_cx = 0
        self._fp_scroll = 0
        self._fp_sel_mark = None
        self._fp_drag_mark = None
        self._fp_drag_selecting = False
        self._fp_compile_error = None
        g = self._flight_program_editor_geometry()
        self._fp_ensure_cursor_in_view(g.text_area)

    def _fp_clamp_pos_to_text_area(self, pos: tuple[int, int], text_area: pygame.Rect) -> tuple[int, int]:
        return (
            int(np.clip(pos[0], text_area.left + 1, text_area.right - 2)),
            int(np.clip(pos[1], text_area.top + 1, text_area.bottom - 2)),
        )

    def _flight_program_cursor_from_click(self, pos: tuple[int, int], text_area: pygame.Rect) -> None:
        lh = self._fp_line_height()
        rel_y = pos[1] - text_area.top
        line_idx = self._fp_scroll + max(0, rel_y // max(1, lh))
        line_idx = int(np.clip(line_idx, 0, max(0, len(self._fp_lines) - 1)))
        self._fp_cy = line_idx
        self._fp_cx = self._fp_col_at_pixel(self._fp_lines[line_idx], pos[0], text_area.left + int(6 * self.ui_scale))
        self._fp_ensure_cursor_in_view(text_area)

    def _fp_draw_selection_highlight(
        self, surf: pygame.Surface, ta: pygame.Rect, line_idx: int, y: int, line: str, lh: int, tx: int
    ) -> None:
        b = self._fp_selection_carets_ordered()
        if not b:
            return
        sy, sx, ey, ex = b
        if line_idx < sy or line_idx > ey:
            return
        if line_idx == sy == ey:
            c0, c1 = sx, ex
        elif line_idx == sy:
            c0, c1 = sx, len(line)
        elif line_idx == ey:
            c0, c1 = 0, ex
        else:
            c0, c1 = 0, len(line)
        if c0 >= c1:
            return
        x0 = tx + self._fp_text_width_prefix(line, c0)
        x1 = tx + self._fp_text_width_prefix(line, c1)
        sel_rect = pygame.Rect(x0, y + 1, max(1, x1 - x0), lh - 2)
        pygame.draw.rect(surf, (55, 65, 95), sel_rect, border_radius=2)

    def _flight_program_insert_text(self, text: str) -> None:
        if not text or text == "\r":
            return
        if self._fp_has_selection():
            self._fp_delete_selection()
        for ch in text:
            if ch == "\n":
                self._flight_program_split_line()
            elif ch == "\r":
                continue
            else:
                line = self._fp_lines[self._fp_cy]
                self._fp_lines[self._fp_cy] = line[: self._fp_cx] + ch + line[self._fp_cx :]
                self._fp_cx += 1
        self._fp_sel_mark = None
        g = self._flight_program_editor_geometry()
        self._fp_ensure_cursor_in_view(g.text_area)

    def _flight_program_split_line(self) -> None:
        if self._fp_has_selection():
            self._fp_delete_selection()
        line = self._fp_lines[self._fp_cy]
        tail = line[self._fp_cx :]
        self._fp_lines[self._fp_cy] = line[: self._fp_cx]
        self._fp_lines.insert(self._fp_cy + 1, tail)
        self._fp_cy += 1
        self._fp_cx = 0
        self._fp_sel_mark = None

    def _flight_program_handle_keydown(self, event: pygame.event.Event, c: Controller) -> None:
        g = self._flight_program_editor_geometry()
        ta = g.text_area
        mod = event.mod
        ctrl = bool(mod & pygame.KMOD_CTRL) or bool(mod & pygame.KMOD_META)
        shift = bool(mod & pygame.KMOD_SHIFT)
        sc = event.scancode

        if ctrl and sc == pygame.KSCAN_A:
            self._fp_sel_mark = (0, 0)
            self._fp_cy = len(self._fp_lines) - 1
            self._fp_cx = len(self._fp_lines[self._fp_cy])
            self._fp_ensure_cursor_in_view(ta)
            return
        if ctrl and sc == pygame.KSCAN_C:
            txt = self._fp_get_selection_text()
            if txt:
                self._fp_clipboard_set(txt)
            return
        if ctrl and sc == pygame.KSCAN_X:
            txt = self._fp_get_selection_text()
            if txt:
                self._fp_clipboard_set(txt)
                self._fp_delete_selection()
            return
        if ctrl and sc == pygame.KSCAN_V:
            self._flight_program_insert_text(self._fp_clipboard_get())
            return

        if sc in (pygame.KSCAN_RETURN, pygame.KSCAN_KP_ENTER):
            self._flight_program_split_line()
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_TAB:
            if self._fp_try_tab_complete():
                return
            self._flight_program_insert_text("    ")
            return
        if sc == pygame.KSCAN_BACKSPACE:
            if self._fp_has_selection():
                self._fp_delete_selection()
            else:
                line = self._fp_lines[self._fp_cy]
                if self._fp_cx > 0:
                    self._fp_lines[self._fp_cy] = line[: self._fp_cx - 1] + line[self._fp_cx :]
                    self._fp_cx -= 1
                elif self._fp_cy > 0:
                    prev = self._fp_lines[self._fp_cy - 1]
                    self._fp_cx = len(prev)
                    self._fp_lines[self._fp_cy - 1] = prev + line
                    del self._fp_lines[self._fp_cy]
                    self._fp_cy -= 1
            self._fp_sel_mark = None
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_DELETE:
            if self._fp_has_selection():
                self._fp_delete_selection()
            else:
                line = self._fp_lines[self._fp_cy]
                if self._fp_cx < len(line):
                    self._fp_lines[self._fp_cy] = line[: self._fp_cx] + line[self._fp_cx + 1 :]
                elif self._fp_cy < len(self._fp_lines) - 1:
                    self._fp_lines[self._fp_cy] = line + self._fp_lines[self._fp_cy + 1]
                    del self._fp_lines[self._fp_cy + 1]
            self._fp_sel_mark = None
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_LEFT:
            if shift and not ctrl and self._fp_sel_mark is None:
                self._fp_sel_mark = (self._fp_cy, self._fp_cx)
            elif not shift:
                self._fp_sel_mark = None
            if self._fp_cx > 0:
                self._fp_cx -= 1
            elif self._fp_cy > 0:
                self._fp_cy -= 1
                self._fp_cx = len(self._fp_lines[self._fp_cy])
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_RIGHT:
            if shift and not ctrl and self._fp_sel_mark is None:
                self._fp_sel_mark = (self._fp_cy, self._fp_cx)
            elif not shift:
                self._fp_sel_mark = None
            line = self._fp_lines[self._fp_cy]
            if self._fp_cx < len(line):
                self._fp_cx += 1
            elif self._fp_cy < len(self._fp_lines) - 1:
                self._fp_cy += 1
                self._fp_cx = 0
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_UP:
            if shift and not ctrl and self._fp_sel_mark is None:
                self._fp_sel_mark = (self._fp_cy, self._fp_cx)
            elif not shift:
                self._fp_sel_mark = None
            if self._fp_cy > 0:
                self._fp_cy -= 1
                self._fp_cx = min(self._fp_cx, len(self._fp_lines[self._fp_cy]))
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_DOWN:
            if shift and not ctrl and self._fp_sel_mark is None:
                self._fp_sel_mark = (self._fp_cy, self._fp_cx)
            elif not shift:
                self._fp_sel_mark = None
            if self._fp_cy < len(self._fp_lines) - 1:
                self._fp_cy += 1
                self._fp_cx = min(self._fp_cx, len(self._fp_lines[self._fp_cy]))
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_HOME:
            if shift and not ctrl and self._fp_sel_mark is None:
                self._fp_sel_mark = (self._fp_cy, self._fp_cx)
            elif not shift:
                self._fp_sel_mark = None
            self._fp_cx = 0
            if ctrl and self._fp_cy > 0:
                self._fp_cy = 0
            self._fp_ensure_cursor_in_view(ta)
            return
        if sc == pygame.KSCAN_END:
            if shift and not ctrl and self._fp_sel_mark is None:
                self._fp_sel_mark = (self._fp_cy, self._fp_cx)
            elif not shift:
                self._fp_sel_mark = None
            if ctrl:
                self._fp_cy = len(self._fp_lines) - 1
            self._fp_cx = len(self._fp_lines[self._fp_cy])
            self._fp_ensure_cursor_in_view(ta)
            return
        if ctrl and sc == pygame.KSCAN_S:
            self._flight_program_save(c)
            return

    def _fp_draw_highlighted_line(self, surf: pygame.Surface, x: int, y: int, line: str) -> None:
        font = self._fp_font
        x0 = x
        for text, kind in iter_flight_program_tokens(line):
            seg = text.replace("\t", "    ")
            if not seg:
                continue
            col = _FP_SYNTAX_COLORS.get(kind, (218, 220, 228))
            surf.blit(font.render(seg, True, col), (x0, y))
            x0 += font.size(seg)[0]

    def _draw_flight_program_editor(self, surf: pygame.Surface) -> None:
        g = self._flight_program_editor_geometry()
        dim = pygame.Surface(self.visual_rect.size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 200))
        surf.blit(dim, (0, 0))
        pygame.draw.rect(surf, (32, 34, 42), g.panel, border_radius=12)
        pygame.draw.rect(surf, (140, 145, 165), g.panel, width=2, border_radius=12)
        title = self._cached_render(self.font_big, self.t("fp_title"), (240, 240, 245))
        surf.blit(title, (g.panel.left + int(16 * self.ui_scale), g.panel.top + int(12 * self.ui_scale)))
        hint = self._cached_render(self.font_small, self.t("fp_hint"), (170, 175, 190))
        surf.blit(hint, (g.panel.left + int(16 * self.ui_scale), g.panel.top + int(40 * self.ui_scale)))

        ta = g.text_area
        pygame.draw.rect(surf, (18, 18, 22), ta, border_radius=8)
        pygame.draw.rect(surf, (90, 92, 105), ta, width=1, border_radius=8)
        clip = surf.get_clip()
        surf.set_clip(ta.inflate(-4, -4))
        lh = self._fp_line_height()
        line0 = self._fp_scroll
        vis = (ta.height - int(8 * self.ui_scale)) // max(1, lh) + 1
        y0 = ta.top + int(4 * self.ui_scale)
        tx = ta.left + int(6 * self.ui_scale)
        for j, i in enumerate(range(line0, min(len(self._fp_lines), line0 + vis + 1))):
            ln_i = self._fp_lines[i]
            self._fp_draw_selection_highlight(surf, ta, i, y0 + j * lh, ln_i, lh, tx)
            self._fp_draw_highlighted_line(surf, tx, y0 + j * lh, ln_i)
        cur_line = self._fp_lines[self._fp_cy]
        cx_px = tx + self._fp_text_width_prefix(cur_line, self._fp_cx)
        cy_px = y0 + (self._fp_cy - self._fp_scroll) * lh
        if (pygame.time.get_ticks() // 500) % 2 == 0 and ta.inflate(-2, -2).collidepoint(cx_px, cy_px + lh // 2):
            pygame.draw.line(surf, (250, 200, 120), (cx_px, cy_px + 2), (cx_px, cy_px + lh - 4), 2)
        surf.set_clip(clip)

        hp = g.hint_panel
        pygame.draw.rect(surf, (22, 24, 30), hp, border_radius=8)
        pygame.draw.rect(surf, (85, 90, 105), hp, width=1, border_radius=8)
        ht = self._cached_render(self.font_small, self.t("fp_hints_title"), (210, 215, 225))
        surf.blit(ht, (hp.left + int(8 * self.ui_scale), hp.top + int(6 * self.ui_scale)))
        doc = self._cached_render(self.font_small, self.t("fp_doc_ref"), (130, 150, 175))
        surf.blit(doc, (hp.left + int(8 * self.ui_scale), hp.top + int(22 * self.ui_scale)))
        hclip = surf.get_clip()
        surf.set_clip(hp.inflate(-6, -6))
        entries = self._fp_hint_entries()
        row_h = self._fp_hint_row_height()
        list_y0 = hp.top + int(48 * self.ui_scale)
        first = self._fp_hint_scroll
        max_rows = int((hp.bottom - list_y0 - int(6 * self.ui_scale)) / max(1, row_h)) + 1
        for row_i in range(first, min(len(entries), first + max_rows)):
            text, is_hdr = entries[row_i]
            yy = list_y0 + (row_i - first) * row_h
            col = (200, 190, 120) if is_hdr else (175, 210, 235)
            lbl = self._cached_render(self.font_small, text, col)
            surf.blit(lbl, (hp.left + int(8 * self.ui_scale), yy))
        surf.set_clip(hclip)
        if self._fp_hint_max_scroll(hp) > 0:
            sb = f"↕ {self._fp_hint_scroll + 1}/{self._fp_hint_max_scroll(hp) + 1}"
            surf.blit(self._cached_render(self.font_small, sb, (120, 125, 140)), (hp.right - int(70 * self.ui_scale), hp.bottom - int(16 * self.ui_scale)))

        def pill(rect: pygame.Rect, label: str, hot: bool) -> None:
            bgc = (52, 58, 72) if hot else (40, 42, 50)
            pygame.draw.rect(surf, bgc, rect, border_radius=8)
            pygame.draw.rect(surf, (160, 165, 180), rect, width=2, border_radius=8)
            t = self._cached_render(self.font_small, label, (235, 235, 240))
            surf.blit(t, t.get_rect(center=rect.center))

        pill(g.template_spec_btn, self.t("fp_reset_default"), True)
        pill(g.template_huygens_btn, self.t("fp_reset_huygens"), True)
        pill(g.save_btn, self.t("fp_save"), True)
        pill(g.cancel_btn, self.t("fp_cancel"), True)
        if self._fp_compile_error:
            err_s = self._cached_render(self.font_small, self._fp_compile_error[:200], (230, 120, 110))
            surf.blit(err_s, (g.panel.left + int(14 * self.ui_scale), g.save_btn.top - int(22 * self.ui_scale)))

    def apply_continuous_controls(self, c: Controller) -> None:
        c.queue(Command(throttle_0_1=self.throttle_slider.value if c.model.engine_on else 0.0))

        if self.auto_mode and c.model.result == SimResult.RUNNING:
            FlightProgramRunner.run(self, c)

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
            "t_hs_skin": float(c.model.heatshield_skin_temp_c),
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
        h_km = float(max(0.0, h_km))
        if h_km <= 20.0:
            return 0.55 * (h_km / 20.0)
        if h_km <= 200.0:
            return 0.55 + 0.30 * ((h_km - 20.0) / 180.0)
        span = max(1.0, _ALT_H_KM_TOP - 200.0)
        return 0.85 + 0.15 * (min(h_km, _ALT_H_KM_TOP) - 200.0) / span

    def _h_from_bar_value(self, t: float) -> float:
        # Inverse of _h_bar_value(). Input t is 0..1, output km.
        t = float(np.clip(t, 0.0, 1.0))
        if t <= 0.55:
            return 20.0 * (t / 0.55)
        if t <= 0.85:
            return 20.0 + 180.0 * ((t - 0.55) / 0.30)
        span = max(1.0, _ALT_H_KM_TOP - 200.0)
        return 200.0 + span * ((t - 0.85) / 0.15)

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
        for km in [0.0, 2.0, 20.0, 200.0, _ALT_H_KM_TOP]:
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
        *,
        skin_entry: bool = False,
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
        if skin_entry:
            if temp_c < -100:
                col = (90, 160, 255)
            elif temp_c < 200:
                col = (130, 220, 160)
            elif temp_c < 700:
                col = (255, 150, 60)
            elif temp_c < 1400:
                col = (255, 210, 90)
            else:
                col = (255, 245, 220)
        elif temp_c < -120:
            col = (80, 170, 255)   # cold
        elif temp_c < -30:
            col = (120, 230, 190)  # ok
        else:
            col = (255, 120, 70)   # hot
        pygame.draw.rect(surf, col, fill, border_radius=6)
        pygame.draw.circle(surf, col, bulb_c, max(3, bulb_r - int(3 * self.ui_scale)))

        # ticks
        tick_vals = (
            [-200, 0, 400, 800, 1200, 1800]
            if skin_entry
            else [-200, -180, -150, -120, -90, -60, -30, 0]
        )
        for tv in tick_vals:
            if tv < vmin or tv > vmax:
                continue
            tt = (tv - vmin) / max(1e-9, (vmax - vmin))
            yy = tube.bottom - (pad + 1) - int(np.clip(tt, 0.0, 1.0) * (tube.height - 2 * pad - 2))
            pygame.draw.line(surf, (180, 180, 190), (tube.right + 4, yy), (tube.right + 8, yy), 1)

        lab = self._cached_render(self.font_small, label, (220, 220, 230))
        surf.blit(lab, (rect.left + 10, rect.top + 8))
        val = self._cached_render(self.font, f"{temp_c:.1f} C", (240, 240, 245))
        surf.blit(val, (rect.left + 10, rect.bottom - val.get_height() - 8))

    def _draw_overlay_hud(self, surf: pygame.Surface, c: Controller) -> None:
        # HUD only (instruments, map chrome, controls). Modals draw separately after minimap.
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
        t_hs_skin = float(self.stats.get("t_hs_skin", -150.0))
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
        # Center each gauge under its dial; middle gauge between dials — no horizontal inset
        # (inset pulled the outer two inward and overlapped the heatshield thermometer).
        therm1.center = (dial1_c[0], center_y)
        therm2.center = (dial2_c[0], center_y)
        therm_hs_w = int(np.clip(56 * self.ui_scale, 50 * self.ui_scale, 64 * self.ui_scale))
        therm_hs = pygame.Rect(0, 0, therm_hs_w, therm_h)
        therm_hs.center = ((dial1_c[0] + dial2_c[0]) // 2, center_y)

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
        if not c.model.heatshield_jettisoned:
            thermo_bg = thermo_bg.union(therm_hs)
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
        if not c.model.heatshield_jettisoned:
            self._draw_thermometer(
                surf,
                therm_hs,
                self.t("t_hs_skin"),
                t_hs_skin,
                vmin=-200.0,
                vmax=2000.0,
                skin_entry=True,
            )
        self._draw_thermometer(surf, therm2, self.t("t_int"), t_int)

        # Digital G-load display (replaces text block)
        pygame.draw.rect(surf, (10, 10, 12), g_rect, border_radius=10)
        pygame.draw.rect(surf, (120, 120, 135), g_rect, width=2, border_radius=10)
        pygame.draw.rect(surf, (40, 40, 45), g_rect.inflate(-6, -6), width=1, border_radius=9)
        g_label = self._cached_render(self.font_small, self.t("g"), (220, 220, 230))
        surf.blit(g_label, (g_rect.left + int(10 * self.ui_scale), g_rect.top + int(6 * self.ui_scale)))
        g_txt = self._cached_render(self.font_mono, f"{g_load:05.2f} g", (120, 230, 190))
        surf.blit(g_txt, g_txt.get_rect(center=(g_rect.centerx, g_rect.centery + int(8 * self.ui_scale))))

        self._draw_controls(surf, c)
        self._draw_mission_dossier_button(surf, c)

    def draw_failure_outcome(self, surf: pygame.Surface, c: Controller) -> None:
        """Success / failure banner and reason list; drawn after pause so tooltips are not covered."""
        self._failure_hud_hover_rect = None
        if c.model.result == SimResult.SUCCESS:
            msg = self._cached_render(self.font_big, self.t("success"), (80, 220, 120))
            surf.blit(msg, (self.visual_rect.right - msg.get_width() - 16, self.visual_rect.top + 16))
            return
        if not bool(c.model.failed):
            return
        msg = self._cached_render(self.font_big, self.t("failure"), (230, 80, 80))
        mx = self.visual_rect.right - msg.get_width() - 16
        my = self.visual_rect.top + 16
        surf.blit(msg, (mx, my))
        msg_rect = pygame.Rect(mx, my, msg.get_width(), msg.get_height())
        reason_key = str(c.model.failure_reason)
        union_r = msg_rect
        if reason_key:
            reason_txt = self._cached_render(
                self.font_small, self._t_failure_reason(reason_key), (230, 190, 90)
            )
            rx = self.visual_rect.right - reason_txt.get_width() - 16
            ry = self.visual_rect.top + 50
            surf.blit(reason_txt, (rx, ry))
            union_r = msg_rect.union(pygame.Rect(rx, ry, reason_txt.get_width(), reason_txt.get_height()))
        self._failure_hud_hover_rect = union_r
        keys = tuple(c.model.failure_reasons)
        lines = [self._t_failure_reason(k) for k in keys]
        if len(lines) > 1 and self._failure_hud_hover_rect.collidepoint(pygame.mouse.get_pos()):
            pad = int(8 * self.ui_scale)
            line_h = int(16 * self.ui_scale)
            tw = max(self._cached_render(self.font_small, s, (220, 215, 200)).get_width() for s in lines)
            th = len(lines) * line_h + 2 * pad
            tip = pygame.Surface((tw + 2 * pad, th), pygame.SRCALPHA)
            tip.fill((18, 16, 14, 245))
            pygame.draw.rect(tip, (120, 100, 70), tip.get_rect(), width=1, border_radius=8)
            for i, line in enumerate(lines):
                ln = self._cached_render(self.font_small, line, (220, 210, 195))
                tip.blit(ln, (pad, pad + i * line_h))
            tw_total = tw + 2 * pad
            margin = int(8 * self.ui_scale)
            vr = self.visual_rect
            tx = min(self._failure_hud_hover_rect.left, vr.right - tw_total - margin)
            tx = max(margin, tx)
            ty_preferred = float(self._failure_hud_hover_rect.bottom + margin)
            max_ty = float(vr.bottom - th - margin)
            min_ty = float(vr.top + margin)
            if max_ty >= min_ty:
                ty = float(np.clip(ty_preferred, min_ty, max_ty))
            else:
                ty = min_ty
            surf.blit(tip, (int(tx), int(ty)))

    def draw_overlay(self, surf: pygame.Surface, c: Controller) -> None:
        self._draw_overlay_hud(surf, c)
        self._draw_modal_overlays(surf, c)
        self.draw_failure_outcome(surf, c)

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

        if self.esc_menu_open:
            self._draw_esc_menu(surf, c)

        if self.flight_program_editor_open:
            self._draw_flight_program_editor(surf)

        if self.mission_setup_open:
            self._draw_mission_setup(surf, c)

        if self.show_help:
            self._draw_help_panel(surf)

        if now < self._toast_until_ms:
            toast_txt = self._toast_custom if self._toast_custom else self.t(self._toast_key)
            toast = self._cached_render(self.font_small, toast_txt, (255, 200, 120))
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
        r_res, r_mode, r_lang, r_rst, r_ms, r_log, r_fp, r_quit = self._esc_menu_layout_full()
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
        pill(r_ms, self.t("esc_mission_setup"), True)
        pill(r_log, self.t("csv_log_on") if log_on else self.t("csv_log_off"), log_on)
        pill(r_fp, self.t("esc_flight_program"), True)
        pill(r_quit, self.t("esc_quit"), True)

    def _draw_mission_setup(self, surf: pygame.Surface, c: Controller) -> None:
        dim = pygame.Surface(self.visual_rect.size, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 185))
        surf.blit(dim, (0, 0))
        lay = self._mission_setup_layout()
        (
            panel,
            viewport,
            scroll_track,
            row_h,
            gap,
            label_w,
            inp_w,
            pad,
            def_btn,
            start_btn,
            lang_r,
            max_scroll,
        ) = lay
        viewport_h = float(viewport.height)
        content_h = max_scroll + viewport_h
        sc = float(np.clip(self._ms_scroll_px, 0.0, max_scroll))

        bg = pygame.Surface(panel.size, pygame.SRCALPHA)
        bg.fill((14, 16, 22, 245))
        surf.blit(bg, panel.topleft)
        pygame.draw.rect(surf, (95, 100, 120), panel, width=2, border_radius=14)
        title = self._cached_render(self.font_big, self.t("ms_title"), (240, 240, 245))
        surf.blit(title, (panel.left + int(16 * self.ui_scale), panel.top + int(14 * self.ui_scale)))
        lang_t = self._cached_render(self.font_small, self.t("lang"), (210, 210, 220))
        pygame.draw.rect(surf, (42, 44, 52), lang_r, border_radius=8)
        pygame.draw.rect(surf, (130, 135, 150), lang_r, width=1, border_radius=8)
        surf.blit(lang_t, lang_t.get_rect(center=lang_r.center))

        pygame.draw.rect(surf, (26, 28, 34), viewport, border_radius=8)
        pygame.draw.rect(surf, (75, 80, 92), viewport, width=1, border_radius=8)
        prev_clip = surf.get_clip()
        surf.set_clip(viewport)

        for i, fk in enumerate(_MS_FIELD_ORDER):
            ry = viewport.top + i * (row_h + gap) - sc
            if ry + row_h < viewport.top or ry > viewport.bottom:
                continue
            lb = self.t(_MS_FIELD_LABEL_KEYS[fk])
            lt = self._cached_render(self.font_small, lb, (200, 200, 210))
            surf.blit(lt, (viewport.left + 4, ry + (row_h - lt.get_height()) // 2))
            fr = pygame.Rect(viewport.left + label_w + 6, ry, inp_w, row_h)
            hot = i == self._ms_focus_idx
            pygame.draw.rect(surf, (28, 30, 38) if hot else (22, 24, 30), fr, border_radius=8)
            pygame.draw.rect(surf, (130, 145, 175) if hot else (80, 85, 95), fr, width=2, border_radius=8)
            val = self._ms_field_text.get(fk, "")
            vt = self._cached_render(self.font_mono, val if val else " ", (235, 238, 245))
            surf.blit(vt, (fr.left + int(8 * self.ui_scale), fr.centery - vt.get_height() // 2))

        surf.set_clip(prev_clip)

        if max_scroll > 1e-3:
            pygame.draw.rect(surf, (32, 34, 42), scroll_track, border_radius=6)
            pygame.draw.rect(surf, (70, 75, 88), scroll_track, width=1, border_radius=6)
            thumb_h = max(int(18 * self.ui_scale), int(viewport_h * viewport_h / max(content_h, 1.0)))
            travel = max(1.0, float(scroll_track.height - thumb_h))
            thumb_y = scroll_track.top + (sc / max_scroll) * travel if max_scroll > 1e-6 else float(scroll_track.top)
            thumb = pygame.Rect(scroll_track.left, int(thumb_y), scroll_track.width, thumb_h)
            pygame.draw.rect(surf, (95, 100, 118), thumb, border_radius=5)

        def pill_btn(rect: pygame.Rect, label: str) -> None:
            pygame.draw.rect(surf, (48, 52, 62), rect, border_radius=10)
            pygame.draw.rect(surf, (150, 155, 170), rect, width=2, border_radius=10)
            t = self._cached_render(self.font_small, label, (235, 235, 240))
            surf.blit(t, t.get_rect(center=rect.center))

        pill_btn(def_btn, self.t("ms_defaults"))
        pill_btn(start_btn, self.t("ms_start"))
        hint = self._cached_render(self.font_small, self.t("ms_hint"), (140, 140, 155))
        surf.blit(hint, (panel.left + pad, panel.bottom - pad - hint.get_height()))

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
        scroll_h = int(14 * scale)
        gap_below_plots = int(6 * scale)
        plot_h = max(36, (avail_h - 2 * plot_gap - scroll_h - gap_below_plots) // 3)
        p0 = pygame.Rect(px, content_top, plot_w, plot_h)
        p1 = pygame.Rect(px, content_top + plot_h + plot_gap, plot_w, plot_h)
        p2 = pygame.Rect(px, content_top + 2 * (plot_h + plot_gap), plot_w, plot_h)
        scroll_y = p2.bottom + gap_below_plots
        plot_scroll = pygame.Rect(px, scroll_y, plot_w, scroll_h)
        plots_union = p0.union(p1).union(p2)
        bw = int(np.clip(168 * scale, 96, 220))
        bh = int(36 * scale)
        restart_btn = pygame.Rect(inner.right - bw - 4, inner.bottom - bh - 6, bw, bh)
        close_btn = pygame.Rect(restart_btn.left - bw - 12, restart_btn.top, bw, bh)
        return _MissionReportGeom(panel, close_btn, restart_btn, photo, p0, p1, p2, plot_scroll, plots_union)

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

    def _plot_event_caption(self, tag_id: str, detail: str) -> str:
        if tag_id == "EN":
            return self.t("plot_evt_en_on") if detail == "ON" else self.t("plot_evt_en_off")
        key = f"plot_evt_{tag_id.lower()}"
        return self.t(key)

    def _plot_markers_from_log(
        self, hist: list[dict], c: Controller
    ) -> list[tuple[float, tuple[int, int, int], int, str, str]]:
        """(t_rel, rgb, tri_dir, tag_id, label) from Controller-registered actions."""
        if len(hist) < 1:
            return []
        t0 = float(hist[0]["t_s"])
        t_end = float(hist[-1]["t_s"])
        col_hs = (200, 95, 40)
        col_dr = (40, 150, 185)
        col_mn = (45, 130, 65)
        col_cj = (155, 55, 155)
        col_tgt = (95, 90, 155)
        colors = {"HS": col_hs, "DR": col_dr, "MN": col_mn, "CJ": col_cj, "TGT": col_tgt}
        out: list[tuple[float, tuple[int, int, int], int, str, str]] = []
        for ts, tag, det in c.model.plot_action_log:
            if ts < t0 - 1e-9 or ts > t_end + 1e-9:
                continue
            tr = float(ts) - t0
            if tag == "EN":
                tri = 1 if det == "ON" else -1
                col = (215, 165, 40) if det == "ON" else (115, 75, 45)
            else:
                tri = 1
                col = colors.get(tag, (130, 130, 140))
            out.append((tr, col, tri, tag, self._plot_event_caption(tag, det)))
        return out

    def _markers_from_telemetry_diff(
        self, hist: list[dict]
    ) -> list[tuple[float, tuple[int, int, int], int, str, str]]:
        """Fallback when plot log is empty or missing an edge: (t_rel, rgb, tri, tag_id, label)."""
        col_hs = (200, 95, 40)
        col_dr = (40, 150, 185)
        col_mn = (45, 130, 65)
        col_cj = (155, 55, 155)
        col_en_on = (215, 165, 40)
        col_en_off = (115, 75, 45)
        out: list[tuple[float, tuple[int, int, int], int, str, str]] = []
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
                out.append((tr, col_hs, 1, "HS", self._plot_event_caption("HS", "")))
            a, b = _g(prev, "dr"), _g(cur, "dr")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_dr, 1, "DR", self._plot_event_caption("DR", "")))
            a, b = _g(prev, "mn"), _g(cur, "mn")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_mn, 1, "MN", self._plot_event_caption("MN", "")))
            a, b = _g(prev, "cj"), _g(cur, "cj")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_cj, 1, "CJ", self._plot_event_caption("CJ", "")))
            a, b = _g(prev, "eng"), _g(cur, "eng")
            if a < 0.5 and b > 0.5:
                out.append((tr, col_en_on, 1, "EN", self._plot_event_caption("EN", "ON")))
            elif a > 0.5 and b < 0.5:
                out.append((tr, col_en_off, -1, "EN", self._plot_event_caption("EN", "OFF")))
        return out

    def _merge_plot_markers(
        self,
        from_log: list[tuple[float, tuple[int, int, int], int, str, str]],
        from_state: list[tuple[float, tuple[int, int, int], int, str, str]],
        eps_s: float = 0.35,
    ) -> list[tuple[float, tuple[int, int, int], int, str, str]]:
        merged = list(from_log)
        for s in from_state:
            tr_s, col, tri, tid, lab = s
            if any(abs(tr_s - tr_l) < eps_s and tid_l == tid for tr_l, _, _, tid_l, _ in from_log):
                continue
            merged.append(s)
        merged.sort(key=lambda x: x[0])
        return merged

    def _mission_graph_markers(self, hist: list[dict], c: Controller) -> list[tuple[float, tuple[int, int, int], int, str]]:
        """Markers for drawing: t_rel, color, tri_dir, caption."""
        log_m = self._plot_markers_from_log(hist, c)
        st_m = self._markers_from_telemetry_diff(hist)
        full = self._merge_plot_markers(log_m, st_m) if log_m else st_m
        return [(a, b, c, e) for a, b, c, _, e in full]

    @staticmethod
    def _mission_downsample_series(
        xs: list[float],
        ys: list[float],
        max_pts: int,
        mode: str,
    ) -> tuple[list[float], list[float]]:
        """Reduce points for drawing long missions; preserves peaks (max) or extrema (minmax)."""
        n = min(len(xs), len(ys))
        if n <= max_pts or max_pts < 16:
            return xs, ys
        xa = np.asarray(xs[:n], dtype=np.float64)
        ya = np.asarray(ys[:n], dtype=np.float64)
        if mode == "line":
            idx = np.unique(np.linspace(0, n - 1, min(max_pts, n), dtype=int))
            return [float(xa[i]) for i in idx], [float(ya[i]) for i in idx]
        nb_bins = max(1, max_pts // 2) if mode == "minmax" else max_pts
        out_x: list[float] = []
        out_y: list[float] = []
        for b in range(nb_bins):
            i0 = (b * n) // nb_bins
            i1 = ((b + 1) * n) // nb_bins
            i1 = max(i0 + 1, min(i1, n))
            sl = slice(i0, i1)
            if mode == "max":
                j = i0 + int(np.argmax(ya[sl]))
                out_x.append(float(xa[j]))
                out_y.append(float(ya[j]))
            else:  # minmax — altitude envelope
                jmin = i0 + int(np.argmin(ya[sl]))
                jmax = i0 + int(np.argmax(ya[sl]))
                if float(xa[jmin]) <= float(xa[jmax]):
                    out_x.extend([float(xa[jmin]), float(xa[jmax])])
                    out_y.extend([float(ya[jmin]), float(ya[jmax])])
                else:
                    out_x.extend([float(xa[jmax]), float(xa[jmin])])
                    out_y.extend([float(ya[jmax]), float(ya[jmin])])
        return out_x, out_y

    def _mission_timeline_clamp(self, t_total: float) -> None:
        t_total = max(1e-9, float(t_total))
        span = float(self._mission_view_hi) - float(self._mission_view_lo)
        if span >= t_total - 1e-9:
            self._mission_view_lo = 0.0
            self._mission_view_hi = t_total
            return
        span = max(1e-9, min(span, t_total))
        lo = float(np.clip(float(self._mission_view_lo), 0.0, t_total - span))
        self._mission_view_lo = lo
        self._mission_view_hi = lo + span

    def _draw_mission_timeline_scrubber(
        self,
        surf: pygame.Surface,
        track: pygame.Rect,
        t_total: float,
    ) -> None:
        ink = (28, 42, 68)
        fill = (200, 196, 180)
        hi = (228, 224, 212)
        pygame.draw.rect(surf, hi, track, border_radius=4)
        pygame.draw.rect(surf, ink, track, width=1, border_radius=4)
        if t_total <= 1e-9:
            return
        span = float(self._mission_view_hi) - float(self._mission_view_lo)
        if span >= t_total - 1e-9:
            th = max(4, track.height - 4)
            tr = pygame.Rect(track.left + 2, track.centery - th // 2, track.width - 4, th)
            pygame.draw.rect(surf, fill, tr, border_radius=3)
            return
        frac_w = max(0.08, span / t_total)
        thumb_w = max(18, int(frac_w * track.width))
        room = max(1, track.width - thumb_w - 4)
        rem = max(1e-9, t_total - span)
        pos = float(self._mission_view_lo) / rem
        tx = int(track.left + 2 + pos * room)
        tx = int(np.clip(tx, track.left + 2, track.right - thumb_w - 2))
        th = max(4, track.height - 4)
        tr = pygame.Rect(tx, track.centery - th // 2, thumb_w, th)
        pygame.draw.rect(surf, fill, tr, border_radius=3)
        pygame.draw.rect(surf, ink, tr, width=1, border_radius=3)

    def _mission_timeline_thumb_rect(self, track: pygame.Rect, t_total: float) -> pygame.Rect:
        span = float(self._mission_view_hi) - float(self._mission_view_lo)
        if t_total <= 1e-9 or span >= t_total - 1e-9:
            return pygame.Rect(track.left + 2, track.top + 2, track.width - 4, track.height - 4)
        frac_w = max(0.08, span / t_total)
        thumb_w = max(18, int(frac_w * track.width))
        room = max(1, track.width - thumb_w - 4)
        rem = max(1e-9, t_total - span)
        pos = float(self._mission_view_lo) / rem
        tx = int(track.left + 2 + pos * room)
        tx = int(np.clip(tx, track.left + 2, track.right - thumb_w - 2))
        th = max(4, track.height - 4)
        return pygame.Rect(tx, track.centery - th // 2, thumb_w, th)

    @staticmethod
    def _nice_y_axis(lo: float, hi: float, max_ticks: int = 7) -> tuple[float, float, list[float]]:
        """Round Y limits to readable tick step; returns (y_min, y_max, tick values)."""
        if hi < lo:
            lo, hi = hi, lo
        span = hi - lo
        if span < 1e-30:
            lo -= 1e-6
            hi += 1e-6
            span = hi - lo
        raw = span / max(3, max_ticks - 1)
        exp = math.floor(math.log10(max(raw, 1e-30)))
        base = 10.0**exp
        step = base
        for mul in (1.0, 2.0, 2.5, 5.0, 10.0):
            cand = mul * base
            if cand >= raw * 0.82:
                step = cand
                break
        else:
            step = 10.0 * base
        y0 = math.floor(lo / step) * step
        y1 = math.ceil(hi / step) * step
        if y1 <= y0 + 1e-30:
            y1 = y0 + step
        ticks: list[float] = []
        t = y0
        for _ in range(40):
            if t > y1 + step * 1e-6:
                break
            ticks.append(float(round(t, 12)))
            t += step
        return y0, y1, ticks

    @staticmethod
    def _dashed_hline(
        surf: pygame.Surface,
        y: int,
        x0: int,
        x1: int,
        color: tuple[int, int, int],
        dash: int = 5,
        gap: int = 4,
    ) -> None:
        x = x0
        while x < x1:
            xe = min(x + dash, x1)
            pygame.draw.line(surf, color, (x, y), (xe, y), 1)
            x = xe + gap

    def _draw_plot_blueprint(
        self,
        surf: pygame.Surface,
        rect: pygame.Rect,
        xs: list[float],
        ys: list[float],
        title: str,
        color: tuple[int, int, int],
        events: Optional[list[tuple[float, tuple[int, int, int], int, str]]] = None,
        y_tick_fmt: Callable[[float], str] | None = None,
        *,
        y_span_min: float = 0.0,
        y_include_zero: bool = False,
        x_window: Optional[tuple[float, float]] = None,
        series_downsample: Optional[str] = None,
        max_plot_points: int = 12000,
        subtitle: Optional[str] = None,
        show_x_axis_caption: bool = False,
    ) -> None:
        events = events or []
        yfmt = y_tick_fmt or (lambda v: f"{v:.2f}")
        paper = (228, 224, 212)
        grid_h = (210, 206, 198)
        grid_v = (218, 214, 206)
        ink = (28, 42, 68)
        muted = (88, 82, 76)
        pygame.draw.rect(surf, paper, rect, border_radius=4)
        pygame.draw.rect(surf, ink, rect, width=2, border_radius=4)
        fs_h = self.font_small.get_height()
        head_gap = 5

        wxs = list(xs)
        wys = list(ys)
        xv0: Optional[float] = None
        xv1: Optional[float] = None
        if x_window is not None:
            xv0 = float(x_window[0])
            xv1 = float(x_window[1])
            if xv1 < xv0:
                xv0, xv1 = xv1, xv0
            if len(wxs) > 0:
                i0 = bisect.bisect_left(wxs, xv0)
                i1 = bisect.bisect_right(wxs, xv1)
                i1 = max(i0 + 1, min(i1, len(wxs)))
                wxs = wxs[i0:i1]
                wys = wys[i0:i1]
        n0 = min(len(wxs), len(wys))
        if n0 < 2:
            return
        wys_stats = list(wys)
        st_min = float(min(wys_stats))
        st_max = float(max(wys_stats))
        st_end = float(wys_stats[-1])
        if series_downsample:
            wxs, wys = self._mission_downsample_series(wxs, wys, max_plot_points, series_downsample)
            n = min(len(wxs), len(wys))
        else:
            n = n0
        if n < 2:
            return
        ymin = float(min(wys[:n]))
        ymax = float(max(wys[:n]))
        if abs(ymax - ymin) < 1e-9:
            c = ymin
            half = max(abs(c) * 0.02, 1e-9)
            if y_span_min > 0.0:
                half = max(half, y_span_min * 0.5)
            else:
                half = max(half, 1e-3)
            ymin, ymax = c - half, c + half
        span_y = ymax - ymin
        pad_y = max(span_y * 0.08, 1e-6)
        ymin -= pad_y
        ymax += pad_y
        if y_span_min > 0.0 and (ymax - ymin) < y_span_min:
            mid = 0.5 * (ymin + ymax)
            ext = 0.5 * y_span_min
            ymin, ymax = mid - ext, mid + ext
        if y_include_zero:
            ymin = min(ymin, 0.0)
        ymin, ymax, y_ticks = self._nice_y_axis(ymin, ymax, max_ticks=7)
        if x_window is not None and xv0 is not None and xv1 is not None:
            xl0, xl1 = xv0, xv1
        else:
            xl0 = float(wxs[0])
            xl1 = float(wxs[n - 1])
        if abs(xl1 - xl0) < 1e-9:
            xl0 -= 0.5
            xl1 += 0.5

        mw = 0
        for yv in y_ticks:
            tlw = self._cached_render(self.font_small, yfmt(yv), ink).get_width()
            mw = max(mw, tlw)
        y_axis_w = int(np.clip(mw + 20, 48, 132))

        stats_txt = self.t("mission_plot_stats").format(mn=yfmt(st_min), mx=yfmt(st_max), end=yfmt(st_end))
        y_head = rect.top + 6
        cap = self._cached_render(self.font_small, title.upper(), ink)
        surf.blit(cap, (rect.left + 6, y_head))
        y_head += fs_h + head_gap
        if subtitle:
            sub = self._cached_render(self.font_small, subtitle, muted)
            surf.blit(sub, (rect.left + 6, y_head))
            y_head += fs_h + head_gap
        st_surf = self._cached_render(self.font_small, stats_txt, muted)
        surf.blit(st_surf, (rect.left + 6, y_head))
        y_head += fs_h + 8

        x_tick_h = fs_h + 8
        x_caption_h = (fs_h + 6) if show_x_axis_caption else 0
        axis_h = 10 + x_tick_h + x_caption_h
        inner_top = y_head
        inner_left = rect.left + 5
        inner_w = rect.width - 10
        inner_bottom = rect.bottom - 6 - axis_h
        inner = pygame.Rect(inner_left, inner_top, inner_w, max(12, inner_bottom - inner_top))
        plot_inner = pygame.Rect(
            inner.left + y_axis_w,
            inner.top,
            max(12, inner.width - y_axis_w - 4),
            inner.height,
        )

        def _y_to_px(yv: float) -> int:
            return int(plot_inner.bottom - (float(yv) - ymin) / max(1e-30, (ymax - ymin)) * plot_inner.height)

        def _x_to_px(t_rel: float) -> int:
            return int(plot_inner.left + (float(t_rel) - xl0) / (xl1 - xl0) * plot_inner.width)

        for yv in y_ticks:
            py = _y_to_px(yv)
            if plot_inner.top <= py <= plot_inner.bottom:
                pygame.draw.line(surf, grid_h, (plot_inner.left, py), (plot_inner.right, py), 1)
        nt = 6 if (xl1 - xl0) > 3600.0 else 5
        for k in range(nt + 1):
            t_rel = xl0 + (xl1 - xl0) * (k / float(nt))
            px = _x_to_px(t_rel)
            pygame.draw.line(surf, grid_v, (px, plot_inner.top), (px, plot_inner.bottom), 1)

        for yv in y_ticks:
            py = int(np.clip(_y_to_px(yv), plot_inner.top, plot_inner.bottom))
            pygame.draw.line(surf, ink, (plot_inner.left - 5, py), (plot_inner.left, py), 2)
            tl = self._cached_render(self.font_small, yfmt(yv), ink)
            lx = plot_inner.left - 8 - tl.get_width()
            surf.blit(tl, (lx, int(np.clip(py - tl.get_height() // 2, inner.top, inner.bottom - tl.get_height()))))

        if ymin <= 0.0 <= ymax:
            py0 = int(np.clip(_y_to_px(0.0), plot_inner.top, plot_inner.bottom))
            self._dashed_hline(surf, py0, plot_inner.left, plot_inner.right, (150, 145, 138))

        pts: list[tuple[int, int]] = []
        for i in range(n):
            tx = (float(wxs[i]) - xl0) / (xl1 - xl0)
            ty = (float(wys[i]) - ymin) / (ymax - ymin)
            px = int(plot_inner.left + tx * plot_inner.width)
            py = int(plot_inner.bottom - ty * plot_inner.height)
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(surf, color, False, pts, 2)
            pygame.draw.lines(surf, ink, False, pts, 1)

        ax_y = plot_inner.bottom
        pygame.draw.line(surf, ink, (plot_inner.left, ax_y), (plot_inner.right, ax_y), 2)
        tick_y = ax_y + 5
        for k in range(nt + 1):
            t_rel = xl0 + (xl1 - xl0) * (k / float(nt))
            px = _x_to_px(t_rel)
            pygame.draw.line(surf, ink, (px, ax_y), (px, ax_y + 4), 2)
            tl = self._cached_render(self.font_small, self._format_mission_elapsed(t_rel), ink)
            lx = int(np.clip(px - tl.get_width() // 2, plot_inner.left, max(plot_inner.left, plot_inner.right - tl.get_width())))
            surf.blit(tl, (lx, tick_y))
        if show_x_axis_caption:
            cap_y = tick_y + fs_h + 4
            ax_cap = self._cached_render(self.font_small, self.t("mission_x_axis_label"), muted)
            surf.blit(ax_cap, (plot_inner.left, cap_y))

        tri_s = max(5, int(6 * self.ui_scale))
        by_px: dict[int, list[tuple[tuple[int, int, int], int, str]]] = {}
        for t_rel, evc, tri_dir, caption in events:
            if t_rel < xl0 - 1e-6 or t_rel > xl1 + 1e-6:
                continue
            px = _x_to_px(t_rel)
            if px < plot_inner.left - 2 or px > plot_inner.right + 2:
                continue
            by_px.setdefault(px, []).append((evc, tri_dir, caption))
        for px in sorted(by_px.keys()):
            group = by_px[px]
            evc0, tri_dir0, _cap0 = group[0]
            pygame.draw.line(surf, evc0, (px, plot_inner.top), (px, plot_inner.bottom), 2)
            if tri_dir0 >= 0:
                pts_t = [(px, plot_inner.top), (px - tri_s, plot_inner.top + tri_s + 2), (px + tri_s, plot_inner.top + tri_s + 2)]
            else:
                pts_t = [
                    (px, plot_inner.bottom),
                    (px - tri_s, plot_inner.bottom - tri_s - 2),
                    (px + tri_s, plot_inner.bottom - tri_s - 2),
                ]
            pygame.draw.polygon(surf, evc0, pts_t)
            pygame.draw.polygon(surf, ink, pts_t, 1)
            ly0 = plot_inner.top + 4
            for gi, (_evc, _tri, caption) in enumerate(group):
                cap_s = self._cached_render(self.font_small, caption, ink)
                prefer_right = px < plot_inner.left + plot_inner.width // 2
                lx = (px + 5) if prefer_right else (px - 5 - cap_s.get_width())
                lx = int(np.clip(lx, inner.left + 2, plot_inner.right - cap_s.get_width() - 2))
                ycap = ly0 + gi * (cap_s.get_height() + 2)
                if ycap + cap_s.get_height() < plot_inner.bottom - 6:
                    surf.blit(cap_s, (lx, ycap))

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
            t_total = max(1e-9, float(xs[-1]))
            if self._mission_timeline_reset:
                self._mission_view_lo = 0.0
                self._mission_view_hi = t_total
                self._mission_timeline_reset = False
            self._mission_timeline_clamp(t_total)
            x_win = (float(self._mission_view_lo), float(self._mission_view_hi))
            markers = self._mission_graph_markers(hist, c)
            # Landing-phase noise can make altitude_m slightly negative; graph shows h ≥ 0 only.
            ys_alt_km = [max(0.0, float(p["altitude_m"]) / 1000.0) for p in hist]
            ys_vv = [abs(float(p["v_vert_mps"])) for p in hist]
            ys_g = [max(0.0, float(p["g_load"])) for p in hist]
            blueprint = (
                (ys_alt_km, self.t("plot_h_km"), colors[0], lambda v: f"{v:.2f}", "minmax", True),
                (ys_vv, self.t("plot_v_vert_abs"), colors[1], lambda v: f"{v:.0f}", "line", False),
                (ys_g, self.t("plot_g"), colors[2], lambda v: f"{v:.2f}", "max", True),
            )
            rng = self.t("mission_plot_time_range").format(
                a=self._format_mission_elapsed(float(self._mission_view_lo)),
                b=self._format_mission_elapsed(float(self._mission_view_hi)),
                total=self._format_mission_elapsed(t_total),
            )
            for i, pr in enumerate(plots):
                ys, ptitle, col, yfmt, ds, y0 = blueprint[i]
                self._draw_plot_blueprint(
                    surf,
                    pr,
                    xs,
                    ys,
                    ptitle,
                    col,
                    markers,
                    yfmt,
                    y_span_min=0.0,
                    y_include_zero=y0,
                    x_window=x_win,
                    series_downsample=ds,
                    max_plot_points=12000,
                    subtitle=rng if i == 0 else None,
                    show_x_axis_caption=(i == 2),
                )
            hint = self._cached_render(self.font_small, self.t("mission_timeline_hint"), (90, 85, 78))
            surf.blit(hint, (g.plot_scroll.left, g.plot_scroll.top - hint.get_height() - 2))
            self._draw_mission_timeline_scrubber(surf, g.plot_scroll, t_total)
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

