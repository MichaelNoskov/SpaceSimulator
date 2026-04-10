from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pygame

from digital_twin.config import BodyConfig
from digital_twin.model import PhysicsModel, SimResult

# --- Saturn / Sun in the sky (Titan) ------------------------------------------
# Target: visible planet+rings ≈ 1/9 of sky panel height at **surface**; sprite bbox is larger (padding).
# Angular size ∝ 1/D; D ≈ (a − R_Titan) − h with h = altitude (Saturn-facing radial, flat model).
# Ref: https://science.nasa.gov/solar-system/planets/saturn/saturn-moons/titan/
_SATURN_VISIBLE_SKY_FRACTION = 1.0 / 9.0
_SATURN_SPRITE_FILL = 0.48
_TITAN_ORBIT_SEMI_MAJOR_M = 1_221_870_000.0  # ~1221870 km
_TITAN_RADIUS_MEAN_M = 2_574_730.0  # ~2574.7 km
# Fixed screen anchor for Saturn (no libration / drift).
_SATURN_SKY_U = 0.265
_SATURN_SKY_V = 0.069
# Slight artistic boost: overall size (~4%) + stronger “grows while high” vs raw physics (~2.3× the Δθ).
_SATURN_GLOBAL_VISUAL_SCALE = 1.042
_SATURN_DISTANCE_EXAGGERATE = 2.35
# Gameplay: ~1.5× at typical entry altitude vs 1× on the ground (smoothstep); ref ≈ model start altitude.
_SATURN_ENTRY_BOOST_ALT_REF_M = float(BodyConfig().entry_start_altitude_m)
_SATURN_ENTRY_BOOST_MAX = 1.5
# The sky gradient is drawn in `rect`; treat its vertical span as this elevation arc (artistic).
_SKY_VERTICAL_EXTENT_DEG = 52.0
# Titan orbital period ≈ 15.945 d (synchronous rotation); sky motion uses model time_s.
_TITAN_ORBIT_PERIOD_S = 15.945 * 24.0 * 3600.0
# Sun from Titan: d ≈ Saturn heliocentric distance ~9.54 AU → angular diameter ~0.056°
# (R☉ / d; ~1/9.5 of Earth's ~0.53°). Flux ~1/90 of 1 AU — still bright but heavily attenuated by haze.
_SUN_ANGULAR_DIAMETER_DEG = 0.056
# Initial sky angle Sun vs subsaturn meridian (visual).
_SUN_SKY_PHASE0_RAD = 1.05
# Surface reference density (kg/m³); matches `digital_twin.config.TitanBody.rho_surface_kg_m3`.
_TITAN_RHO_SURFACE_REF_KG_M3 = 5.9
# Cloud advection: drift [px/frame] ≈ (wind − probe)·px_per_m·dt·_CLOUD_ADVECT_SCALE (tuned for readability).
_CLOUD_ADVECT_SCALE = 100.0
# Low-pass gusts so cloud motion does not jitter frame-to-frame (large inertia vs probe).
_CLOUD_WIND_SMOOTH_TAU_S = 3.5
# Per-shape haze variants (same wind; stacked decks differ in silhouette + transparency).
_HAZE_VARIANTS_PER_SHAPE = 3
_HAZE_SHAPE_COUNT = 5
# Sun nameplate: show without hover for the first N seconds of sim time, then hover-only.
_SKY_SUN_LABEL_INTRO_S = 14.0
_SKY_LABEL_BELOW_GAP = 10
_SKY_LABEL_OUTLINE_PX = 2


@dataclass
class EventFrag:
    kind: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    ang: float
    ang_v: float
    t: float
    dur: float
    scale: float
    color: tuple[int, int, int] = (220, 205, 190)


@dataclass
class Camera:
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    scale_px_per_m: float = 0.001


@dataclass
class CloudLayer:
    """One atmospheric deck: `shape_kind` selects silhouette family; `deck_alpha` scales layer opacity."""
    shape_kind: int
    deck_alpha: float
    ref_sprite: pygame.Surface
    # blob: (x_px, alt_m, yoff_px, _unused, inst_alpha, scale, wob_phase, sprite)
    blobs: list[tuple[float, float, float, float, float, float, float, pygame.Surface]]


class Renderer:
    def __init__(self, screen_size: Tuple[int, int]):
        self.w, self.h = screen_size
        self.camera = Camera()
        self._rng = np.random.default_rng(12345)
        self._particles: list[tuple[float, float, float, float, float, tuple[int, int, int]]] = []
        self._cloud_layers: list[CloudLayer] = self._make_cloud_layers()
        self._t = 0.0
        self._prev_flags: Optional[dict[str, bool]] = None
        self._event_frags: list[EventFrag] = []
        self._drogue_anim_t: Optional[float] = None
        self._main_anim_t: Optional[float] = None
        self._engine_spool: float = 0.0
        self._heatshield_rgb: tuple[int, int, int] = (220, 205, 190)
        # Fixed zoom: base vertical span [m] (chase camera).
        self._fixed_span_m: float = 2_500.0
        # Terrain profile fade: invisible above start; full opacity below end.
        self._terrain_fade_start_m: float = 2_450.0
        self._terrain_fade_end_m: float = 980.0
        # Blend to landing framing: once profile is visible, shift toward terrain slice.
        self._landing_cam_start_m: float = 1_480.0
        self._landing_cam_end_m: float = 520.0
        self._jettison_anim_t: Optional[float] = None
        # Smoothed (wx, wz) at probe altitude — one field for all clouds.
        self._cloud_wind_smooth: tuple[float, float] = (0.0, 0.0)
        # `_haze_pools` filled in `_make_cloud_layers` (do not assign [] after that).
        # Minimap cache for performance (procedural terrain is expensive to sample per frame).
        self._minimap_cache: Optional[pygame.Surface] = None
        self._minimap_cache_rect: Optional[pygame.Rect] = None
        self._minimap_cache_mpp: float = 0.0  # meters-per-pixel used for cache
        self._minimap_cache_center: tuple[float, float] = (float("nan"), float("nan"))  # (x_m, z_m)
        self._minimap_font: Optional[pygame.font.Font] = None
        self._minimap_cache_scaled: Optional[pygame.Surface] = None
        self._minimap_cache_scaled_size: tuple[int, int] = (0, 0)
        # Sky gradient cache: rebuild on resize or h/rho quantization step.
        self._sky_cache: Optional[pygame.Surface] = None
        self._sky_cache_key: Optional[tuple[int, int, int, int]] = None
        # Vertical-gradient fog plane (cached; rebuild on size / coarse h·ρ).
        self._fog_surf: Optional[pygame.Surface] = None
        self._fog_grad_key: Optional[tuple[int, int, int, int, int]] = None
        # Saturn in the sky (Titan view): optional data/saturn_sky.png or procedural fallback.
        self._saturn_tex: Optional[pygame.Surface] = None
        self._saturn_tex_key: Optional[int] = None
        # One high-res sky sprite per conditions; scaled each frame to smoothed size (avoids cache jumps).
        self._saturn_master: Optional[pygame.Surface] = None
        self._saturn_master_key: Optional[tuple[int, int, int, int, int, int]] = None
        # Entry-boost curve uses max( nominal ref, peak altitude this run ) so u=1 at mission top.
        self._saturn_entry_peak_alt_m: Optional[float] = None
        self._saturn_last_sim_time_s: float = -1.0
        # Sun disk + haze bloom (quantized size cache).
        self._sun_radial: Optional[pygame.Surface] = None
        self._sun_radial_key: Optional[tuple[int, int, int]] = None
        self._sky_sun_rect: Optional[pygame.Rect] = None
        self._sky_saturn_rect: Optional[pygame.Rect] = None
        self._celestial_label_font: Optional[pygame.font.Font] = None
        # Terrain profile layer (SRCALPHA) for smooth compositing without per-frame alloc.
        self._profile_layer: Optional[pygame.Surface] = None
        # When profile visible: probe screen (x,y) matches profile coords (touchdown = terrain line).
        self._profile_probe_xy: Optional[tuple[int, int]] = None
        # SRCALPHA buffer for particles (reused each frame).
        self._particle_buf: Optional[pygame.Surface] = None
        # Surfaces for engine flame and wind pennant.
        self._flame_surf: Optional[pygame.Surface] = None
        self._flame_surf_size: tuple[int, int] = (0, 0)
        self._pennant_surf: Optional[pygame.Surface] = None
        self._pennant_surf_size: tuple[int, int] = (0, 0)
        # 3D terrain grid cache (heightmap + surface types).
        self._tgrid_cache: Optional[dict] = None
        # Water landing: splash rings + probe sinking.
        self._water_splash: Optional[dict] = None
        self._was_water_landed: bool = False
        self._sink_t: float = 0.0

    def migrate_from(self, prev: Renderer) -> None:
        if (prev.w, prev.h) != (self.w, self.h):
            self._saturn_master = None
            self._saturn_master_key = None
            self._saturn_entry_peak_alt_m = None
            self._saturn_last_sim_time_s = -1.0
            self._cloud_wind_smooth = (0.0, 0.0)
            self._celestial_label_font = None

    def handle_orbit_event(self, event: pygame.event.Event, model: PhysicsModel) -> bool:
        return False

    @staticmethod
    def _smoothstep01(t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return float(t * t * (3.0 - 2.0 * t))

    def _terrain_profile_alpha(self, altitude_m: float) -> float:
        """0 = no profile; 1 = fully visible (smooth fade with altitude)."""
        h0, h1 = self._terrain_fade_start_m, self._terrain_fade_end_m
        a = float(altitude_m)
        if a >= h0:
            return 0.0
        if a <= h1:
            return 1.0
        return self._smoothstep01((h0 - a) / (h0 - h1))

    def _landing_camera_blend(self, altitude_m: float) -> float:
        """0 = chase camera on probe; 1 = landing framing with terrain profile."""
        a0, a1 = self._landing_cam_start_m, self._landing_cam_end_m
        a = float(altitude_m)
        if a >= a0:
            return 0.0
        if a <= a1:
            return 1.0
        return self._smoothstep01((a0 - a) / (a0 - a1))

    def draw(
        self,
        screen: pygame.Surface,
        model: PhysicsModel,
        ui,
        controller: Optional[object],
        frame_dt: float = 1 / 60.0,
    ) -> None:
        self._t += float(frame_dt)
        self._profile_probe_xy = None
        ui_rect = ui.visual_rect
        self._update_camera(model, ui_rect)
        self._update_event_anims(model, dt=float(frame_dt))

        self._sky_sun_rect = None
        self._sky_saturn_rect = None

        self._draw_sky(screen, ui_rect, model)
        self._draw_sun(screen, ui_rect, model)
        self._draw_saturn(screen, ui_rect, model, frame_dt=float(frame_dt))
        self._draw_wind_clouds(screen, ui_rect, model, frame_dt)
        self._draw_fog(screen, ui_rect, model)
        self._draw_celestial_labels(screen, ui_rect, model, ui)
        self._draw_ground(screen, ui_rect, model)
        self._draw_probe(screen, ui_rect, model)
        self._draw_event_frags(screen, ui_rect, model)
        self._draw_effects(screen, ui_rect, model, frame_dt)
        if controller is not None:
            ui._draw_overlay_hud(screen, controller)  # type: ignore[arg-type]
        self.draw_minimap(screen, model, ui.map_rect)
        if controller is not None:
            ui._draw_modal_overlays(screen, controller)  # type: ignore[arg-type]
            ui.draw_mission_report_modal(screen, controller)  # type: ignore[arg-type]
            ui.draw_pause_control(screen)  # type: ignore[arg-type]
            ui.draw_failure_outcome(screen, controller)  # type: ignore[arg-type]
        pygame.display.flip()

    def draw_minimap(self, screen: pygame.Surface, model: PhysicsModel, rect: pygame.Rect) -> None:
        """Draw minimap in the given rect (visualization only)."""
        self._draw_minimap(screen, model, rect)

    def _update_camera(self, model: PhysicsModel, rect: pygame.Rect) -> None:
        h_m = float(max(0.0, model.altitude_m))
        u = self._landing_camera_blend(h_m)
        chase_cy = h_m
        span_chase = float(self._fixed_span_m)
        # Near the ground, pull camera down so less empty sky sits above terrain.
        land_cy = h_m * 0.48
        span_land = float(min(span_chase, max(80.0, 1.50 * max(30.0, h_m))))

        self.camera.center_x_m = float(model.pos_x_m)
        self.camera.center_y_m = (1.0 - u) * chase_cy + u * land_cy
        span_eff = (1.0 - u) * span_chase + u * span_land
        self.camera.scale_px_per_m = rect.height / max(1e-6, span_eff)

    def _world_to_screen(self, rect: pygame.Rect, x_m: float, y_m: float) -> tuple[int, int]:
        scale = self.camera.scale_px_per_m
        sx = rect.centerx + int((x_m - self.camera.center_x_m) * scale)
        sy = rect.centery - int((y_m - self.camera.center_y_m) * scale)
        return sx, sy

    def _probe_screen_xy(self, rect: pygame.Rect, model: PhysicsModel) -> tuple[int, int]:
        """Probe screen center; when profile visible, matches profile-line geometry."""
        if self._profile_probe_xy is not None:
            return self._profile_probe_xy
        return self._world_to_screen(rect, model.pos_x_m, max(0.0, model.altitude_m))

    def _sky_gradient_colors(self, h: float, rho: float) -> tuple[np.ndarray, np.ndarray]:
        """Top/bottom sky gradient colors (compute only, no draw)."""
        c_high_top = np.array([70, 45, 65], dtype=float)
        c_high_bot = np.array([35, 20, 35], dtype=float)
        c_mid_top = np.array([160, 90, 70], dtype=float)
        c_mid_bot = np.array([90, 45, 40], dtype=float)
        c_low_top = np.array([235, 140, 70], dtype=float)
        c_low_bot = np.array([120, 60, 45], dtype=float)
        c_ground_top = np.array([255, 175, 110], dtype=float)
        c_ground_bot = np.array([140, 80, 60], dtype=float)

        def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
            t = float(np.clip(t, 0.0, 1.0))
            return a * (1 - t) + b * t

        if h <= 2_000.0:
            u = h / 2_000.0
            top = lerp(c_ground_top, c_low_top, u)
            bottom = lerp(c_ground_bot, c_low_bot, u)
        elif h <= 30_000.0:
            u = (h - 2_000.0) / (28_000.0)
            top = lerp(c_low_top, c_mid_top, u * 0.4)
            bottom = lerp(c_low_bot, c_mid_bot, u * 0.4)
        elif h <= 200_000.0:
            u = (h - 30_000.0) / (170_000.0)
            top = lerp(c_mid_top, c_high_top, u)
            bottom = lerp(c_mid_bot, c_high_bot, u)
        else:
            top, bottom = c_high_top, c_high_bot

        haze = float(np.clip((rho / float(_TITAN_RHO_SURFACE_REF_KG_M3)) ** 0.35, 0.0, 1.0))
        milky = np.array([245, 205, 170], dtype=float)
        top = lerp(top, milky, 0.20 * haze)
        bottom = lerp(bottom, milky * 0.75, 0.28 * haze)
        # Deeper in the atmosphere: stronger orange smog (Cassini “orange haze” imagery).
        depth = float(np.clip((1.0 - min(h, 5_500.0) / 5_500.0), 0.0, 1.0))
        smog = depth * float(np.clip((rho / 4.2) ** 0.55, 0.0, 1.15))
        smog_col = np.array([255, 118, 42], dtype=float)
        top = lerp(top, smog_col, 0.14 * smog)
        bottom = lerp(bottom, smog_col * 0.82, 0.22 * smog)
        return top, bottom

    @staticmethod
    def _titan_celestial_haze(h_km: float, rho: float) -> tuple[float, tuple[int, int, int], float]:
        """
        Visibility and warm haze tint vs atmospheric density (primary) and slant path (secondary).
        Extinction scales with ρ/ρ₀ (ρ₀ ≈ surface); altitude only nudges path length near the ground.
        Returns (alpha scale 0..1, RGB multiply for BLEND_RGBA_MULT, sharpness of disk 0..1).
        """
        rho0 = float(_TITAN_RHO_SURFACE_REF_KG_M3)
        rho_r = float(np.clip(rho / max(rho0, 1e-9), 0.0, 2.4))
        low = float(np.clip(1.0 - h_km / 2_800.0, 0.0, 1.0))
        tau = 0.92 * rho_r * (1.0 + 0.28 * low)
        vis = 0.14 + 0.86 * float(np.exp(-1.05 * tau))
        vis = float(np.clip(vis, 0.16, 1.0))
        o = float(np.clip((rho_r ** 0.62) * (0.78 + 0.22 * low), 0.0, 1.0))
        tint = (
            int(np.clip(255 - 22 * o, 150, 255)),
            int(np.clip(248 - 52 * o, 105, 255)),
            int(np.clip(218 - 98 * o, 78, 255)),
        )
        sharp = float(np.clip(1.0 - 0.42 * rho_r - 0.06 * low, 0.62, 1.0))
        return vis, tint, sharp

    @staticmethod
    def _titan_sun_uv(t_s: float) -> tuple[float, float]:
        """Sun only: synchronous Titan; ~one circuit per orbit vs Saturn meridian."""
        T = float(_TITAN_ORBIT_PERIOD_S)
        phi = 2.0 * math.pi * (t_s / T)
        psi = phi + float(_SUN_SKY_PHASE0_RAD)
        u_sun = 0.5 + 0.48 * math.cos(psi + 0.06)
        v_sun = 0.048 + 0.13 * max(-0.2, math.sin(psi + 0.52))
        return u_sun, v_sun

    @staticmethod
    def _saturn_distance_angular_scale(altitude_m: float) -> float:
        """
        θ ∝ 1/D with D ≈ D0−h; embellish (raw−1) so approach is readable in-game (still capped).
        """
        d0 = float(_TITAN_ORBIT_SEMI_MAJOR_M) - float(_TITAN_RADIUS_MEAN_M)
        h = max(0.0, float(altitude_m))
        dh = d0 - h
        lo = max(0.15 * d0, 1e6)
        if dh < lo:
            dh = lo
        raw = float(d0 / dh)
        ex = float(_SATURN_DISTANCE_EXAGGERATE)
        vis = 1.0 + ex * (raw - 1.0)
        return float(np.clip(vis, 1.0, 1.14))

    @staticmethod
    def _uv_to_sky_xy(rect: pygame.Rect, u: float, v: float, tex_w: int, tex_h: int) -> tuple[int, int]:
        margin_x = int(rect.width * 0.04 + 8)
        margin_t = int(rect.height * 0.06 + 6)
        usable = max(1, rect.width - tex_w - 2 * margin_x)
        x = rect.left + margin_x + int(float(u) * usable)
        y = rect.top + margin_t + int(float(v) * rect.height)
        return x, y

    @staticmethod
    def _uv_to_sky_xy_center(rect: pygame.Rect, u: float, v: float, tex_w: int, tex_h: int) -> tuple[int, int]:
        """(u,v) = position of sprite center; blit top-left so scaling is about the disk center."""
        margin_x = int(rect.width * 0.04 + 8)
        margin_t = int(rect.height * 0.06 + 6)
        margin_b = margin_t
        inner_w = max(1, rect.width - 2 * margin_x)
        inner_h = max(1, rect.height - margin_t - margin_b)
        cx = rect.left + margin_x + tex_w // 2 + int(float(u) * max(0, inner_w - tex_w))
        cy = rect.top + margin_t + tex_h // 2 + int(float(v) * max(0, inner_h - tex_h))
        return cx - tex_w // 2, cy - tex_h // 2

    def _draw_sky(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        h = max(0.0, float(model.altitude_m))
        rho = float(model.atm_density_kg_m3)
        # Coarser quantization: on fast descent int(rho*500) would change almost every frame —
        # full sky rebuild + scale would tank FPS and cause long stalls from alloc/GC.
        qh = int(h // 380.0) if h < 4_200.0 else int(h // 900.0)
        qr = int(rho * 80.0) // 4
        key = (rect.width, rect.height, qh, qr)
        if self._sky_cache is not None and self._sky_cache_key == key:
            screen.blit(self._sky_cache, rect.topleft)
            return

        top, bottom = self._sky_gradient_colors(h, rho)
        hh = max(1, rect.height)
        t = np.linspace(0.0, 1.0, hh, dtype=np.float64)
        grad = np.outer(1.0 - t, top) + np.outer(t, bottom)
        grad = np.clip(grad, 0.0, 255.0).astype(np.uint8)

        # One 1×h vertical strip, then scale to frame width (instead of thousands of draw.line).
        col = pygame.Surface((1, hh))
        px = pygame.surfarray.pixels3d(col)
        px[0, :, 0] = grad[:, 0]
        px[0, :, 1] = grad[:, 1]
        px[0, :, 2] = grad[:, 2]
        del px

        scaled = pygame.transform.scale(col, (rect.width, rect.height))
        self._sky_cache = scaled
        self._sky_cache_key = key
        screen.blit(scaled, rect.topleft)

    def _saturn_sprite(self, max_w: int) -> pygame.Surface:
        """Cached texture: optional data/saturn_sky.png, else procedural gas-giant + rings."""
        key = max(48, max_w) // 8
        if self._saturn_tex is not None and self._saturn_tex_key == key:
            return self._saturn_tex

        w = int(np.clip(max_w, 48, 640))
        data_path = Path(__file__).resolve().parent / "data" / "saturn_sky.png"
        surf: Optional[pygame.Surface] = None
        if data_path.is_file():
            try:
                raw = pygame.image.load(str(data_path)).convert_alpha()
                iw, ih = raw.get_size()
                if iw > 0 and ih > 0:
                    nh = max(40, int(w * ih / iw))
                    surf = pygame.transform.smoothscale(raw, (w, nh))
            except pygame.error:
                surf = None
        if surf is None:
            surf = self._make_procedural_saturn_sprite(w)

        self._saturn_tex = surf
        self._saturn_tex_key = key
        return surf

    @staticmethod
    def _saturn_target_height_px(
        sky_h_px: float, h_km: float, rho: float, altitude_m: float, entry_boost_href_m: float
    ) -> float:
        """
        Surface: base ~1/9 panel; entry boost 1→1.5 with u = alt / entry_boost_href_m (smoothstep).
        Cap scales with panel height — no fixed 620px ceiling (that caused a size plateau on tall windows).
        """
        fill = max(0.28, min(float(_SATURN_SPRITE_FILL), 0.75))
        base = float(sky_h_px) * (float(_SATURN_VISIBLE_SKY_FRACTION) / fill)
        dist_scale = Renderer._saturn_distance_angular_scale(altitude_m)
        haze = float(np.clip(rho / float(_TITAN_RHO_SURFACE_REF_KG_M3), 0.0, 1.25))
        vis = 1.0 - 0.01 * haze
        href = max(1.0, float(entry_boost_href_m))
        u = float(np.clip(float(altitude_m) / href, 0.0, 1.0))
        st = u * u * (3.0 - 2.0 * u)
        entry_boost = 1.0 + (float(_SATURN_ENTRY_BOOST_MAX) - 1.0) * st
        th = base * dist_scale * vis * float(_SATURN_GLOBAL_VISUAL_SCALE) * entry_boost
        cap = float(sky_h_px) * 0.52
        return float(np.clip(th, 24.0, cap))

    @staticmethod
    def _make_procedural_saturn_sprite(w: int) -> pygame.Surface:
        """Disk + tilted rings (view from Titan; not to scale)."""
        h = max(int(w * 0.42), 72)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        cx, cy = w // 2, int(h * 0.48)
        rp = max(8, int(w * 0.105))
        rw, rh = int(rp * 3.15), max(6, int(rp * 0.52))
        ring_rect = pygame.Rect(cx - rw // 2, cy - rh // 2, rw, rh)
        ring = pygame.Surface((rw, rh), pygame.SRCALPHA)
        pygame.draw.ellipse(ring, (195, 168, 130, 95), ring.get_rect())
        pygame.draw.ellipse(ring, (165, 125, 95, 150), ring.get_rect(), width=max(1, rp // 16))
        s.blit(ring, ring_rect.topleft)

        pygame.draw.ellipse(s, (235, 205, 155), (cx - rp, cy - rp, 2 * rp, 2 * rp))
        pygame.draw.ellipse(s, (200, 155, 105), (cx - rp, cy - rp, 2 * rp, 2 * rp), width=max(1, rp // 18))
        pygame.draw.arc(s, (110, 75, 52), (cx - rp, cy - rp, 2 * rp, 2 * rp), 0.35 * math.pi, 0.95 * math.pi, max(2, rp // 7))

        shine = pygame.Surface((2 * rp, 2 * rp), pygame.SRCALPHA)
        pygame.draw.ellipse(shine, (255, 230, 190, 55), (rp // 3, rp // 5, rp, int(rp * 0.75)))
        s.blit(shine, (cx - rp, cy - rp))

        front = pygame.Surface((rw, rh), pygame.SRCALPHA)
        pygame.draw.arc(front, (225, 200, 165, 130), front.get_rect(), 3.7, 5.9, max(2, rp // 9))
        s.blit(front, ring_rect.topleft)
        return s

    @staticmethod
    def _sun_disk_diameter_px(
        sky_h_px: float, h_km: float, rho: float, sharp: float
    ) -> tuple[float, float]:
        """
        Physical angular diameter at ~9.5 AU; sub-pixel → clamp for a visible core.
        Returns (d_core_px, bloom_scale). Low `sharp` widens halo vs crisp disk (aerosol scatter).
        Uses fixed window height (not camera zoom).
        """
        span = float(_SKY_VERTICAL_EXTENT_DEG)
        ang = float(_SUN_ANGULAR_DIAMETER_DEG)
        haze = float(np.clip(rho / float(_TITAN_RHO_SURFACE_REF_KG_M3), 0.0, 1.25))
        sh = float(np.clip(sharp, 0.62, 1.0))
        ang_eff = ang * (1.0 + 0.10 * haze) * sh
        d_phys = float(sky_h_px) * (ang_eff / span)
        d_core = max(2.0, min(d_phys, float(sky_h_px) * 0.035))
        bloom = 4.2 + 2.8 * haze + 0.22 * float(np.clip(1.0 - h_km / 900.0, 0.0, 1.0))
        bloom += 2.1 * (1.0 - sh)
        return d_core, bloom

    def _sun_radial_sprite(self, r_core: int, r_glow: int, qr: int, vis_b: int) -> pygame.Surface:
        """Disk + soft corona; cached on quantized radii + haze / visibility bucket."""
        key = (r_core, r_glow, qr, vis_b)
        if self._sun_radial is not None and self._sun_radial_key == key:
            return self._sun_radial
        r_glow = max(r_core + 2, r_glow)
        n = 2 * r_glow + 1
        s = pygame.Surface((n, n), pygame.SRCALPHA)
        cx = cy = r_glow
        rc = float(max(1, r_core))
        rg = float(r_glow)
        for y in range(n):
            for x in range(n):
                d = math.hypot(float(x - cx), float(y - cy))
                if d <= rc:
                    rr = int(255 - 8 * (d / max(rc, 1.0)))
                    gg = int(245 - 18 * (d / max(rc, 1.0)))
                    bb = int(215 - 35 * (d / max(rc, 1.0)))
                    s.set_at((x, y), (rr, gg, bb, 255))
                elif d >= rg:
                    continue
                else:
                    t = float((d - rc) / max(rg - rc, 1.0))
                    # Smooth edge; stronger falloff than rings (point source).
                    a = int(255 * ((1.0 - t) ** 2.2))
                    if a < 6:
                        continue
                    rr = int(255 - 40 * t)
                    gg = int(210 - 50 * t)
                    bb = int(140 - 40 * t)
                    s.set_at((x, y), (rr, gg, bb, a))

        self._sun_radial = s
        self._sun_radial_key = key
        return s

    def _draw_sun(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        """
        Sun at ~9.5 AU: small disk (~0.06°); scattered halo in photochemical haze (Cassini).
        Motion: synchronous lock keeps Saturn ~fixed; Sun moves ~360° per Titan orbit vs that meridian.
        """
        h_km = max(0.0, float(model.altitude_m) / 1000.0)
        rho = float(model.atm_density_kg_m3)
        qr = int(rho * 80.0) // 4
        vis, tint, sharp = self._titan_celestial_haze(h_km, rho)
        vis_b = int(vis * 14)

        d_core, bloom_mul = self._sun_disk_diameter_px(float(self.h), h_km, rho, sharp)
        r_core = max(1, int(round(0.5 * d_core)))
        r_glow = int(max(float(r_core) * bloom_mul, float(r_core) + 5.0))
        r_glow = min(r_glow, int(self.h * 0.22))

        tex = self._sun_radial_sprite(r_core, r_glow, qr, vis_b)
        tw, th = tex.get_size()

        t_s = float(model.time_s)
        u_sun, v_sun = self._titan_sun_uv(t_s)
        x, y = self._uv_to_sky_xy_center(rect, u_sun, v_sun, tw, th)

        dim = float(np.clip(0.38 + 0.62 * max(0.0, 1.0 - h_km / 1180.0), 0.38, 1.0))
        a = int(np.clip(108 + 118 * dim, 78, 236))
        a = int(np.clip(a * 0.76, 64, 236))
        a = int(np.clip(a * vis, 42, 234))
        out = tex.copy()
        out.fill(tint, special_flags=pygame.BLEND_RGBA_MULT)
        out.set_alpha(a)
        screen.blit(out, (x, y))
        self._sky_sun_rect = pygame.Rect(int(x), int(y), int(tw), int(th))

    def _draw_saturn(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel, frame_dt: float) -> None:
        """
        Fixed screen anchor; height tracks altitude every frame (no smoothing lag vs descent).
        """
        alt_m = max(0.0, float(model.altitude_m))
        h_km = alt_m / 1000.0
        rho = float(model.atm_density_kg_m3)
        t_sim = float(model.time_s)
        if t_sim < self._saturn_last_sim_time_s - 1e-3:
            self._saturn_entry_peak_alt_m = None
        self._saturn_last_sim_time_s = t_sim
        if self._saturn_entry_peak_alt_m is None:
            self._saturn_entry_peak_alt_m = alt_m
        else:
            self._saturn_entry_peak_alt_m = max(float(self._saturn_entry_peak_alt_m), alt_m)
        entry_href = max(float(_SATURN_ENTRY_BOOST_ALT_REF_M), float(self._saturn_entry_peak_alt_m))

        dim = float(np.clip(0.38 + 0.62 * max(0.0, 1.0 - h_km / 1180.0), 0.38, 1.0))
        vis, tint, _sharp = self._titan_celestial_haze(h_km, rho)

        sky_h = float(rect.height)
        th_tgt = self._saturn_target_height_px(sky_h, h_km, rho, alt_m, entry_href)
        th_cap = sky_h * 0.52
        th_want = float(np.clip(th_tgt, 24.0, th_cap))
        th_px = max(24, int(round(th_want)))

        qr = int(rho * 80.0) // 4
        vis_b = int(vis * 14)
        master_h = int(np.clip(int(sky_h * 0.48), 140, 520))
        # No altitude bucket in key — avoids spurious master rebuild / jumps (e.g. near 600 km).
        mk = (master_h, qr, rect.width, rect.height, vis_b)
        if self._saturn_master is None or self._saturn_master_key != mk:
            tw_load = max(96, int(master_h * (512.0 / 215.0)))
            tex0 = self._saturn_sprite(tw_load)
            iw0, ih0 = tex0.get_size()
            if ih0 <= 0:
                self._sky_saturn_rect = None
                return
            tw_m = max(22, int(iw0 * float(master_h) / float(ih0)))
            self._saturn_master = pygame.transform.smoothscale(tex0, (tw_m, master_h))
            self._saturn_master_key = mk

        mw, mh = self._saturn_master.get_size()
        if mh <= 0:
            self._sky_saturn_rect = None
            return
        tw_px = max(22, int(round(mw * float(th_px) / float(mh))))
        tex = pygame.transform.smoothscale(self._saturn_master, (tw_px, th_px))
        tw, th = tex.get_size()
        u_sat = float(_SATURN_SKY_U)
        v_sat = float(_SATURN_SKY_V)
        x, y = self._uv_to_sky_xy_center(rect, u_sat, v_sat, tw, th)

        a = int(np.clip(168 + 82 * dim, 118, 255))
        a = int(np.clip(a * vis, 55, 252))
        out = tex.copy()
        out.fill(tint, special_flags=pygame.BLEND_RGBA_MULT)
        out.set_alpha(a)
        screen.blit(out, (x, y))
        self._sky_saturn_rect = pygame.Rect(int(x), int(y), int(tw), int(th))

    @staticmethod
    def _celestial_name_strings(lang: str) -> tuple[str, str]:
        if (lang or "RU").upper().startswith("EN"):
            return "Sun", "Saturn"
        return "Солнце", "Сатурн"

    @staticmethod
    def _render_outlined_label(
        font: pygame.font.Font, text: str, fg: tuple[int, int, int], outline: tuple[int, int, int], ow: int
    ) -> pygame.Surface:
        core = font.render(text, True, fg)
        w, h = core.get_size()
        pad = ow * 2 + 2
        surf = pygame.Surface((w + pad, h + pad), pygame.SRCALPHA)
        otxt = font.render(text, True, outline)
        for dx in range(-ow, ow + 1):
            for dy in range(-ow, ow + 1):
                if dx * dx + dy * dy > ow * ow + 0.5:
                    continue
                surf.blit(otxt, (ow + 1 + dx, ow + 1 + dy))
        surf.blit(core, (ow + 1, ow + 1))
        return surf

    def _draw_celestial_labels(
        self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel, ui: object
    ) -> None:
        """Sun: caption at intro + on hover. Saturn: caption on hover only. Outlined text below body."""
        sun_r = self._sky_sun_rect
        sat_r = self._sky_saturn_rect
        if sun_r is None and sat_r is None:
            return

        lang = str(getattr(ui, "lang", "RU"))
        sun_txt, sat_txt = self._celestial_name_strings(lang)
        scale = float(getattr(ui, "ui_scale", 1.0))
        if self._celestial_label_font is None:
            self._celestial_label_font = pygame.font.SysFont("DejaVu Sans", max(13, int(14 * scale)))

        font = self._celestial_label_font
        fg = (248, 240, 220)
        ol = (18, 14, 28)
        pad = 14

        mx, my = pygame.mouse.get_pos()
        t_sim = float(model.time_s)
        intro = t_sim < float(_SKY_SUN_LABEL_INTRO_S)

        def blit_below(body: pygame.Rect, label: pygame.Surface) -> None:
            r = label.get_rect()
            r.midtop = (body.centerx, body.bottom + int(_SKY_LABEL_BELOW_GAP))
            # Clip to sky panel
            clip = screen.get_clip()
            try:
                screen.set_clip(rect)
                screen.blit(label, r)
            finally:
                screen.set_clip(clip)

        if sun_r is not None and rect.colliderect(sun_r):
            hover = sun_r.inflate(pad, pad).collidepoint(mx, my)
            if intro or hover:
                surf = self._render_outlined_label(font, sun_txt, fg, ol, int(_SKY_LABEL_OUTLINE_PX))
                blit_below(sun_r, surf)

        if sat_r is not None and rect.colliderect(sat_r):
            if sat_r.inflate(pad, pad).collidepoint(mx, my):
                surf = self._render_outlined_label(font, sat_txt, fg, ol, int(_SKY_LABEL_OUTLINE_PX))
                blit_below(sat_r, surf)

    def _draw_fog(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        """Orange nitrogen haze: stronger near the bottom of the view and in denser air."""
        h = float(model.altitude_m)
        rho = float(model.atm_density_kg_m3)
        a_max = float(np.clip(42.0 + 0.062 * (2800.0 - min(h, 2800.0)) + rho * 1050.0, 18.0, 248.0))
        qh = int(h // 650.0)
        qr = int(rho * 100.0) // 3
        qa = int(a_max) // 6
        key = (rect.width, rect.height, qh, qr, qa)
        if self._fog_surf is None or self._fog_grad_key != key:
            hh = max(1, rect.height)
            col = pygame.Surface((1, hh), pygame.SRCALPHA)
            for y in range(hh):
                t = y / max(hh - 1, 1)
                al = min(255, int(a_max * (0.06 + 0.94 * (t**1.18))))
                rr = int(188 + 42 * t)
                gg = int(118 + 48 * t)
                bb = int(72 + 38 * t)
                col.set_at((0, y), (rr, gg, bb, al))
            self._fog_surf = pygame.transform.smoothscale(col, (rect.width, rect.height))
            self._fog_grad_key = key
        screen.blit(self._fog_surf, rect.topleft)

    def _draw_ground(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        """
        3D isometric terrain: world-aligned grid, oblique projection, back-to-front
        with cliff walls and slope shading.  No wireframe — shading alone conveys shape.
        """
        alt = float(model.altitude_m)
        alpha = self._terrain_profile_alpha(alt)
        if alpha < 0.008:
            return

        probe_x = float(model.pos_x_m)
        probe_z = float(model.pos_z_m)
        h_ref = float(model.world.height_m_at(probe_x, probe_z))
        cam_sc = max(1e-9, self.camera.scale_px_per_m)

        vis = max(0.0, min(1.0, float(alpha)))
        terrain_strip_h = int(min(rect.height * (0.22 + 0.35 * vis), 540))
        flat_y_target = rect.top + int(rect.height * (2.0 / 3.0))
        flat_y = min(flat_y_target, rect.bottom - 24)
        band_top = max(rect.top + 8, flat_y - terrain_strip_h)
        if flat_y - band_top < 38:
            flat_y = min(rect.bottom - 8, band_top + 38)

        if self._profile_layer is None or self._profile_layer.get_size() != rect.size:
            self._profile_layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        layer = self._profile_layer
        layer.fill((0, 0, 0, 0))
        a255 = max(0, min(255, int(255 * alpha)))

        NX, NZ = 36, 22
        world_w_m = rect.width / cam_sc

        # Quantised step → grid snaps to world coordinates, no shape jitter.
        raw_sx = world_w_m * 1.15 / max(1, NX - 1)
        raw_sz = world_w_m * 0.70 / max(1, NZ - 1)
        step_x = max(10.0, round(raw_sx / 10.0) * 10.0)
        step_z = max(10.0, round(raw_sz / 10.0) * 10.0)

        # World-aligned origin (snapped to step multiples).
        gx0 = math.floor((probe_x / step_x) - NX * 0.50 + 0.5) * step_x
        gz0 = math.floor((probe_z / step_z) - NZ * 0.18 + 0.5) * step_z

        # --- heightmap grid with cache ---
        gc = self._tgrid_cache
        cache_ok = (
            gc is not None
            and gc.get('sx') == step_x and gc.get('sz') == step_z
            and gc.get('gx0') == gx0 and gc.get('gz0') == gz0
        )
        if cache_ok:
            grid_h = gc['grid_h']
            grid_st = gc['grid_st']
        else:
            grid_h: list[list[float]] = []
            grid_st: list[list[str]] = []
            for ix in range(NX):
                ch: list[float] = []
                cs: list[str] = []
                wx = gx0 + ix * step_x
                for iz in range(NZ):
                    wz = gz0 + iz * step_z
                    ch.append(float(model.world.height_m_at(wx, wz)))
                    cs.append(model.world.surface_type_at(wx, wz).value)
                grid_h.append(ch)
                grid_st.append(cs)
            self._tgrid_cache = {
                'sx': step_x, 'sz': step_z,
                'gx0': gx0, 'gz0': gz0,
                'grid_h': grid_h, 'grid_st': grid_st,
            }

        # --- projection: cam_sc-based for smooth zoom ---
        avail_h = max(30, flat_y - band_top - 10)
        depth_px = cam_sc * 0.45

        raw_max_dh = 80.0
        for col in grid_h:
            for hv in col:
                d = abs(hv - h_ref)
                if d > raw_max_dh:
                    raw_max_dh = d
        raw_max_dh = max(raw_max_dh, 80.0)
        # Use instantaneous range so vertical scale matches the current relief (smoothing caused drift vs blur).
        max_dh = raw_max_dh

        h_px = min(cam_sc * 0.50, avail_h * 0.38 / max(1.0, max_dh))

        base_y = flat_y
        cx = rect.centerx
        # wall_h removed — cliff wall strips no longer drawn.

        # Local aliases for proj() closure.
        _gx0 = gx0
        _gz0 = gz0
        _sx = step_x
        _sz = step_z
        _cs = cam_sc
        _hr = h_ref
        _cx = cx
        _by = base_y
        _dp = depth_px
        _hp = h_px
        _px = probe_x
        _pz = probe_z

        def proj(ix: int, iz: int) -> tuple[int, int]:
            return (
                _cx + int((_gx0 + ix * _sx - _px) * _cs),
                int(_by - (_gz0 + iz * _sz - _pz) * _dp - (grid_h[ix][iz] - _hr) * _hp),
            )

        def proj_world(wx: float, wz: float, wh: float) -> tuple[int, int]:
            return (
                _cx + int((wx - _px) * _cs),
                int(_by - (wz - _pz) * _dp - (wh - _hr) * _hp),
            )

        # Local height range → full color spectrum regardless of absolute scale.
        _hmin = grid_h[0][0]
        _hmax = _hmin
        for col in grid_h:
            for hv in col:
                if hv < _hmin:
                    _hmin = hv
                if hv > _hmax:
                    _hmax = hv
        _hrange = max(8.0, _hmax - _hmin)

        # --- dark ground fill ---
        pygame.draw.rect(
            layer, (32, 22, 16, a255),
            pygame.Rect(rect.left, band_top, rect.width, rect.bottom - band_top),
        )

        # Low-angle light so gentle slopes cast real shadows.
        _lx, _lz, _ly = -0.55, 0.50, 0.42
        _lm = math.sqrt(_lx * _lx + _lz * _lz + _ly * _ly)
        _lx /= _lm; _lz /= _lm; _ly /= _lm

        _bt = band_top
        _rb = rect.bottom
        # _wh alias removed (cliff walls removed).

        # Haze colour for atmospheric perspective (warm orange-grey).
        _haze_r, _haze_g, _haze_b = 118.0, 92.0, 72.0

        # --- back-to-front: top faces ---
        for iz in range(NZ - 2, -1, -1):
            for ix in range(NX - 1):
                pfl = proj(ix, iz)
                pfr = proj(ix + 1, iz)
                pnr = proj(ix + 1, iz + 1)
                pnl = proj(ix, iz + 1)

                if min(pfl[1], pfr[1], pnr[1], pnl[1]) > _rb + 4 \
                        or max(pfl[1], pfr[1], pnr[1], pnl[1]) < _bt - 4:
                    continue

                st = grid_st[ix][iz]
                hc = grid_h[ix][iz]
                hr = grid_h[min(ix + 1, NX - 1)][iz]
                hf = grid_h[ix][min(iz + 1, NZ - 1)]

                exag = 4.5
                nxn = -(hr - hc) / max(1.0, _sx) * exag
                nzn = -(hf - hc) / max(1.0, _sz) * exag
                nyn = 1.0
                nm = math.sqrt(nxn * nxn + nzn * nzn + nyn * nyn)
                if nm > 1e-9:
                    nxn /= nm; nzn /= nm; nyn /= nm
                dot = nxn * _lx + nzn * _lz + nyn * _ly
                shade = max(0.10, min(1.60, dot * 1.50 + 0.20))

                slope = math.sqrt(nxn * nxn + nzn * nzn)
                slope_dark = max(0.50, 1.0 - slope * 0.55)

                lit = 0.40 + 0.60 * shade * slope_dark

                # Atmospheric perspective: blend toward warm haze.
                zt = iz / max(1.0, NZ - 2.0)
                fog_vis = 0.52 + 0.48 * zt
                fog_haze = (1.0 - zt) * 0.30

                # World-coordinate micro-texture for visual variety.
                wx = _gx0 + ix * _sx
                wz = _gz0 + iz * _sz
                tex = 0.92 + 0.16 * math.sin(wx * 0.0085) * math.cos(wz * 0.012)

                if st == "lake":
                    # Liquid hydrocarbon (methane/ethane): dark umber, not Earth-water blue.
                    shimmer = 0.06 * math.sin(self._t * 1.8 + wx * 0.0003 + wz * 0.0004)
                    br = 48.0 + shimmer * 18
                    bg = 40.0 + shimmer * 15
                    bb = 28.0 + shimmer * 8
                    spec = max(0.0, math.sin(self._t * 0.7 + wx * 0.00015)) ** 8 * 0.25
                    lit = 0.60 + 0.40 * lit + spec
                else:
                    ht = max(0.0, min(1.0, (hc - _hmin) / _hrange))
                    br = 120.0 + 85.0 * ht
                    bg = 78.0 + 62.0 * ht
                    bb = 48.0 + 42.0 * ht

                sf = lit * fog_vis * tex
                r = max(0, min(255, int(br * sf + _haze_r * fog_haze)))
                g = max(0, min(255, int(bg * sf + _haze_g * fog_haze)))
                b = max(0, min(255, int(bb * sf + _haze_b * fog_haze)))
                pygame.draw.polygon(layer, (r, g, b, a255), [pfl, pfr, pnr, pnl])

        # --- smooth terrain: 3-pass bilinear blur (1/8 resolution) ---
        bw1 = max(32, rect.width // 2)
        bh1 = max(32, rect.height // 2)
        bw2 = max(16, bw1 // 2)
        bh2 = max(16, bh1 // 2)
        bw3 = max(8, bw2 // 2)
        bh3 = max(8, bh2 // 2)
        s1 = pygame.transform.smoothscale(layer, (bw1, bh1))
        s2 = pygame.transform.smoothscale(s1, (bw2, bh2))
        s3 = pygame.transform.smoothscale(s2, (bw3, bh3))
        blurred = pygame.transform.smoothscale(s3, rect.size)
        layer.fill((0, 0, 0, 0))
        layer.blit(blurred, (0, 0))

        # --- near-edge fill (below closest terrain row to screen bottom) ---
        near_pts = [proj(ix, 0) for ix in range(NX)]
        pygame.draw.polygon(
            layer,
            (52, 35, 24, a255),
            list(near_pts) + [(near_pts[-1][0], _rb), (near_pts[0][0], _rb)],
        )
        nc = (185, 148, 118, a255)
        for k in range(len(near_pts) - 1):
            pygame.draw.line(layer, nc, near_pts[k], near_pts[k + 1], 2)

        # --- horizon silhouette (farthest row) ---
        far_pts = [proj(ix, NZ - 1) for ix in range(NX)]
        # Blur softens the terrain upward; nudge the outline down so it meets the visible ridge.
        horizon_dy = max(1, min(6, int(2 + rect.height * 0.0028)))
        hzc = (125, 95, 78, max(0, min(255, int(a255 * 0.80))))
        bt_l = max(0, _bt - rect.top)
        clip_h = max(1, min(rect.height, _rb - rect.top) - bt_l)
        prev_clip = layer.get_clip()
        layer.set_clip(pygame.Rect(0, bt_l, rect.width, clip_h))
        try:
            for k in range(len(far_pts) - 1):
                p0 = (far_pts[k][0], far_pts[k][1] + horizon_dy)
                p1 = (far_pts[k + 1][0], far_pts[k + 1][1] + horizon_dy)
                pygame.draw.line(layer, hzc, p0, p1, 2)
        finally:
            layer.set_clip(prev_clip)

        # --- probe shadow on terrain ---
        if alt < 300.0:
            sh_sx, sh_sy = proj_world(probe_x, probe_z, h_ref)
            sh_a = max(0, min(130, int(150 * (1.0 - alt / 300.0))))
            sh_rx = max(2, int(3.5 * cam_sc))
            sh_ry = max(1, int(sh_rx * 0.38))
            if sh_a > 6 and _bt <= sh_sy <= _rb:
                sh_surf = pygame.Surface((sh_rx * 2, sh_ry * 2), pygame.SRCALPHA)
                pygame.draw.ellipse(sh_surf, (12, 10, 6, sh_a), sh_surf.get_rect())
                layer.blit(sh_surf, (sh_sx - sh_rx, sh_sy - sh_ry))

        # --- splash rings on water ---
        if self._water_splash is not None:
            sp = self._water_splash
            sp_t = sp["t"] / max(0.01, sp["dur"])
            sp_alpha = max(0, int(180 * (1.0 - sp_t)))
            sp_r_m = sp["max_r_m"] * sp_t
            sp_h = float(model.world.height_m_at(sp["x_m"], sp["z_m"]))
            sp_sx, sp_sy = proj_world(sp["x_m"], sp["z_m"], sp_h)
            if sp_alpha > 4 and _bt <= sp_sy <= _rb:
                for ring_f in (1.0, 0.6, 0.3):
                    rr_m = sp_r_m * ring_f
                    rrx = max(1, int(rr_m * cam_sc))
                    rry = max(1, int(rrx * 0.38))
                    ra = max(0, int(sp_alpha * ring_f))
                    if rrx > 1 and rry > 1 and ra > 4:
                        rs = pygame.Surface((rrx * 2 + 4, rry * 2 + 4), pygame.SRCALPHA)
                        pygame.draw.ellipse(
                            rs, (138, 124, 108, ra),
                            (2, 2, rrx * 2, rry * 2), max(1, int(cam_sc * 0.3)))
                        layer.blit(rs, (sp_sx - rrx - 2, sp_sy - rry - 2))

        # --- target flag with real perspective scaling ---
        tgt_wx = float(model.target_x_m)
        tgt_wz = float(model.target_z_m)
        if (gx0 <= tgt_wx <= gx0 + (NX - 1) * step_x
                and gz0 <= tgt_wz <= gz0 + (NZ - 1) * step_z):
            tgt_h = float(model.world.height_m_at(tgt_wx, tgt_wz))
            tgt_sx, tgt_sy = proj_world(tgt_wx, tgt_wz, tgt_h)
            if _bt <= tgt_sy <= _rb:
                flag_m = 7.0
                pole_h = max(1, int(flag_m * cam_sc))
                pole_top = tgt_sy - pole_h
                pole_w = max(1, min(3, int(cam_sc * 0.4)))
                pygame.draw.line(layer, (215, 195, 85, a255),
                                 (tgt_sx, tgt_sy), (tgt_sx, pole_top), pole_w)
                fw = max(2, pole_h * 2 // 3)
                fh = max(2, pole_h // 3)
                pygame.draw.polygon(
                    layer, (240, 65, 45, a255),
                    [(tgt_sx + 1, pole_top),
                     (tgt_sx + 1 + fw, pole_top + fh // 2),
                     (tgt_sx + 1, pole_top + fh)],
                )
                dot_r = max(1, int(cam_sc * 0.5))
                pygame.draw.circle(layer, (255, 220, 100, a255),
                                   (tgt_sx, tgt_sy), dot_r, 0)

        screen.blit(layer, rect.topleft)

        # --- probe position ---
        if alpha >= 0.008:
            pscale = self._probe_scale(model)
            body_r = int(16 * pscale)
            alt_c = max(0.0, float(model.altitude_m))
            sy_probe = int(_by - alt_c * _hp - body_r)
            sy_probe = max(_bt + body_r + 4, min(flat_y - body_r - 2, sy_probe))
            self._profile_probe_xy = (_cx, sy_probe)

    def _probe_scale(self, model: PhysicsModel) -> float:
        h_km = model.altitude_m / 1000.0
        return float(np.clip(1.2 - 0.0006 * h_km, 0.5, 1.2))

    def _heatshield_color(self, model: PhysicsModel) -> tuple[int, int, int]:
        # Char from dynamic-pressure proxy; incandescent tint from thermal skin model.
        v = float(model.air_rel_speed_mps)
        q = float(model.atm_density_kg_m3) * (v * v)  # proxy (same |v_rel| family as skin heating)
        t_char = float(np.clip((q - 50_000.0) / 450_000.0, 0.0, 1.0))
        t_char = t_char * t_char
        base = np.array([220, 205, 190], dtype=float)
        char = np.array([35, 30, 28], dtype=float)
        c = base * (1 - t_char) + char * t_char
        t_skin = float(model.heatshield_skin_temp_c)
        inc = float(np.clip((t_skin - 380.0) / 920.0, 0.0, 1.0))
        hot = np.array([255, 118, 55], dtype=float)
        c = c * (1 - inc) + hot * inc
        c = np.clip(c, 0, 255).astype(int)
        return int(c[0]), int(c[1]), int(c[2])

    def _heatshield_glow_alpha(self, model: PhysicsModel) -> float:
        """0..1 for additive glow around the shell (re-entry heating)."""
        if model.heatshield_jettisoned:
            return 0.0
        t_skin = float(model.heatshield_skin_temp_c)
        # Onset ~400 °C so plausible peak (~650 °C in nominal entry) reads clearly.
        return float(np.clip((t_skin - 400.0) / 650.0, 0.0, 1.0))

    def _draw_probe(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        scale = self._probe_scale(model)

        x, y = self._probe_screen_xy(rect, model)

        sinking = self._sink_t > 0.0 and model.water_landed
        if sinking:
            sink_offset_px = int(self._sink_t * 4.0 * self.camera.scale_px_per_m)
            y += sink_offset_px
            sink_alpha = max(0, int(255 * (1.0 - self._sink_t / 1.5)))
            if sink_alpha <= 0:
                return
            if self._sink_t < 1.5:
                bub_chance = 0.4 if self._sink_t < 0.6 else 0.15
                if float(self._rng.random()) < bub_chance:
                    bx = x + float(self._rng.normal(0, 6))
                    by = float(y - 4)
                    self._particles.append((bx, by, float(self._rng.normal(0, 3)),
                                            float(self._rng.uniform(-25, -10)),
                                            float(self._rng.uniform(0.4, 1.0)),
                                            (92, 84, 74)))

        body_r = int(16 * scale)
        inner_color = (245, 245, 248)

        if sinking:
            sz = body_r * 3 + 4
            tmp = pygame.Surface((sz, sz), pygame.SRCALPHA)
            cx_t, cy_t = sz // 2, sz // 2
            if not model.heatshield_jettisoned:
                self._heatshield_rgb = self._heatshield_color(model)
                shell_r = int(body_r * 1.28)
                glow = self._heatshield_glow_alpha(model)
                if glow > 0.04:
                    for li in range(3):
                        gr = int(shell_r + 3 + li * max(4, int(5 * scale)))
                        alpha = int(sink_alpha * 0.35 * glow * (1.0 - 0.28 * li))
                        if alpha < 4:
                            continue
                        box = 2 * gr + 4
                        layer = pygame.Surface((box, box), pygame.SRCALPHA)
                        ccx, ccy = box // 2, box // 2
                        pygame.draw.circle(
                            layer,
                            (255, min(255, 120 + int(80 * glow)), 50, alpha),
                            (ccx, ccy),
                            gr,
                        )
                        tmp.blit(layer, (cx_t - ccx, cy_t - ccy), special_flags=pygame.BLEND_RGBA_ADD)
                pygame.draw.circle(tmp, (*self._heatshield_rgb, sink_alpha), (cx_t, cy_t), shell_r)
                pygame.draw.circle(tmp, (80, 75, 70, sink_alpha), (cx_t, cy_t), shell_r, width=2)
            pygame.draw.circle(tmp, (*inner_color, sink_alpha), (cx_t, cy_t), body_r)
            pygame.draw.circle(tmp, (90, 85, 80, sink_alpha), (cx_t, cy_t), body_r, width=2)
            pygame.draw.line(tmp, (180, 175, 165, sink_alpha),
                             (cx_t, cy_t - body_r), (cx_t, cy_t - body_r - int(14 * scale)), 2)
            screen.blit(tmp, (x - cx_t, y - cy_t))
            return

        # heatshield shell (outer) + inner probe (white)
        if not model.heatshield_jettisoned:
            self._heatshield_rgb = self._heatshield_color(model)
            shell_r = int(body_r * 1.28)
            glow = self._heatshield_glow_alpha(model)
            if glow > 0.04:
                for li in range(3):
                    gr = int(shell_r + 3 + li * max(4, int(5 * scale)))
                    alpha = int(55 * glow * (1.0 - 0.28 * li))
                    if alpha < 6:
                        continue
                    box = 2 * gr + 4
                    layer = pygame.Surface((box, box), pygame.SRCALPHA)
                    cx, cy = box // 2, box // 2
                    pygame.draw.circle(
                        layer,
                        (255, min(255, 120 + int(80 * glow)), 50, alpha),
                        (cx, cy),
                        gr,
                    )
                    screen.blit(layer, (x - cx, y - cy), special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.circle(screen, self._heatshield_rgb, (x, y), shell_r)
            pygame.draw.circle(screen, (80, 75, 70), (x, y), shell_r, width=2)

        # inner body
        pygame.draw.circle(screen, inner_color, (x, y), body_r)
        pygame.draw.circle(screen, (90, 85, 80), (x, y), body_r, width=2)

        # antenna
        pygame.draw.line(screen, (180, 175, 165), (x, y - body_r), (x, y - body_r - int(14 * scale)), 2)

        # chute lines
        # chute lines / canopy
        if (model.drogue_deployed or model.main_deployed):
            prog = self._chute_progress(main=model.main_deployed)
            alpha_mul = 1.0
            # Smooth transition on jettison: briefly keep canopy while fading/collapsing.
            if model.chute_jettisoned:
                if self._jettison_anim_t is None:
                    prog = 0.0
                    alpha_mul = 0.0
                else:
                    u = float(np.clip(self._jettison_anim_t / 0.55, 0.0, 1.0))
                    alpha_mul = (1.0 - u) ** 2
                    prog *= (1.0 - 0.75 * u)
            if alpha_mul > 0.02 and prog > 0.02:
                self._draw_parachute(screen, (x, y), scale, main=model.main_deployed, progress=prog, alpha_mul=alpha_mul)

        # engine flame (if thrusting)
        if self._engine_spool > 0.02 and model.throttle_0_1 > 0 and model.fuel_kg > 0:
            self._draw_engine_flame(
                screen,
                nozzle_xy=(x, y + body_r),
                scale=scale,
                throttle=float(model.throttle_0_1),
                spool=float(self._engine_spool),
            )

    def _draw_engine_flame(
        self,
        screen: pygame.Surface,
        nozzle_xy: tuple[int, int],
        scale: float,
        throttle: float,
        spool: float,
    ) -> None:
        # Dense plume made of layered ellipses (no triangle).
        x0, y0 = nozzle_xy
        thr = float(np.clip(throttle, 0.0, 1.0))
        sp = float(np.clip(spool, 0.0, 1.0))
        power = thr * sp
        if power <= 0.01:
            return

        # Height depends on user throttle; width only slightly.
        length = (18 + 70 * power) * scale
        width = (12 + 4 * power) * scale
        w = int(max(18, width * 2.6))
        h = int(max(22, length * 1.25))
        need = (w, h)
        if self._flame_surf is None or self._flame_surf_size != need:
            self._flame_surf = pygame.Surface(need, pygame.SRCALPHA)
            self._flame_surf_size = need
        surf = self._flame_surf
        surf.fill((0, 0, 0, 0))

        cx = w // 2
        # Build a dense teardrop by stacking puffs.
        n = 14
        flick = 0.85 + 0.15 * math.sin(self._t * (18.0 + 8.0 * thr) + 0.9)
        for i in range(n):
            u = i / (n - 1)
            # shift downwards with u
            yy = int(h * (0.10 + 0.82 * u * flick))
            # radius decays with u
            rx = int(max(2, (width * (1.15 - 0.95 * u))))
            ry = int(max(2, (width * (0.95 - 0.70 * u))))
            # color gradient: white/yellow core -> orange -> faint red
            if u < 0.22:
                col = (255, 245, 220)
            elif u < 0.55:
                col = (255, 200, 110)
            else:
                col = (255, 120, 70)
            # Keep it mostly opaque; fade only near tail.
            a = int(np.clip((245 - 35 * u) * (0.55 + 0.45 * power), 0, 255))
            pygame.draw.ellipse(surf, (*col, a), pygame.Rect(cx - rx, yy - ry, 2 * rx, 2 * ry))

        # Add a small bright core near nozzle.
        core_r = int(max(2, (6 + 7 * power) * scale))
        pygame.draw.circle(surf, (255, 250, 235, 255), (cx, int(h * 0.14)), core_r)

        # Draw under the probe.
        # Normal alpha blending looks better with our non-premultiplied colors.
        screen.blit(surf, (x0 - w // 2, y0 + int(2 * scale)))

    def _draw_parachute(
        self,
        screen: pygame.Surface,
        probe_xy: tuple[int, int],
        scale: float,
        main: bool,
        progress: float,
        alpha_mul: float = 1.0,
    ) -> None:
        x, y = probe_xy
        p = float(np.clip(progress, 0.0, 1.0))
        # Slight overshoot for main for a "snap" feel.
        if main:
            p2 = min(1.12, 1.05 * (1 - (1 - p) ** 2))
        else:
            p2 = p

        chute_w = int((120 if main else 60) * scale * (0.35 + 0.65 * p2))
        chute_h = int((40 if main else 24) * scale * (0.25 + 0.75 * p2))
        cx, cy = x, y - int((120 if main else 90) * scale)
        rect = pygame.Rect(0, 0, chute_w, chute_h)
        rect.center = (cx, cy)

        wth = 2 if p < 0.3 else 3
        a = int(np.clip(255 * float(alpha_mul), 0, 255))
        buf_w = max(1, chute_w + 12)
        buf_h = max(1, abs(cy - y) + chute_h + 20)
        buf_ox = cx - buf_w // 2
        buf_oy = min(cy - chute_h // 2, y) - 6
        buf = pygame.Surface((buf_w, buf_h), pygame.SRCALPHA)
        def _l(px: int, py: int) -> tuple[int, int]:
            return (px - buf_ox, py - buf_oy)
        arc_r = pygame.Rect(0, 0, chute_w, chute_h)
        arc_r.center = _l(cx, cy)
        pygame.draw.arc(buf, (235, 235, 240, a), arc_r, math.pi, 2 * math.pi, wth)
        n_lines = 4 if p < 0.35 else 7
        for t in np.linspace(-0.45, 0.45, n_lines):
            sx = cx + int((chute_w / 2) * t)
            tip_y = y - int((12 * scale) * (0.4 + 0.6 * p))
            pygame.draw.line(buf, (210, 210, 215, int(a * 0.8)), _l(sx, cy), _l(x, tip_y), 1)
        screen.blit(buf, (buf_ox, buf_oy))

    def _draw_effects(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel, frame_dt: float) -> None:
        # Stop entry/engine VFX after success, or after failure once on/near the surface.
        alt = float(model.altitude_m)
        landed = model.result == SimResult.SUCCESS or (model.failed and alt <= 3.0)

        # entry glow when fast and high-ish density
        if abs(model.vertical_speed_mps) > 1200 and model.atm_density_kg_m3 > 0.05 and not landed:
            self._emit_particles(model, rect, count=2, color=(255, 180, 80))

        if not landed and self._engine_spool > 0.05 and model.throttle_0_1 > 0 and model.fuel_kg > 0:
            # Emit exactly at nozzle position in screen space (under probe, behind flame).
            scale = self._probe_scale(model)
            x_px, y_px = self._probe_screen_xy(rect, model)
            body_r = int(16 * scale)
            nozzle_px = (x_px, y_px + body_r + int(2 * scale))

            bright_n = int(2 + 5 * self._engine_spool * float(model.throttle_0_1))
            self._emit_particles_at_screen(
                count=bright_n,
                x_px=float(nozzle_px[0]),
                y_px=float(nozzle_px[1]),
                color=(255, 120, 60),
                vx_sigma=16.0,
                vy_mu=55.0,
                vy_sigma=22.0,
            )
            self._emit_particles_at_screen(
                count=1,
                x_px=float(nozzle_px[0]),
                y_px=float(nozzle_px[1]) + 6.0,
                color=(90, 80, 75),
                vx_sigma=12.0,
                vy_mu=70.0,
                vy_sigma=28.0,
            )

        if not landed:
            self._draw_wind_pennant(screen, rect, model)

        self._step_particles(rect, dt=float(frame_dt))
        self._draw_particles(screen, rect)

    def _emit_particles_at_screen(
        self,
        count: int,
        x_px: float,
        y_px: float,
        color: tuple[int, int, int],
        vx_sigma: float,
        vy_mu: float,
        vy_sigma: float,
    ) -> None:
        # Emit particles at explicit screen position.
        for _ in range(count):
            px = x_px + float(self._rng.normal(0, 3))
            py = y_px + float(self._rng.normal(0, 2))
            vx = float(self._rng.normal(0, vx_sigma))
            vy = float(self._rng.normal(vy_mu, vy_sigma))
            life = float(self._rng.uniform(0.18, 0.50))
            self._particles.append((px, py, vx, vy, life, color))

    def _update_event_anims(self, model: PhysicsModel, dt: float) -> None:
        flags = {
            "heatshield": bool(model.heatshield_jettisoned),
            "drogue": bool(model.drogue_deployed),
            "main": bool(model.main_deployed),
            "jettison": bool(model.chute_jettisoned),
            "engine": bool(model.engine_on),
        }
        if self._prev_flags is None:
            self._prev_flags = flags
        prev = self._prev_flags

        # detect transitions
        if (not prev["heatshield"]) and flags["heatshield"]:
            self._spawn_heatshield_frag(model)
        if (not prev["drogue"]) and flags["drogue"]:
            self._drogue_anim_t = 0.0
        if (not prev["main"]) and flags["main"]:
            self._main_anim_t = 0.0
        if (not prev["jettison"]) and flags["jettison"]:
            self._spawn_chute_jettison_frags(model)
            self._jettison_anim_t = 0.0

        self._prev_flags = flags

        # advance timers
        if self._drogue_anim_t is not None:
            self._drogue_anim_t += dt
            if self._drogue_anim_t > 1.2:
                self._drogue_anim_t = None
        if self._main_anim_t is not None:
            self._main_anim_t += dt
            if self._main_anim_t > 1.6:
                self._main_anim_t = None
        if self._jettison_anim_t is not None:
            self._jettison_anim_t += dt
            if self._jettison_anim_t > 0.55:
                self._jettison_anim_t = None

        # engine spool (smooth)
        target = 1.0 if model.engine_on else 0.0
        rate = 7.0
        self._engine_spool += (target - self._engine_spool) * float(np.clip(rate * dt, 0.0, 1.0))

        # water landing: splash + sink
        now_water = bool(model.water_landed)
        if now_water and not self._was_water_landed:
            self._water_splash = {
                "x_m": float(model.pos_x_m),
                "z_m": float(model.pos_z_m),
                "t": 0.0,
                "dur": 3.0,
                "max_r_m": 40.0,
            }
            self._sink_t = 0.0
            for _ in range(8):
                ang_r = float(self._rng.uniform(0, 2 * math.pi))
                spd = float(self._rng.uniform(8, 35))
                self._event_frags.append(
                    EventFrag(
                        kind="drop",
                        x_m=float(model.pos_x_m),
                        y_m=0.0,
                        vx_mps=math.cos(ang_r) * spd,
                        vy_mps=float(self._rng.uniform(15, 45)),
                        ang=0.0,
                        ang_v=0.0,
                        t=0.0,
                        dur=1.8,
                        scale=float(self._rng.uniform(0.3, 0.8)),
                        color=(88, 80, 72),
                    )
                )
        self._was_water_landed = now_water
        if now_water:
            self._sink_t += dt
        if self._water_splash is not None:
            self._water_splash["t"] += dt
            if self._water_splash["t"] >= self._water_splash["dur"]:
                self._water_splash = None

        # frags
        kept: list[EventFrag] = []
        for f in self._event_frags:
            f.t += dt
            if f.t >= f.dur:
                continue
            f.x_m += f.vx_mps * dt
            f.y_m += f.vy_mps * dt
            f.vy_mps -= 0.4 * 9.81 * dt
            f.ang += f.ang_v * dt
            kept.append(f)
        self._event_frags = kept[-24:]

    def _chute_progress(self, main: bool) -> float:
        # Two-stage: 0..0.4 pilot/lines, 0.4..1.2 canopy.
        t = self._main_anim_t if main else self._drogue_anim_t
        if t is None:
            return 1.0
        if t <= 0.4:
            return float(np.clip(t / 0.4, 0.0, 1.0)) * 0.35
        return 0.35 + float(np.clip((t - 0.4) / (1.2 if not main else 1.6), 0.0, 1.0)) * 0.65

    def _spawn_heatshield_frag(self, model: PhysicsModel) -> None:
        # spawn near probe in world coords
        x0, y0 = float(model.pos_x_m), float(max(0.0, model.altitude_m))
        wind = float(model.wind_vec_mps_at(model.altitude_m)[0])
        vx = float(0.3 * (model.speed_x_mps - wind) + self._rng.normal(0, 2.0))
        vy = float(-18.0 + self._rng.normal(0, 2.5))
        frag = EventFrag(
            kind="heatshield",
            x_m=x0,
            y_m=y0,
            vx_mps=vx,
            vy_mps=vy,
            ang=float(self._rng.uniform(-0.6, 0.6)),
            ang_v=float(self._rng.uniform(-2.2, 2.2)),
            t=0.0,
            dur=float(self._rng.uniform(2.2, 3.6)),
            scale=1.0,
            color=self._heatshield_rgb,
        )
        self._event_frags.append(frag)
        # short soot puff (at the probe position)
        self._emit_particles(model, rect=pygame.Rect(0, 0, self.w, self.h), count=6, color=(80, 70, 65))

    def _spawn_chute_jettison_frags(self, model: PhysicsModel) -> None:
        x0, y0 = float(model.pos_x_m), float(max(0.0, model.altitude_m))
        wind = float(model.wind_vec_mps_at(model.altitude_m)[0])
        v_air = float(model.speed_x_mps - wind)
        # 2 small fragments, drifting with airflow
        for k in range(2):
            self._event_frags.append(
                EventFrag(
                    kind="chute",
                    x_m=x0,
                    y_m=y0 + 20.0,
                    vx_mps=float(0.6 * v_air + self._rng.normal(0, 3.0)),
                    vy_mps=float(12.0 + self._rng.normal(0, 2.0) + 2.0 * k),
                    ang=float(self._rng.uniform(-1.0, 1.0)),
                    ang_v=float(self._rng.uniform(-1.5, 1.5)),
                    t=0.0,
                    dur=float(self._rng.uniform(1.2, 2.0)),
                    scale=float(self._rng.uniform(0.8, 1.15)),
                )
            )

    def _draw_event_frags(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        # Skip rotate/blit for fragments far outside the view.
        cull_m = 120
        for f in self._event_frags:
            a = float(np.clip(1.0 - (f.t / max(1e-6, f.dur)), 0.0, 1.0))
            alpha = int(255 * a)
            sx, sy = self._world_to_screen(rect, f.x_m, f.y_m)
            if (
                sx < rect.left - cull_m
                or sx > rect.right + cull_m
                or sy < rect.top - cull_m
                or sy > rect.bottom + cull_m
            ):
                continue
            if f.kind == "heatshield":
                r = int(18 * self._probe_scale(model) * f.scale)
                tmp = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
                pygame.draw.arc(tmp, (*f.color, alpha), tmp.get_rect(), math.pi, 2 * math.pi, 5)
                rot = pygame.transform.rotate(tmp, math.degrees(f.ang))
                screen.blit(rot, rot.get_rect(center=(sx, sy)))
            elif f.kind == "chute":
                # More parachute-like fragment: small canopy + ribs + a few lines.
                pscale = self._probe_scale(model) * f.scale
                # Slight "collapse" over lifetime: canopy flattens a bit.
                u = float(np.clip(f.t / max(1e-6, f.dur), 0.0, 1.0))
                collapse = 1.0 - 0.25 * u

                # Fade-in to avoid popping at spawn, then fade out.
                fade_in = float(np.clip(f.t / 0.12, 0.0, 1.0))
                fade_out = float(np.clip((1.0 - u) / 0.25, 0.0, 1.0))
                a_mul = fade_in * min(1.0, fade_out)
                alpha = int(alpha * a_mul)

                w = int(68 * pscale)
                h = int(34 * pscale * collapse)
                tmp = pygame.Surface((w + 4, h + 34), pygame.SRCALPHA)

                cx = (w + 4) // 2
                cy = int(h * 0.55) + 2
                arc_rect = pygame.Rect(0, 0, w, h)
                arc_rect.center = (cx, cy)

                # canopy fill (semi-opaque)
                fill_col = (235, 235, 240, int(alpha * 0.55))
                pygame.draw.ellipse(tmp, fill_col, arc_rect)
                # mask lower half to keep it a dome
                pygame.draw.rect(tmp, (0, 0, 0, 0), pygame.Rect(0, cy, w + 4, h))

                # canopy outline + ribs
                pygame.draw.arc(tmp, (235, 235, 240, alpha), arc_rect, math.pi, 2 * math.pi, 3)
                ribs = 6
                for t in np.linspace(-0.45, 0.45, ribs):
                    sx2 = cx + int((w / 2) * t)
                    pygame.draw.line(tmp, (210, 210, 215, int(alpha * 0.75)), (sx2, cy), (cx, cy + int(h * 0.10)), 1)

                # a few lines (shrouds)
                line_y = cy + int(h * 0.10)
                tip = (cx, line_y + int(20 * pscale))
                for t in np.linspace(-0.35, 0.35, 4):
                    sx3 = cx + int((w / 2) * t)
                    pygame.draw.line(tmp, (210, 210, 215, int(alpha * 0.55)), (sx3, cy), tip, 1)

                rot = pygame.transform.rotate(tmp, math.degrees(f.ang))
                screen.blit(rot, rot.get_rect(center=(sx, sy)))
            elif f.kind == "drop":
                dr = max(2, int(4 * f.scale))
                ds = pygame.Surface((dr * 2 + 2, dr * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(ds, (*f.color, alpha), (dr + 1, dr + 1), dr)
                screen.blit(ds, (sx - dr - 1, sy - dr - 1))

    def _emit_particles(self, model: PhysicsModel, rect: pygame.Rect, count: int, color: tuple[int, int, int]) -> None:
        x, y = self._probe_screen_xy(rect, model)
        for _ in range(count):
            px = x + float(self._rng.normal(0, 4))
            py = y + float(self._rng.normal(0, 4))
            vx = float(self._rng.normal(0, 20))
            vy = float(self._rng.normal(10, 30))
            life = float(self._rng.uniform(0.2, 0.6))
            size = float(self._rng.uniform(2.0, 5.0))
            self._particles.append((px, py, vx, vy, life, color))

    def _step_particles(self, rect: pygame.Rect, dt: float) -> None:
        new = []
        for px, py, vx, vy, life, color in self._particles:
            life -= dt
            if life <= 0:
                continue
            px += vx * dt
            py += vy * dt
            vy += 40 * dt
            if px < rect.left - 50 or px > rect.right + 50 or py > rect.bottom + 80:
                continue
            new.append((px, py, vx, vy, life, color))
        self._particles = new[-800:]

    def _draw_particles(self, screen: pygame.Surface, rect: pygame.Rect) -> None:
        if not self._particles:
            return
        need = (rect.width, rect.height)
        if self._particle_buf is None or self._particle_buf.get_size() != need:
            self._particle_buf = pygame.Surface(need, pygame.SRCALPHA)
        buf = self._particle_buf
        buf.fill((0, 0, 0, 0))
        for px, py, _vx, _vy, life, color in self._particles:
            a = int(max(0, min(255, int(255 * (life / 0.6)))))
            pygame.draw.circle(buf, (*color, a), (int(px) - rect.left, int(py) - rect.top), 2)
        screen.blit(buf, rect.topleft)

    def _make_cloud_layers(self) -> list[CloudLayer]:
        self._haze_pools = self._build_all_haze_pools()
        layers: list[CloudLayer] = []
        for shape_kind, deck_alpha, seed in [
            (0, 0.42, 11),
            (1, 0.54, 22),
            (2, 0.58, 33),
            (3, 0.54, 44),
            (4, 0.68, 55),
        ]:
            da = float(deck_alpha)
            blobs = self._make_cloud_blobs(seed=seed, shape_kind=shape_kind, deck_alpha=da)
            pool = self._haze_pools[shape_kind]
            layers.append(
                CloudLayer(shape_kind=shape_kind, deck_alpha=da, ref_sprite=pool[0], blobs=blobs)
            )
        return layers

    @staticmethod
    def _haze_blur_np(a: np.ndarray, passes: int) -> np.ndarray:
        """Isotropic-ish smoothing (cardinals + diagonals) — avoids axis-aligned corners."""
        x = np.asarray(a, dtype=np.float64)
        for _ in range(passes):
            c = (
                np.roll(x, 1, 0)
                + np.roll(x, -1, 0)
                + np.roll(x, 1, 1)
                + np.roll(x, -1, 1)
            )
            d = (
                np.roll(np.roll(x, 1, 0), 1, 1)
                + np.roll(np.roll(x, 1, 0), -1, 1)
                + np.roll(np.roll(x, -1, 0), 1, 1)
                + np.roll(np.roll(x, -1, 0), -1, 1)
            )
            x = (3.2 * x + c + 0.62 * d) / (3.2 + 4.0 + 4.0 * 0.62)
        return np.clip(x, 0.0, 1.0)

    @staticmethod
    def _haze_shape_envelope(shape_kind: int, gh: int, gw: int, rng: np.random.Generator) -> np.ndarray:
        """Non-rectangular silhouettes: α→0 toward edges (organic mist, not a solid quad)."""
        ii = np.linspace(0.0, 1.0, gh, dtype=np.float64)[:, None]
        jj = np.linspace(0.0, 1.0, gw, dtype=np.float64)[None, :]
        sk = int(shape_kind) % _HAZE_SHAPE_COUNT

        if sk == 0:
            dx = (jj - 0.48) / (0.52 + 0.06 * rng.random())
            dy = (ii - 0.48) / (0.48 + 0.05 * rng.random())
            r2 = dx * dx + dy * dy
            env = np.exp(-np.clip(r2, 0.0, None) / (0.42 + 0.08 * rng.random())) ** (
                0.95 + 0.08 * rng.random()
            )
        elif sk == 1:
            vy = np.sin(np.pi * ii) ** (0.88 + 0.08 * rng.random())
            j0 = 0.45 + 0.08 * rng.random()
            sig = 0.28 + 0.06 * rng.random()
            hx = np.exp(-(((jj - j0) / sig) ** 2))
            env = np.clip(vy * hx, 0.0, 1.0)
        elif sk == 2:
            t = jj * 0.72 + ii * 0.28
            t0 = 0.38 + 0.12 * rng.random()
            sig = 0.16 + 0.05 * rng.random()
            ridge = np.exp(-(((t - t0) / sig) ** 2))
            env = (np.sin(np.pi * ii) ** 0.78) * ridge * (0.55 + 0.45 * jj)
            env = np.clip(env, 0.0, 1.0)
        elif sk == 3:
            g1 = np.exp(-(((ii - 0.36) ** 2 + (jj - 0.34) ** 2) / (0.055 + 0.02 * rng.random())))
            g2 = np.exp(-(((ii - 0.58) ** 2 + (jj - 0.64) ** 2) / (0.048 + 0.02 * rng.random())))
            env = np.clip(g1 + 0.82 * g2, 0.0, 1.0) ** (0.92 + 0.06 * rng.random())
        else:
            cx = 0.78 + 0.08 * rng.random()
            cy = 0.22 + 0.08 * rng.random()
            env = np.exp(-(((1.0 - jj) - cx) ** 2 + (ii - cy) ** 2) / (0.065 + 0.025 * rng.random()))
            env *= np.sin(np.pi * ii) ** (0.62 + 0.12 * rng.random())

        return np.clip(env, 0.0, 1.0)

    def _make_haze_sheet(self, rng: np.random.Generator, shape_kind: int, variant: int) -> pygame.Surface:
        """Noise × envelope → irregular silhouette; upscale keeps edges soft."""
        sk = int(shape_kind) % _HAZE_SHAPE_COUNT
        vi = int(variant) % max(1, _HAZE_VARIANTS_PER_SHAPE)
        gw = int(36 + (vi % 4) * 8 + rng.integers(-2, 4))
        gh = int(28 + (vi % 5) * 6 + rng.integers(-2, 4))
        gw = max(26, min(80, gw))
        gh = max(22, min(64, gh))

        n = rng.random((gh, gw), dtype=np.float64)
        n = Renderer._haze_blur_np(n, 8 + (vi % 4))
        n = np.clip(n * float(rng.uniform(0.85, 1.14)), 0.0, 1.0)
        n = np.power(n, 0.72 + 0.06 * float(sk))
        env = Renderer._haze_shape_envelope(sk, gh, gw, rng)
        env = Renderer._haze_blur_np(env, 5)
        n = np.clip(n * env, 0.0, 1.0)
        n = Renderer._haze_blur_np(n, 3)
        n = np.power(np.clip(n, 0.0, 1.0), 0.92)

        amp = 78.0 + 22.0 * float(vi % 4) + float(rng.uniform(-4.0, 14.0))
        alpha = np.clip(np.power(n, 1.0 + 0.03 * float(sk)) * amp, 0.0, 178.0)

        br = 192.0 + 26.0 * rng.random()
        bg = 142.0 + 48.0 * rng.random()
        bb = 102.0 + 42.0 * rng.random()
        nr = np.clip(br + 44.0 * n, 0.0, 255.0).astype(np.uint8)
        ng = np.clip(bg + 36.0 * n, 0.0, 255.0).astype(np.uint8)
        nb = np.clip(bb + 28.0 * n, 0.0, 255.0).astype(np.uint8)

        rgba = np.stack([nr, ng, nb, alpha.astype(np.uint8)], axis=-1)
        small = pygame.image.frombuffer(
            np.ascontiguousarray(rgba).tobytes(), (gw, gh), "RGBA"
        ).copy()

        if sk == 1:
            tw = int(rng.uniform(520, 820))
            th = int(rng.uniform(100, 190))
        elif sk == 2:
            tw = int(rng.uniform(420, 680))
            th = int(rng.uniform(160, 280))
        elif sk == 0:
            tw = int(rng.uniform(420, 620))
            th = int(rng.uniform(200, 340))
        else:
            tw = int(rng.uniform(460, 760))
            th = int(rng.uniform(140, 270))
        return pygame.transform.smoothscale(small, (tw, th))

    def _build_all_haze_pools(self) -> list[list[pygame.Surface]]:
        rng = np.random.default_rng(77102)
        pools: list[list[pygame.Surface]] = []
        for kind in range(_HAZE_SHAPE_COUNT):
            pools.append(
                [self._make_haze_sheet(rng, shape_kind=kind, variant=i) for i in range(_HAZE_VARIANTS_PER_SHAPE)]
            )
        return pools

    def _make_cloud_blobs(
        self, seed: int, shape_kind: int, deck_alpha: float
    ) -> list[tuple[float, float, float, float, float, float, float, pygame.Surface]]:
        rng = np.random.default_rng(seed)
        pool = self._haze_pools[shape_kind]
        blobs: list[tuple[float, float, float, float, float, float, float, pygame.Surface]] = []
        alt_max = 120_000.0
        count = 6
        for _ in range(count):
            spr0 = pool[int(rng.integers(0, len(pool)))]
            x = float(rng.uniform(-self.w * 0.45, self.w * 0.12))
            u = float(rng.uniform(0.0, 1.0))
            alt_m = float((u**1.6) * alt_max)
            yoff = float(rng.uniform(-self.h * 0.16, self.h * 0.12))
            inst = float(rng.uniform(0.68, 1.0))
            scale = float(rng.uniform(0.88, 1.58))
            wob_phase = float(rng.uniform(0.0, 2 * math.pi))
            sw = max(24, int(spr0.get_width() * scale))
            sh = max(24, int(spr0.get_height() * scale))
            spr2 = pygame.transform.smoothscale(spr0, (sw, sh))
            combined = float(np.clip(float(deck_alpha) * inst, 0.22, 0.99))
            spr2.set_alpha(int(np.clip(255 * combined, 32, 255)))
            blobs.append((x, alt_m, yoff, 1.0, combined, scale, wob_phase, spr2))
        return blobs

    def _spawn_haze_blob(
        self,
        layer: CloudLayer,
        rect: pygame.Rect,
        center_m: float,
        span_m: float,
        rng: np.random.Generator,
        density_boost: float = 1.0,
    ) -> tuple[float, float, float, pygame.Surface]:
        pool = self._haze_pools[layer.shape_kind]
        if not pool:
            spr0 = pygame.Surface((120, 80), pygame.SRCALPHA)
            spr0.fill((195, 160, 130, 22))
            spr0.set_alpha(int(255 * np.clip(layer.deck_alpha * 0.62 * density_boost, 0.0, 1.0)))
            return (
                float(rng.uniform(rect.left, rect.right)),
                float(np.clip(center_m - 0.55 * span_m, 0.0, 120_000.0)),
                float(rng.uniform(-self.h * 0.08, self.h * 0.08)),
                spr0,
            )

        spr0 = pool[int(rng.integers(0, len(pool)))]
        scale = float(rng.uniform(0.85, 1.62))
        inst = float(rng.uniform(0.62, 1.0))
        sw = max(24, int(spr0.get_width() * scale))
        sh = max(24, int(spr0.get_height() * scale))
        spr = pygame.transform.smoothscale(spr0, (sw, sh))
        db = float(np.clip(density_boost, 0.55, 1.45))
        combined = float(np.clip(layer.deck_alpha * inst * db, 0.20, 0.99))
        spr.set_alpha(int(np.clip(255 * combined, 28, 255)))

        x = float(rng.uniform(rect.left - rect.width * 0.38, rect.right + rect.width * 0.08))
        alt_m = float(
            np.clip(center_m - 0.64 * span_m + rng.uniform(-0.12, 0.12) * span_m, 0.0, 120_000.0)
        )
        yoff = float(rng.uniform(-self.h * 0.18, self.h * 0.12))
        if rng.random() < 0.32:
            x += float(rng.normal(0.0, rect.width * 0.04))
            alt_m = float(np.clip(alt_m + rng.normal(0.0, span_m * 0.022), 0.0, 120_000.0))
        return (x, alt_m, yoff, spr)

    def _draw_wind_clouds(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel, frame_dt: float) -> None:
        # Stacked haze decks: same wind; draw back (faint) → front (denser).
        dt = float(frame_dt)
        center_m = float(self.camera.center_y_m)
        if center_m > 95_000.0:
            return
        fall_speed = abs(float(model.vertical_speed_mps))
        speed_mul = float(np.clip(0.55 + 0.9 * (fall_speed / 250.0), 0.55, 1.75))
        rho = float(model.atm_density_kg_m3)
        rho0 = float(_TITAN_RHO_SURFACE_REF_KG_M3)
        density_f = float(np.clip((rho / max(rho0, 1e-9)) ** 0.38, 0.0, 1.35))
        # Thicker, more visible haze as atmospheric ρ increases.
        density_boost = float(np.clip(0.62 + 0.52 * min(density_f, 1.35), 0.62, 1.38))

        px_per_m = self.camera.scale_px_per_m
        span_m = rect.height / max(1e-9, px_per_m)
        v_probe_x = float(model.speed_x_mps)
        alpha_w = float(1.0 - math.exp(-max(dt, 0.0) / max(_CLOUD_WIND_SMOOTH_TAU_S, 1e-3)))

        h_probe = float(max(0.0, model.altitude_m))
        wx_t, wz_t = model.wind_vec_mps_at(h_probe)
        sm0, sm1 = self._cloud_wind_smooth
        nwx = sm0 + alpha_w * (float(wx_t) - sm0)
        nwz = sm1 + alpha_w * (float(wz_t) - sm1)
        self._cloud_wind_smooth = (nwx, nwz)
        v_rel_x = nwx - v_probe_x

        base_desired = int(
            np.clip(
                round(
                    2.4
                    + 1.35 * density_f
                    + 0.35 * (1.0 - min(center_m, 80_000.0) / 80_000.0)
                    + 0.45 * min(density_boost, 1.25)
                ),
                3,
                9,
            )
        )
        rng = self._rng

        decks = sorted(self._cloud_layers, key=lambda L: L.deck_alpha)
        for layer in decks:

            def spawn_blob() -> tuple[float, float, float, float, float, float, float, pygame.Surface]:
                x, alt_m, yoff, spr = self._spawn_haze_blob(
                    layer, rect, center_m, span_m, rng, density_boost
                )
                wob_phase = float(rng.uniform(0.0, 2 * math.pi))
                return (x, alt_m, yoff, 1.0, 1.0, 1.0, wob_phase, spr)

            desired = max(3, min(9, base_desired))
            while len(layer.blobs) < desired:
                layer.blobs.append(spawn_blob())

            new_blobs: list[tuple[float, float, float, float, float, float, float, pygame.Surface]] = []
            for x, alt_m, yoff, _sj, a_mul, scale, wob_phase, spr in layer.blobs:
                base_dx = v_rel_x * speed_mul * px_per_m * dt * float(_CLOUD_ADVECT_SCALE)
                x = x + base_dx

                _sx, sy = self._world_to_screen(rect, model.pos_x_m, float(alt_m))
                wob = 5.5 * math.sin(self._t * 0.26 + wob_phase + 0.0025 * x)
                yy = int(sy + yoff + wob)

                if yy < rect.top - 320:
                    continue
                if yy > rect.bottom + 340:
                    continue

                if int(x) < rect.right and int(x) + spr.get_width() > rect.left:
                    screen.blit(spr, (int(x), yy))
                new_blobs.append((x, alt_m, yoff, 1.0, a_mul, scale, wob_phase, spr))

            while len(new_blobs) < desired:
                new_blobs.append(spawn_blob())
            layer.blobs = new_blobs

    def _draw_wind_pennant(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        # Pennant wind: a ribbon attached to the probe, trailing with the *relative airflow* vector.
        # Relative airflow from model properties.
        v_air_x = float(model.air_rel_speed_x_mps)
        v_air_y = float(model.air_rel_speed_vert_mps)

        # Visibility scales with density and speed.
        speed = float(math.hypot(v_air_x, v_air_y))
        vis = float(np.clip((model.atm_density_kg_m3 / 1.0) * (speed / 30.0), 0.0, 1.0))
        if vis < 0.05:
            return

        x, y = self._probe_screen_xy(rect, model)

        # Direction: VISUAL SHLEIF trails opposite to the relative airflow (downstream wake).
        # If v_air points where air moves relative to probe, wake goes opposite.
        # world->screen: +y(world) is up, but +y(screen) is down, so invert y component.
        eps = 1e-6
        dx = -v_air_x
        dy = -(-v_air_y)
        mag = float(math.hypot(dx, dy))
        if mag < eps:
            return
        ux, uy = dx / mag, dy / mag
        # Perpendicular for flutter.
        px, py = -uy, ux

        # Teardrop ("comet") wake: rounded head around the body + tapered tail behind.
        scale = self._probe_scale(model)
        body_r = int(16 * scale)

        # Head is wider than the body and visible around it.
        head_r = float(body_r) * (1.25 + 0.55 * vis)
        tail_len = float(body_r) * (3.5 + 7.0 * np.clip(speed / 90.0, 0.0, 1.0))
        tail_len = float(np.clip(tail_len, body_r * 3.0, body_r * 12.0))

        base_alpha = int(np.clip(30 + 140 * vis, 0, 170))
        col = (235, 240, 245, base_alpha)

        # Tail polygon, width decays with distance.
        steps = 18
        phase = self._t * (2.4 + 1.8 * np.clip(speed / 90.0, 0.0, 1.0))
        left: list[tuple[int, int]] = []
        right: list[tuple[int, int]] = []
        for i in range(steps):
            u = i / (steps - 1)
            flutter = math.sin(phase + u * (4.5 + 1.4 * vis)) + 0.35 * math.sin(phase * 0.65 + u * 9.0)
            off = flutter * (2.0 + 6.0 * vis) * (1 - u) ** 0.7

            cx = x + int(ux * (u * tail_len) + px * off)
            cy = y + int(uy * (u * tail_len) + py * off)
            w = head_r * (1 - u) ** 0.85
            lx = cx + int(px * w)
            ly = cy + int(py * w)
            rx = cx - int(px * w)
            ry = cy - int(py * w)
            left.append((lx, ly))
            right.append((rx, ry))

        # Local bbox around pennant — avoid allocating a Surface for the full rect (costly at 1440p+).
        halo_r = int(head_r * 1.45) + 6
        xs = [x - halo_r, x + halo_r, x + int(ux * tail_len) + halo_r, x + int(ux * tail_len) - halo_r]
        ys = [y - halo_r, y + halo_r, y + int(uy * tail_len) + halo_r, y + int(uy * tail_len) - halo_r]
        for p in left + right:
            xs.append(p[0])
            ys.append(p[1])
        min_x = int(min(xs)) - 8
        max_x = int(max(xs)) + 8
        min_y = int(min(ys)) - 8
        max_y = int(max(ys)) + 8
        bw = max(1, max_x - min_x)
        bh = max(1, max_y - min_y)
        # Fully off-screen — skip draw.
        if max_x < rect.left or min_x > rect.right or max_y < rect.top or min_y > rect.bottom:
            return

        need_p = (bw, bh)
        if self._pennant_surf is None or self._pennant_surf_size != need_p:
            self._pennant_surf = pygame.Surface(need_p, pygame.SRCALPHA)
            self._pennant_surf_size = need_p
        tmp = self._pennant_surf
        tmp.fill((0, 0, 0, 0))
        ox, oy = min_x, min_y

        def loc(p: tuple[int, int]) -> tuple[int, int]:
            return (p[0] - ox, p[1] - oy)

        pygame.draw.circle(tmp, col, loc((x, y)), int(head_r))
        pygame.draw.circle(tmp, (235, 240, 245, int(base_alpha * 0.55)), loc((x, y)), int(head_r * 1.45))

        poly = [loc(p) for p in left + right[::-1]]
        if len(poly) >= 6:
            pygame.draw.polygon(tmp, (235, 240, 245, int(base_alpha * 0.75)), poly)
            left_l = [loc(p) for p in left]
            right_l = [loc(p) for p in right]
            pygame.draw.lines(tmp, (235, 240, 245, int(base_alpha * 0.25)), False, left_l, 5)
            pygame.draw.lines(tmp, (235, 240, 245, int(base_alpha * 0.25)), False, right_l, 5)

        screen.blit(tmp, (ox, oy))

    def _draw_minimap(self, screen: pygame.Surface, model: PhysicsModel, rect: pygame.Rect) -> None:
        # Infinite procedural minimap: probe stays centered, terrain scrolls under it.
        # Requirements:
        # - scale varies with altitude (closer to ground = zoom in)
        # - cache the rendered minimap and redraw only when needed
        # - target marker must not draw outside the minimap rectangle

        h = float(model.altitude_m)
        # meters-per-pixel: high altitude => show more area; near ground => zoom in.
        mpp = float(np.clip(8.0 + 0.004 * h, 8.0, 420.0))
        mpp = float(round(mpp / 2.0) * 2.0) if mpp < 20.0 else float(round(mpp / 5.0) * 5.0)

        # Visual LOD:
        # - at high altitude (large mpp) small lakes should fade out
        # - reduce speckle by sampling a lower-frequency field
        def _smooth01(t: float) -> float:
            t = float(np.clip(t, 0.0, 1.0))
            return t * t * (3.0 - 2.0 * t)

        lake_vis = 1.0 - _smooth01((mpp - 140.0) / (320.0 - 140.0))  # 1 near ground, 0 when zoomed out
        # Coarser world sampling for minimap at large mpp (stabilizes the image).
        sample_cell_m = 900.0 + 8.0 * max(0.0, mpp - 60.0)  # grows with mpp

        x0_m = float(model.pos_x_m)
        z0_m = float(model.pos_z_m)

        # Create or resize cache surface when needed.
        if (
            self._minimap_cache is None
            or self._minimap_cache_rect is None
            or self._minimap_cache_rect.size != rect.size
        ):
            # Render to a small fixed buffer and upscale
            # (much cheaper than near-pixel-perfect fill at full UI size).
            buf_w = int(np.clip(rect.width // 2, 96, 160))
            buf_h = int(np.clip(rect.height // 2, 96, 160))
            self._minimap_cache = pygame.Surface((buf_w, buf_h))
            self._minimap_cache_rect = pygame.Rect(0, 0, rect.width, rect.height)
            self._minimap_cache_center = (float("nan"), float("nan"))
            self._minimap_cache_mpp = 0.0
            self._minimap_cache_scaled = None
            self._minimap_cache_scaled_size = (0, 0)

        # Decide whether terrain cache needs a redraw.
        cx_last, cz_last = self._minimap_cache_center
        moved_m = float(math.hypot(x0_m - cx_last, z0_m - cz_last)) if math.isfinite(cx_last) else float("inf")
        # Redraw when the center moved enough or mpp changed.
        redraw = (moved_m >= (mpp * 18.0)) or (abs(mpp - self._minimap_cache_mpp) >= 0.5)

        if redraw:
            assert self._minimap_cache is not None
            surf = self._minimap_cache
            # LOD: when zoomed out, sample coarser blocks.
            step_px = int(np.clip(int(mpp / 22.0) * 2 + 2, 3, 16))
            cxs = surf.get_width() // 2
            cys = surf.get_height() // 2
            # Fill blocks.
            land_r, land_g, land_b = 80.0, 55.0, 40.0
            lake_r, lake_g, lake_b = 34.0, 28.0, 22.0
            rh_scale = rect.height / max(1, surf.get_height())
            rw_scale = rect.width / max(1, surf.get_width())
            inv_sample = 1.0 / sample_cell_m
            blk = step_px + 1
            for y in range(0, surf.get_height(), step_px):
                dz_m = (cys - y) * mpp * rh_scale
                wz_base = z0_m + dz_m
                wz_s = math.floor(wz_base * inv_sample) * sample_cell_m
                for x in range(0, surf.get_width(), step_px):
                    dx_m = (x - cxs) * mpp * rw_scale
                    wx = x0_m + dx_m
                    wx_s = math.floor(wx * inv_sample) * sample_cell_m
                    st = model.world.surface_type_at(wx_s, wz_s)
                    if st.value == "lake":
                        cr = land_r * (1.0 - lake_vis) + lake_r * lake_vis
                        cg = land_g * (1.0 - lake_vis) + lake_g * lake_vis
                        cb = land_b * (1.0 - lake_vis) + lake_b * lake_vis
                    else:
                        h_m = float(model.world.height_m_at(wx_s, wz_s))
                        t = (h_m + 5200.0) / 10400.0
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        calm = 1.0 - 0.14 * (1.0 - lake_vis)
                        shade = 0.72 + 0.44 * t
                        cr = land_r * calm * shade
                        cg = land_g * calm * shade
                        cb = land_b * calm * shade
                    color = (
                        max(0, min(255, int(cr))),
                        max(0, min(255, int(cg))),
                        max(0, min(255, int(cb))),
                    )
                    pygame.draw.rect(surf, color, pygame.Rect(x, y, blk, blk))

            self._minimap_cache_center = (x0_m, z0_m)
            self._minimap_cache_mpp = mpp
            self._minimap_cache_scaled = None  # invalidate upscaled copy

        # Blit minimap cache (upscale small buffer to screen rect).
        assert self._minimap_cache is not None
        if self._minimap_cache_scaled is None or self._minimap_cache_scaled_size != rect.size:
            # scale is much cheaper than smoothscale; smoothing not critical for minimap.
            self._minimap_cache_scaled = pygame.transform.scale(self._minimap_cache, rect.size)
            self._minimap_cache_scaled_size = rect.size
        screen.blit(self._minimap_cache_scaled, rect.topleft)

        # Clip overlays to the minimap rectangle.
        prev_clip = screen.get_clip()
        screen.set_clip(rect)

        # Scale label ("m/px") in corner; cache font to avoid per-frame allocation.
        try:
            if self._minimap_font is None:
                self._minimap_font = pygame.font.Font(None, 18)
            label = self._minimap_font.render(f"{int(mpp)} m/px", True, (230, 230, 235))
            screen.blit(label, (rect.left + 6, rect.top + 4))
        except Exception:
            pass

        # Target marker relative to probe; at edge show arrow if off-screen (no line to center).
        dx = float(model.target_x_m) - x0_m
        dz = float(model.target_z_m) - z0_m
        tx = rect.centerx + int(dx / mpp)
        ty = rect.centery - int(dz / mpp)
        cx, cy = rect.centerx, rect.centery
        inner = rect.inflate(-10, -10)
        if inner.collidepoint((tx, ty)):
            pygame.draw.circle(screen, (255, 230, 130), (tx, ty), 7, 2)
            pygame.draw.circle(screen, (80, 70, 40), (tx, ty), 3)
        else:
            fx = tx - cx
            fy = ty - cy
            dist = float(math.hypot(fx, fy)) or 1.0
            ux = fx / dist
            uy = fy / dist
            t_hit = float("inf")
            if ux > 1e-6:
                t_hit = min(t_hit, (inner.right - cx) / ux)
            if ux < -1e-6:
                t_hit = min(t_hit, (inner.left - cx) / ux)
            if uy > 1e-6:
                t_hit = min(t_hit, (inner.bottom - cy) / uy)
            if uy < -1e-6:
                t_hit = min(t_hit, (inner.top - cy) / uy)
            if math.isfinite(t_hit) and t_hit > 0:
                ex = int(cx + ux * t_hit)
                ey = int(cy + uy * t_hit)
                ang = math.atan2(-uy, ux)
                s = 9
                p1 = (ex + int(math.cos(ang) * s), ey - int(math.sin(ang) * s))
                p2 = (ex + int(math.cos(ang + 2.4) * (s * 0.85)), ey - int(math.sin(ang + 2.4) * (s * 0.85)))
                p3 = (ex + int(math.cos(ang - 2.4) * (s * 0.85)), ey - int(math.sin(ang - 2.4) * (s * 0.85)))
                pygame.draw.polygon(screen, (255, 220, 100), (p1, p2, p3))

        # Probe marker always at center.
        pygame.draw.circle(screen, (240, 240, 245), rect.center, 4)

        screen.set_clip(prev_clip)

