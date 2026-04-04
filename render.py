from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pygame

from digital_twin.model import PhysicsModel, SimResult


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
    alt_m: float
    speed_scale: float
    offset_x_px: float
    # blob tuple:
    # (x_px, alt_m, yoff_px, speed_jit, alpha_mul, scale, wob_phase, sprite)
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
        self._cloud_speed_boost: float = 100.0
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
        # Reusable semi-transparent fog plane (no per-frame allocation).
        self._fog_surf: Optional[pygame.Surface] = None
        self._fog_surf_size: tuple[int, int] = (0, 0)
        self._fog_cache_key: Optional[tuple[int, int, int]] = None  # (w, h, alpha//3)
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
        self._tgrid_smooth_max_dh: float = 200.0
        # Water landing: splash rings + probe sinking.
        self._water_splash: Optional[dict] = None
        self._was_water_landed: bool = False
        self._sink_t: float = 0.0

    def migrate_from(self, prev: Renderer) -> None:
        pass

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

        self._draw_sky(screen, ui_rect, model)
        self._draw_wind_clouds(screen, ui_rect, model, frame_dt)
        self._draw_fog(screen, ui_rect, model)
        self._draw_ground(screen, ui_rect, model)
        self._draw_probe(screen, ui_rect, model)
        self._draw_event_frags(screen, ui_rect, model)
        self._draw_effects(screen, ui_rect, model, frame_dt)
        if controller is not None:
            ui.draw_overlay(screen, controller)  # type: ignore[arg-type]
        self.draw_minimap(screen, model, ui.map_rect)
        if controller is not None:
            ui.draw_mission_report_modal(screen, controller)  # type: ignore[arg-type]
            ui.draw_pause_control(screen)  # type: ignore[arg-type]
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

        haze = float(np.clip((rho / 5.9) ** 0.35, 0.0, 1.0))
        milky = np.array([245, 205, 170], dtype=float)
        top = lerp(top, milky, 0.20 * haze)
        bottom = lerp(bottom, milky * 0.75, 0.28 * haze)
        return top, bottom

    def _draw_sky(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        h = max(0.0, float(model.altitude_m))
        rho = float(model.atm_density_kg_m3)
        # Coarser quantization: on fast descent int(rho*500) would change almost every frame —
        # full sky rebuild + scale would tank FPS and cause long stalls from alloc/GC.
        qh = int(h // 900.0)
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

    def _draw_fog(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel) -> None:
        h = float(model.altitude_m)
        rho = float(model.atm_density_kg_m3)
        base = np.clip(160 - (h / 2000.0), 0, 160)
        dens = np.clip((rho / 1.5) * 80.0, 0, 80)
        a = int(np.clip(base + dens, 0, 180))
        if a <= 0:
            return
        key = (rect.width, rect.height, a // 3)
        if self._fog_surf is None or self._fog_cache_key != key:
            if self._fog_surf is None or self._fog_surf_size != (rect.width, rect.height):
                self._fog_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                self._fog_surf_size = (rect.width, rect.height)
            self._fog_surf.fill((200, 140, 90, a))
            self._fog_cache_key = key
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
        self._tgrid_smooth_max_dh += (raw_max_dh - self._tgrid_smooth_max_dh) * 0.06
        max_dh = max(80.0, self._tgrid_smooth_max_dh)

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
                    shimmer = 0.06 * math.sin(self._t * 1.8 + wx * 0.0003 + wz * 0.0004)
                    br = 45.0 + shimmer * 20
                    bg = 105.0 + shimmer * 60
                    bb = 185.0 + shimmer * 30
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
        hzc = (125, 95, 78, max(0, min(255, int(a255 * 0.80))))
        for k in range(len(far_pts) - 1):
            if _bt - 4 <= far_pts[k][1] <= _rb + 4:
                pygame.draw.line(layer, hzc, far_pts[k], far_pts[k + 1], 2)

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
                            rs, (180, 210, 235, ra),
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
        # Approximate "charring" based on dynamic pressure proxy ~ rho * v^2.
        v = abs(float(model.vertical_speed_mps))
        q = float(model.atm_density_kg_m3) * (v * v)  # proxy
        # Map to 0..1 over a reasonable range.
        t = float(np.clip((q - 50_000.0) / 450_000.0, 0.0, 1.0))
        # Slightly nonlinear so blackening happens mostly during stronger heating.
        t = t * t
        base = np.array([220, 205, 190], dtype=float)
        char = np.array([35, 30, 28], dtype=float)
        c = (base * (1 - t) + char * t).astype(int)
        return int(c[0]), int(c[1]), int(c[2])

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
                                            (140, 190, 220)))

        body_r = int(16 * scale)
        inner_color = (245, 245, 248)

        if sinking:
            sz = body_r * 3 + 4
            tmp = pygame.Surface((sz, sz), pygame.SRCALPHA)
            cx_t, cy_t = sz // 2, sz // 2
            if not model.heatshield_jettisoned:
                self._heatshield_rgb = self._heatshield_color(model)
                shell_r = int(body_r * 1.28)
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
        landed = model.result != SimResult.RUNNING

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
                        color=(100, 160, 210),
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
        # Three semi-transparent cloud plans (parallax).
        layers: list[CloudLayer] = []
        for alt_m, speed_scale, seed in [
            (6_000.0, 1.00, 1),   # near plan (fast)
            (22_000.0, 0.62, 2),  # mid plan
            (65_000.0, 0.36, 3),  # far plan (slow)
        ]:
            blobs = self._make_cloud_blobs(seed=seed)
            layers.append(CloudLayer(alt_m=float(alt_m), speed_scale=float(speed_scale), offset_x_px=0.0, blobs=blobs))
        return layers

    def _make_cloud_sprite(self, rng: np.random.Generator) -> pygame.Surface:
        # One "cloud": smooth, homogeneous blob (rocket-plume-like), not noisy/curly inside.
        # 2x smaller clouds (performance + visibility)
        w = int(rng.uniform(150, 260))
        h = int(rng.uniform(90, 170))
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        base_color = (235, 220, 210)

        def puff_oval(cx: int, cy: int, rx: int, ry: int, a: int) -> None:
            for k in range(4):
                rrx = max(2, int(rx * (1.0 - 0.18 * k)))
                rry = max(2, int(ry * (1.0 - 0.18 * k)))
                aa = max(0, int(a * (1.0 - 0.18 * k)))
                pygame.draw.ellipse(surf, (*base_color, aa), pygame.Rect(cx - rrx, cy - rry, 2 * rrx, 2 * rry))

        alpha = int(rng.uniform(36, 62))
        cx0, cy0 = w // 2, h // 2
        # A few large overlapping ovals with similar alpha => homogeneous interior.
        for _ in range(int(rng.integers(4, 7))):
            cx = int(cx0 + rng.normal(0, w * 0.08))
            cy = int(cy0 + rng.normal(0, h * 0.08))
            rx = int(rng.uniform(w * 0.18, w * 0.30))
            ry = int(rng.uniform(h * 0.18, h * 0.30))
            puff_oval(cx, cy, rx, ry, alpha)
        return surf

    def _make_cloud_blobs(self, seed: int) -> list[tuple[float, float, float, float, float, float, float, pygame.Surface]]:
        rng = np.random.default_rng(seed)
        blobs: list[tuple[float, float, float, float, float, float, float, pygame.Surface]] = []
        # Seed a tiny pool; we will spawn/despawn around the camera dynamically.
        alt_max = 120_000.0
        count = 2
        for _ in range(count):
            spr = self._make_cloud_sprite(rng)
            # Spawn mostly across (and slightly beyond) the screen width for continuous wrap.
            # Wind is positive in our profile, so bias spawn to the left so clouds cross the frame.
            x = float(rng.uniform(-self.w * 0.4, self.w * 0.1))
            u = float(rng.uniform(0.0, 1.0))
            alt_m = float((u**1.6) * alt_max)
            # Keep vertical offsets tighter so more clouds stay in view.
            yoff = float(rng.uniform(-self.h * 0.14, self.h * 0.14))
            speed_jit = float(rng.uniform(0.75, 1.30))
            a_mul = float(rng.uniform(0.75, 0.95))
            scale = float(rng.uniform(1.10, 1.55))
            wob_phase = float(rng.uniform(0.0, 2 * math.pi))
            # Pre-scale once to avoid per-frame smoothscale cost.
            sw = max(8, int(spr.get_width() * scale))
            sh = max(8, int(spr.get_height() * scale))
            spr2 = pygame.transform.smoothscale(spr, (sw, sh))
            spr2.set_alpha(int(np.clip(255 * a_mul, 30, 255)))
            blobs.append((x, alt_m, yoff, speed_jit, 1.0, 1.0, wob_phase, spr2))
        return blobs

    def _draw_wind_clouds(self, screen: pygame.Surface, rect: pygame.Rect, model: PhysicsModel, frame_dt: float) -> None:
        # Clouds drift with wind profile and parallax.
        # Performance constraints:
        # - keep a small active set near the camera
        # - spawn below (behind camera) shortly before entering the frame
        # - cull after passing the camera (exit upward) to avoid drawing far away
        dt = float(frame_dt)
        center_m = float(self.camera.center_y_m)
        # At very high altitudes, clouds should not be visible (and we skip all cloud work).
        if center_m > 95_000.0:
            return
        fall_speed = abs(float(model.vertical_speed_mps))
        # Multiplier for all plans: faster descent => clouds move faster (sense of speed-up).
        speed_mul = float(np.clip(0.55 + 0.9 * (fall_speed / 250.0), 0.55, 1.75))
        for layer in self._cloud_layers:
            px_per_m = self.camera.scale_px_per_m
            span_m = rect.height / max(1e-9, px_per_m)

            # Keep a small active set per layer (spawn/despawn).
            desired = 2
            rng = self._rng

            def spawn_blob() -> tuple[float, float, float, float, float, float, float, pygame.Surface]:
                # Spawn below the camera view so it "floats in" soon.
                # Spawn across (and slightly beyond) the visible width so clouds actually appear.
                x = float(rng.uniform(rect.left - rect.width * 0.20, rect.right - rect.width * 0.05))
                alt_m = float(np.clip(center_m - 0.65 * span_m + rng.uniform(-0.10, 0.10) * span_m, 0.0, 120_000.0))
                yoff = float(rng.uniform(-self.h * 0.12, self.h * 0.12))
                s = float(rng.uniform(0.85, 1.20))
                wob_phase = float(rng.uniform(0.0, 2 * math.pi))
                # Reuse an existing sprite from this layer (already pre-scaled + alpha set).
                # Fallback: never create raw unscaled sprites here (too expensive / wrong alpha).
                spr = layer.blobs[0][7]
                return (x, alt_m, yoff, s, 1.0, 1.0, wob_phase, spr)

            # Ensure at least desired blobs exist.
            while len(layer.blobs) < desired:
                layer.blobs.append(spawn_blob())

            new_blobs = []
            for x, alt_m, yoff, s, a_mul, scale, wob_phase, spr in layer.blobs:
                wind = float(model.wind_vec_mps_at(float(alt_m))[0])
                base_dx = float(wind) * float(layer.speed_scale) * speed_mul * px_per_m * dt * self._cloud_speed_boost
                x = x + base_dx * s

                # render position
                _sx, sy = self._world_to_screen(rect, model.pos_x_m, float(alt_m))
                wob = 10.0 * math.sin(self._t * (0.30 + 0.18 * layer.speed_scale) + wob_phase + 0.004 * x)
                yy = int(sy + yoff + wob)

                # Despawn clouds after they pass the camera view (above), and don't keep off-screen far below.
                if yy < rect.top - 220:
                    continue
                if yy > rect.bottom + 260:
                    continue

                # Only blit if in view horizontally (cheap cull).
                if int(x) < rect.right and int(x) + spr.get_width() > rect.left:
                    screen.blit(spr, (int(x), yy))
                new_blobs.append((x, alt_m, yoff, s, a_mul, scale, wob_phase, spr))

            # Top up after despawn.
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
            lake_r, lake_g, lake_b = 40.0, 80.0, 115.0
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

