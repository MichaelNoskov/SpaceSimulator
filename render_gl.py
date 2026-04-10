"""
OpenGL compositing + perspective terrain for landing view (PyOpenGL + pygame).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import pygame

if TYPE_CHECKING:
    from digital_twin.model import PhysicsModel
    from render import Renderer

_GL_INIT = False
_tex_bg: int = 0
_tex_ov: int = 0
_tex_w = 0
_tex_h = 0


def _ensure_gl() -> None:
    global _GL_INIT
    if _GL_INIT:
        return
    from OpenGL.GL import (
        GL_BLEND,
        GL_DEPTH_TEST,
        GL_LEQUAL,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_SRC_ALPHA,
        glBlendFunc,
        glClearColor,
        glClearDepth,
        glDepthFunc,
        glEnable,
    )

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glClearDepth(1.0)
    glClearColor(0.05, 0.03, 0.06, 1.0)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    _GL_INIT = True


def _ensure_textures(w: int, h: int) -> None:
    global _tex_bg, _tex_ov, _tex_w, _tex_h
    from OpenGL.GL import (
        GL_LINEAR,
        GL_RGBA,
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        GL_TEXTURE_MIN_FILTER,
        GL_UNSIGNED_BYTE,
        glBindTexture,
        glDeleteTextures,
        glGenTextures,
        glTexImage2D,
        glTexParameteri,
    )

    if w == _tex_w and h == _tex_h and _tex_bg and _tex_ov:
        return
    if _tex_bg:
        glDeleteTextures([_tex_bg])
    if _tex_ov:
        glDeleteTextures([_tex_ov])
    _tex_bg = int(glGenTextures(1))
    _tex_ov = int(glGenTextures(1))
    _tex_w, _tex_h = w, h
    for tid in (_tex_bg, _tex_ov):
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)


def _upload_surface(tex_id: int, surf: pygame.Surface) -> None:
    from OpenGL.GL import GL_RGBA, GL_TEXTURE_2D, GL_UNSIGNED_BYTE, glBindTexture, glTexSubImage2D

    if surf.get_bitsize() != 32:
        surf = surf.convert_alpha()
    w, h = surf.get_size()
    flipped = pygame.transform.flip(surf, False, True)
    raw = pygame.image.tobytes(flipped, "RGBA", True)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, raw)


def _draw_fullscreen_textured_quad(w: int, h: int, tex_id: int, alpha: float = 1.0) -> None:
    from OpenGL.GL import (
        GL_DEPTH_TEST,
        GL_MODULATE,
        GL_QUADS,
        GL_TEXTURE_2D,
        GL_TEXTURE_ENV,
        GL_TEXTURE_ENV_MODE,
        GL_MODELVIEW,
        GL_PROJECTION,
        glBegin,
        glBindTexture,
        glColor4f,
        glDisable,
        glEnable,
        glEnd,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,
        glTexCoord2f,
        glTexEnvi,
        glVertex2f,
    )

    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0.0, float(w), float(h), 0.0, -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
    glColor4f(1.0, 1.0, 1.0, alpha)
    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(float(w), 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(float(w), float(h))
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0.0, float(h))
    glEnd()
    glDisable(GL_TEXTURE_2D)


def _vertex_color(
    grid_h: list[list[float]],
    grid_st: list[list[str]],
    i: int,
    j: int,
    nx: int,
    nz: int,
    gx0: float,
    gz0: float,
    step_x: float,
    step_z: float,
    hmin: float,
    hrange: float,
    lx: float,
    lz: float,
    ly: float,
    t_anim: float,
    for_gl_lighting: bool = False,
    normal_override: Optional[tuple[float, float, float]] = None,
) -> tuple[float, float, float]:
    hc = grid_h[i][j]
    st = grid_st[i][j]
    if normal_override is not None:
        nxn, nyn, nzn = normal_override
    else:
        nxn, nyn, nzn = _terrain_normal_at(grid_h, i, j, nx, nz, step_x, step_z)
    dot = nxn * lx + nzn * lz + nyn * ly
    shade = max(0.10, min(1.60, dot * 1.50 + 0.20))
    slope = math.sqrt(nxn * nxn + nzn * nzn)
    slope_dark = max(0.50, 1.0 - slope * 0.55)
    lit = 0.40 + 0.60 * shade * slope_dark
    if for_gl_lighting:
        lit = 1.0
        slope_dark = 1.0
    zt = j / max(1.0, nz - 2.0)
    fog_vis = 0.52 + 0.48 * zt
    fog_haze = (1.0 - zt) * 0.30
    wx = gx0 + i * step_x
    wz = gz0 + j * step_z
    haze = (118.0, 92.0, 72.0)
    if st == "lake":
        # Match render.py: Titan lake = liquid hydrocarbon, not blue water.
        shimmer = 0.06 * math.sin(t_anim * 1.8 + wx * 0.0085 + wz * 0.012)
        br = 48.0 + shimmer * 18
        bg = 40.0 + shimmer * 15
        bb = 28.0 + shimmer * 8
        spec = max(0.0, math.sin(t_anim * 0.7 + wx * 0.00015)) ** 8 * 0.25
        if for_gl_lighting:
            lit2 = 0.62 + 0.38 * spec
        else:
            lit2 = 0.60 + 0.40 * lit + spec
    else:
        ht = max(0.0, min(1.0, (hc - hmin) / hrange))
        br = 120.0 + 85.0 * ht
        bg = 78.0 + 62.0 * ht
        bb = 48.0 + 42.0 * ht
        lit2 = lit
    tex = 0.92 + 0.16 * math.sin(wx * 0.0085) * math.cos(wz * 0.012)
    if st != "lake":
        tex *= 1.0 + 0.055 * (
            math.sin(wx * 0.031 + wz * 0.011)
            + math.sin(wz * 0.028 - wx * 0.009)
            + 0.5 * math.sin((wx + wz) * 0.045)
        )
    sf = lit2 * fog_vis * tex
    r = max(0.0, min(1.0, (br * sf + haze[0] * fog_haze) / 255.0))
    g = max(0.0, min(1.0, (bg * sf + haze[1] * fog_haze) / 255.0))
    b = max(0.0, min(1.0, (bb * sf + haze[2] * fog_haze) / 255.0))
    return r, g, b


def _terrain_normal_at(
    grid_h: list[list[float]],
    i: int,
    j: int,
    nx: int,
    nz: int,
    step_x: float,
    step_z: float,
) -> tuple[float, float, float]:
    """Terrain normal from central height differences (smoother than one-sided)."""
    im = max(0, i - 1)
    ip = min(nx - 1, i + 1)
    jm = max(0, j - 1)
    jp = min(nz - 1, j + 1)
    denom_x = max(1e-9, float(ip - im) * step_x)
    denom_z = max(1e-9, float(jp - jm) * step_z)
    dhx = (grid_h[ip][j] - grid_h[im][j]) / denom_x
    dhz = (grid_h[i][jp] - grid_h[i][jm]) / denom_z
    exag = 4.5
    nxn = -dhx * exag
    nzn = -dhz * exag
    nyn = 1.0
    nm = math.sqrt(nxn * nxn + nzn * nzn + nyn * nyn)
    if nm > 1e-9:
        nxn /= nm
        nzn /= nm
        nyn /= nm
    return nxn, nyn, nzn


def _terrain_smoothed_normal_grid(
    grid_h: list[list[float]],
    nx: int,
    nz: int,
    step_x: float,
    step_z: float,
) -> list[list[tuple[float, float, float]]]:
    """
    Per-vertex averaged normals (3×3 box filter on analytic normals).
    Reduces visible quad edges / jagged horizon under smooth shading.
    """
    raw: list[list[tuple[float, float, float]]] = []
    for i in range(nx):
        row: list[tuple[float, float, float]] = []
        for j in range(nz):
            row.append(_terrain_normal_at(grid_h, i, j, nx, nz, step_x, step_z))
        raw.append(row)
    out: list[list[tuple[float, float, float]]] = []
    for i in range(nx):
        row: list[tuple[float, float, float]] = []
        for j in range(nz):
            sx = sy = sz = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < nx and 0 <= jj < nz:
                        nxn, nyn, nzn = raw[ii][jj]
                        sx += nxn
                        sy += nyn
                        sz += nzn
            nm = math.sqrt(sx * sx + sy * sy + sz * sz)
            if nm > 1e-9:
                sx, sy, sz = sx / nm, sy / nm, sz / nm
            row.append((sx, sy, sz))
        out.append(row)
    return out


def _glu_lookat_up_hint(fx: float, fy: float, fz: float) -> tuple[float, float, float]:
    """
    Up vector for gluLookAt: like GLU, side = forward × up — up must not be parallel to forward.
    Do not flip by |fy|>0.82: on landing fy crosses the threshold and causes a mirrored-frame jump.
    """
    for ux, uy, uz in ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
        sx = fy * uz - fz * uy
        sy = fz * ux - fx * uz
        sz = fx * uy - fy * ux
        if sx * sx + sy * sy + sz * sz > 1e-10:
            return ux, uy, uz
    return 0.0, 1.0, 0.0


def draw_background_terrain(
    bg_surf: pygame.Surface,
    model: PhysicsModel,
    renderer: Renderer,
    w: int,
    h: int,
    terrain_alpha: float,
    t_anim: float,
) -> Tuple[Optional[Tuple[int, int]], np.ndarray, np.ndarray, list]:
    """
    Clears GL buffer, draws bg texture, draws 3D terrain, returns (probe_screen_xy or None),
    modelview and projection matrices (4x4 column-major) for gluProject, and viewport [x,y,w,h].
    """
    from OpenGL.GL import (
        GL_AMBIENT,
        GL_AMBIENT_AND_DIFFUSE,
        GL_COLOR_BUFFER_BIT,
        GL_COLOR_MATERIAL,
        GL_CONSTANT_ATTENUATION,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_DIFFUSE,
        GL_FOG,
        GL_FOG_COLOR,
        GL_FOG_END,
        GL_FOG_MODE,
        GL_FOG_START,
        GL_FRONT_AND_BACK,
        GL_LIGHT0,
        GL_LIGHT1,
        GL_LIGHTING,
        GL_LIGHT_MODEL_AMBIENT,
        GL_LINEAR,
        GL_LINEAR_ATTENUATION,
        GL_MODELVIEW,
        GL_MODELVIEW_MATRIX,
        GL_POLYGON_OFFSET_FILL,
        GL_POSITION,
        GL_PROJECTION,
        GL_PROJECTION_MATRIX,
        GL_QUADRATIC_ATTENUATION,
        GL_SHININESS,
        GL_SMOOTH,
        GL_SPECULAR,
        GL_TRIANGLES,
        glBegin,
        glClear,
        glColor3f,
        glColorMaterial,
        glDepthMask,
        glDisable,
        glEnable,
        glEnd,
        glFogf,
        glFogfv,
        glFogi,
        glGetDoublev,
        glLightf,
        glLightfv,
        glLightModelfv,
        glLoadIdentity,
        glMaterialf,
        glMaterialfv,
        glMatrixMode,
        glNormal3f,
        glPolygonOffset,
        glShadeModel,
        glVertex3f,
        glViewport,
    )
    from OpenGL.GLU import gluLookAt, gluPerspective, gluProject

    _ensure_gl()
    _ensure_textures(w, h)
    _upload_surface(_tex_bg, bg_surf)

    glViewport(0, 0, w, h)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    _draw_fullscreen_textured_quad(w, h, _tex_bg, 1.0)

    probe_xy: Optional[Tuple[int, int]] = None
    mv = np.zeros(16, dtype=np.float64)
    pr = np.zeros(16, dtype=np.float64)
    viewport = [0, 0, w, h]

    if terrain_alpha < 0.008:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        mv[:] = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64).ravel()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        pr[:] = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64).ravel()
        return probe_xy, mv, pr, viewport

    alt = float(model.altitude_m)
    probe_x = float(model.pos_x_m)
    probe_z = float(model.pos_z_m)
    h_terr = float(model.world.height_m_at(probe_x, probe_z))
    py = h_terr + float(model.altitude_m)
    # Denser terrain mesh and clearer fog when close to the surface or 3D view is strong.
    landing_approach = alt < 4200.0 or float(terrain_alpha) > 0.38

    # Orbit camera
    yaw = float(renderer.orbit_yaw)
    pitch_raw = float(renderer.orbit_pitch)
    dist = float(np.clip(90.0 + 0.45 * max(0.0, alt), 120.0, 920.0))
    # If eye is below target, forward ≈ up → gluLookAt is ill-conditioned and the view flips horizontally.
    min_eye_above_m = 12.0
    sin_pitch_min = float(np.clip(min_eye_above_m / max(dist, 1e-6), -0.999, 0.999))
    pitch_floor = math.asin(sin_pitch_min)
    pitch = min(max(pitch_raw, pitch_floor), 1.38)
    cp = math.cos(pitch)
    ox = dist * cp * math.sin(yaw)
    oy = dist * math.sin(pitch)
    oz = dist * cp * math.cos(yaw)
    eye_x = probe_x + ox
    eye_y = py + oy
    # Game Z (+Z forward on map, isometric-style) → use -Z in GL or terrain mirrors.
    probe_z_gl = -probe_z
    eye_z_gl = -probe_z - oz

    far_plane = float(min(260_000.0, max(42_000.0, alt * 28.0 + 38_000.0)))
    near_plane = max(2.5, min(8.0, 3.0 + alt * 0.002))

    glEnable(GL_DEPTH_TEST)
    glDepthMask(True)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(64.0 if landing_approach else 68.0, float(w) / max(1, h), near_plane, far_plane)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # View direction (GLU-style: forward = normalize(center − eye)).
    fx = probe_x - eye_x
    fy = py - eye_y
    fz = probe_z_gl - eye_z_gl
    fn = math.sqrt(fx * fx + fy * fy + fz * fz)
    if fn < 1e-9:
        fn = 1.0
    else:
        fx, fy, fz = fx / fn, fy / fn, fz / fn
    upx, upy, upz = _glu_lookat_up_hint(fx, fy, fz)
    gluLookAt(eye_x, eye_y, eye_z_gl, probe_x, py, probe_z_gl, upx, upy, upz)

    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.22, 0.20, 0.24, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.28, 0.26, 0.30, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.72, 0.68, 0.62, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.18, 0.17, 0.15, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.12, 0.11, 0.10, 1.0])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 12.0)

    mv_np = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64).reshape(4, 4, order="F")
    # Sun direction in the same axes as vertices (game XZ → GL with -Z).
    _sx0, _sz0, _sy0 = -0.55, 0.50, 0.65
    _slm = math.sqrt(_sx0 * _sx0 + _sz0 * _sz0 + _sy0 * _sy0)
    sun_gl = np.array([_sx0 / _slm, _sy0 / _slm, -_sz0 / _slm, 0.0], dtype=np.float64)
    sun_eye = mv_np @ sun_gl
    glLightfv(
        GL_LIGHT0,
        GL_POSITION,
        [float(sun_eye[0]), float(sun_eye[1]), float(sun_eye[2]), 0.0],
    )

    eng = float(model.throttle_0_1) if model.engine_on else 0.0
    if eng > 0.02:
        nozzle_y = py - max(1.5, min(8.0, 2.5 + alt * 0.015))
        flame_w = np.array([probe_x, nozzle_y, probe_z_gl, 1.0], dtype=np.float64)
        flame_e = mv_np @ flame_w
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
        glLightfv(
            GL_LIGHT1,
            GL_DIFFUSE,
            [min(2.4, 0.45 + eng * 2.8), min(1.5, 0.12 + eng * 1.6), min(0.55, eng * 0.45), 1.0],
        )
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.35, 0.2, 0.08, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [float(flame_e[0]), float(flame_e[1]), float(flame_e[2]), 1.0])
        glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 1.0)
        glLightf(GL_LIGHT1, GL_LINEAR_ATTENUATION, 0.0)
        q_at = float(np.clip(2.5e-5 + eng * 8e-5, 1.2e-5, 2.2e-4))
        glLightf(GL_LIGHT1, GL_QUADRATIC_ATTENUATION, q_at)
    else:
        glDisable(GL_LIGHT1)

    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, [0.14, 0.11, 0.13, 1.0])
    glFogi(GL_FOG_MODE, GL_LINEAR)
    if landing_approach:
        glFogf(GL_FOG_START, far_plane * 0.14)
        glFogf(GL_FOG_END, far_plane * 0.88)
    else:
        glFogf(GL_FOG_START, far_plane * 0.22)
        glFogf(GL_FOG_END, far_plane * 0.94)

    fade = float(np.clip(0.25 + 0.75 * terrain_alpha, 0.0, 1.0))

    # Extra nz vs nx: more slices along depth → smoother horizon silhouette.
    nx, nz = (92, 72) if landing_approach else (38, 28)
    sc = float(renderer.camera.scale_px_per_m)
    gs = renderer._gl_terrain_scale_smooth
    if gs is None:
        gs = sc
    dr = abs(sc - gs) / max(sc, 1e-12)
    t_sm = float(np.clip(0.06 + 0.55 * dr, 0.06, 0.42))
    gs = gs + (sc - gs) * t_sm
    renderer._gl_terrain_scale_smooth = gs
    world_w_m = w / max(1e-9, gs)
    raw_sx = world_w_m * 1.12 / max(1, nx - 1)
    raw_sz = world_w_m * 0.72 / max(1, nz - 1)
    if landing_approach:
        step_min = 5.0
        step_x = max(step_min, round(raw_sx / step_min) * step_min)
        step_z = max(step_min, round(raw_sz / step_min) * step_min)
    else:
        step_x = max(12.0, round(raw_sx / 10.0) * 10.0)
        step_z = max(12.0, round(raw_sz / 10.0) * 10.0)
    # No floor() grid snap — otherwise terrain pops when moving or step size changes.
    gx0 = probe_x - 0.5 * float(nx - 1) * step_x
    gz0 = probe_z - 0.18 * float(nz - 1) * step_z

    grid_h: list[list[float]] = []
    grid_st: list[list[str]] = []
    for ix in range(nx):
        col_h: list[float] = []
        col_s: list[str] = []
        wx = gx0 + ix * step_x
        for iz in range(nz):
            wz = gz0 + iz * step_z
            col_h.append(float(model.world.height_m_at(wx, wz)))
            col_s.append(model.world.surface_type_at(wx, wz).value)
        grid_h.append(col_h)
        grid_st.append(col_s)

    hmin = min(min(c) for c in grid_h)
    hmax = max(max(c) for c in grid_h)
    hrange = max(8.0, hmax - hmin)
    _lx0, _lz0, _ly0 = -0.55, 0.50, 0.65
    _lm = math.sqrt(_lx0 * _lx0 + _lz0 * _lz0 + _ly0 * _ly0)
    _lx, _lz, _ly = _lx0 / _lm, _lz0 / _lm, _ly0 / _lm

    norm_sm = _terrain_smoothed_normal_grid(grid_h, nx, nz, step_x, step_z)

    rgb_grid: list[list[tuple[float, float, float]]] = []
    for i in range(nx):
        row: list[tuple[float, float, float]] = []
        for j in range(nz):
            nxn, nyn, nzn = norm_sm[i][j]
            row.append(
                _vertex_color(
                    grid_h,
                    grid_st,
                    i,
                    j,
                    nx,
                    nz,
                    gx0,
                    gz0,
                    step_x,
                    step_z,
                    hmin,
                    hrange,
                    _lx,
                    _lz,
                    _ly,
                    t_anim,
                    for_gl_lighting=True,
                    normal_override=(nxn, nyn, nzn),
                )
            )
        rgb_grid.append(row)

    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1.0, 2.0)
    glBegin(GL_TRIANGLES)

    def _emit_corner(ii: int, jj: int) -> None:
        nxn, nyn, nzn = norm_sm[ii][jj]
        glNormal3f(nxn, nyn, -nzn)
        r, g, b = rgb_grid[ii][jj]
        glColor3f(r * fade, g * fade, b * fade)
        glVertex3f(gx0 + ii * step_x, grid_h[ii][jj], -(gz0 + jj * step_z))

    for iz in range(nz - 1):
        for ix in range(nx - 1):
            if landing_approach:
                xc = gx0 + (ix + 0.5) * step_x
                zc = gz0 + (iz + 0.5) * step_z
                hc = float(model.world.height_m_at(xc, zc))
                sx = sy = sz = 0.0
                for di in (0, 1):
                    for dj in (0, 1):
                        nxn, nyn, nzn = norm_sm[ix + di][iz + dj]
                        sx += nxn
                        sy += nyn
                        sz += nzn
                nm = math.sqrt(sx * sx + sy * sy + sz * sz)
                if nm > 1e-9:
                    ncx, ncy, ncz = sx / nm, sy / nm, sz / nm
                else:
                    ncx, ncy, ncz = 0.0, 1.0, 0.0
                rc = gc = bc = 0.0
                for di in (0, 1):
                    for dj in (0, 1):
                        r, g, b = rgb_grid[ix + di][iz + dj]
                        rc += r
                        gc += g
                        bc += b
                rc *= 0.25
                gc *= 0.25
                bc *= 0.25

                def _emit_center() -> None:
                    glNormal3f(ncx, ncy, -ncz)
                    glColor3f(rc * fade, gc * fade, bc * fade)
                    glVertex3f(xc, hc, -zc)

                _emit_corner(ix, iz)
                _emit_corner(ix + 1, iz)
                _emit_center()
                _emit_corner(ix + 1, iz)
                _emit_corner(ix + 1, iz + 1)
                _emit_center()
                _emit_corner(ix + 1, iz + 1)
                _emit_corner(ix, iz + 1)
                _emit_center()
                _emit_corner(ix, iz + 1)
                _emit_corner(ix, iz)
                _emit_center()
            else:
                for tri in (
                    ((ix, iz), (ix + 1, iz), (ix + 1, iz + 1)),
                    ((ix, iz), (ix + 1, iz + 1), (ix, iz + 1)),
                ):
                    for (i, j) in tri:
                        _emit_corner(i, j)
    glEnd()
    glDisable(GL_POLYGON_OFFSET_FILL)

    glDisable(GL_FOG)
    glDisable(GL_LIGHTING)
    glDisable(GL_COLOR_MATERIAL)
    glDisable(GL_LIGHT1)
    glDisable(GL_DEPTH_TEST)

    mv[:] = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float64).ravel()
    pr[:] = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float64).ravel()

    sx, sy, sz = gluProject(probe_x, py, probe_z_gl, mv, pr, viewport)
    if sz is not None and 0.0 <= sz <= 1.0:
        probe_xy = (int(sx), int(h - 1 - sy))

    return probe_xy, mv, pr, viewport


def project_world(
    x: float,
    y: float,
    z: float,
    mv: np.ndarray,
    pr: np.ndarray,
    viewport: list,
    h: int,
) -> Optional[Tuple[int, int]]:
    from OpenGL.GLU import gluProject

    # Same Z axis as terrain vertices in draw_background_terrain.
    sx, sy, sz = gluProject(x, y, -z, mv, pr, viewport)
    if sz is not None and 0.0 <= sz <= 1.0:
        return int(sx), int(h - 1 - sy)
    return None


def draw_overlay_and_flip(ov_surf: pygame.Surface, w: int, h: int) -> None:
    from OpenGL.GL import GL_DEPTH_TEST, glDisable, glViewport

    _ensure_textures(w, h)
    _upload_surface(_tex_ov, ov_surf)
    glViewport(0, 0, w, h)
    glDisable(GL_DEPTH_TEST)
    _draw_fullscreen_textured_quad(w, h, _tex_ov, 1.0)
    pygame.display.flip()


def copy_orbit_state(dst: "Renderer", src: "Renderer") -> None:
    dst.orbit_yaw = src.orbit_yaw
    dst.orbit_pitch = src.orbit_pitch
    dst._orbit_drag = False
    if getattr(src, "_gl_terrain_scale_smooth", None) is not None:
        dst._gl_terrain_scale_smooth = src._gl_terrain_scale_smooth
