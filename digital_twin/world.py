from __future__ import annotations

import math
from dataclasses import dataclass

from .types import SurfaceType


@dataclass(frozen=True)
class WorldParams:
    """
    Procedural world tuning.

    Chosen so that:
    - height is smooth (hills/plains/basins) without sawtooth artifacts
    - lakes form large connected regions plus rare pockets
    - sampling stays fast for frequent use (minimap)
    """

    # Horizontal scales [m] for height layers.
    continental_cell_m: float = 60_000.0
    mountain_cell_m: float = 9_000.0
    detail_cell_m: float = 2_100.0

    # Amplitudes [m] for large height variation (visual + minimap).
    continental_amp_m: float = 3400.0
    mountain_amp_m: float = 2950.0
    detail_amp_m: float = 640.0

    # Ridge component: steeper slopes.
    ridge_amp_m: float = 1400.0
    ridge_power: float = 2.2

    # Dunes [m]: ridges, mostly on dry land.
    dunes_amp_m: float = 95.0
    dunes_cell_m: float = 1_400.0
    dunes_dir_rad: float = 0.55  # ridge orientation [rad], varies with seed
    dunes_sharpness: float = 3.0  # higher = stronger banding

    # Domain warp for more organic blobs.
    warp_cell_m: float = 40_000.0
    warp_amp_m: float = 38_000.0

    # Lakes: base threshold and pocket boost.
    lake_threshold: float = 0.445
    pocket_strength: float = 0.070

    # Water surface depth vs base terrain [m].
    water_depth_m: float = 50.0


class WorldGen:
    """
    Deterministic infinite world from a seed.

    API:
    - height_m_at(x,z): terrain height [m] (smooth hills/plains/basins)
    - surface_type_at(x,z): LAND/LAKE
    """

    def __init__(self, seed: int = 1337, params: WorldParams | None = None):
        self.seed = int(seed)
        self.p = params or WorldParams()

    # -----------------------
    # low-level deterministic noise
    # -----------------------
    def _hash01(self, ix: int, iz: int, salt: int = 0) -> float:
        """
        Fast deterministic hash in [0..1) for grid nodes.

        No np.random: O(1) and stateless.
        """

        # Mix coordinates + seed into float chaos. Pragmatic hash for visuals.
        s = float(self.seed * 0.001 + salt * 0.017)
        v = math.sin((ix * 127.1 + iz * 311.7) * 0.157 + s) * 43758.5453123
        return float(v - math.floor(v))

    @staticmethod
    def _smoothstep(t: float) -> float:
        t = max(0.0, min(1.0, float(t)))
        return t * t * (3.0 - 2.0 * t)

    def _value_noise(self, x: float, z: float, cell_m: float, salt: int = 0) -> float:
        """Smooth value noise in [0..1) with bilinear interpolation and smoothstep."""

        gx = float(x) / float(cell_m)
        gz = float(z) / float(cell_m)
        ix0 = int(math.floor(gx))
        iz0 = int(math.floor(gz))
        fx = gx - ix0
        fz = gz - iz0

        v00 = self._hash01(ix0, iz0, salt=salt)
        v10 = self._hash01(ix0 + 1, iz0, salt=salt)
        v01 = self._hash01(ix0, iz0 + 1, salt=salt)
        v11 = self._hash01(ix0 + 1, iz0 + 1, salt=salt)

        sx = self._smoothstep(fx)
        sz = self._smoothstep(fz)

        vx0 = v00 * (1.0 - sx) + v10 * sx
        vx1 = v01 * (1.0 - sx) + v11 * sx
        return float(vx0 * (1.0 - sz) + vx1 * sz)

    def _fbm(self, x: float, z: float, base_cell_m: float, octaves: int, lacunarity: float, gain: float, salt: int) -> float:
        """
        Fractal Brownian motion on value noise.

        Output roughly ~[0..1], not strict — normalized where used.
        """

        amp = 1.0
        freq = 1.0
        s = 0.0
        amp_sum = 0.0
        for k in range(int(octaves)):
            cell = float(base_cell_m) / max(1e-9, float(freq))
            s += amp * self._value_noise(x, z, cell_m=cell, salt=salt + 31 * k)
            amp_sum += amp
            amp *= float(gain)
            freq *= float(lacunarity)
        return float(s / max(1e-9, amp_sum))

    # -----------------------
    # public: terrain height + lakes
    # -----------------------
    def _warp(self, x: float, z: float) -> tuple[float, float]:
        """Low-frequency domain warp for organic lumps."""

        cell = float(self.p.warp_cell_m)
        n1 = self._fbm(x + 10_000.0, z - 20_000.0, base_cell_m=cell, octaves=3, lacunarity=2.0, gain=0.55, salt=101)
        n2 = self._fbm(x - 30_000.0, z + 5_000.0, base_cell_m=cell, octaves=3, lacunarity=2.0, gain=0.55, salt=202)
        wx = (n1 - 0.5) * float(self.p.warp_amp_m)
        wz = (n2 - 0.5) * float(self.p.warp_amp_m)
        return float(x + wx), float(z + wz)

    def height_m_at(self, x_m: float, z_m: float) -> float:
        """
        Terrain height [m].

        Absolute terrain height; altitude above surface is probe z_msl - height_m_at(x,z)
        (see PhysicsModel.altitude_m).
        """

        x = float(x_m)
        z = float(z_m)
        xw, zw = self._warp(x, z)

        cont = self._fbm(
            xw,
            zw,
            base_cell_m=float(self.p.continental_cell_m),
            octaves=4,
            lacunarity=2.0,
            gain=0.55,
            salt=1,
        )
        mtn = self._fbm(
            xw + 7_000.0,
            zw - 11_000.0,
            base_cell_m=float(self.p.mountain_cell_m),
            octaves=5,
            lacunarity=2.15,
            gain=0.55,
            salt=11,
        )
        det = self._fbm(
            xw - 3_000.0,
            zw + 9_000.0,
            base_cell_m=float(self.p.detail_cell_m),
            octaves=3,
            lacunarity=2.2,
            gain=0.50,
            salt=21,
        )

        # Plains/basins/mountains: continental = large shapes, mountain = bumps/ridges, detail = fine roughness.
        # Center roughly in [-0.5..+0.5] and scale to meters.
        cont_c = (cont - 0.5) * 2.0
        mtn_c = (mtn - 0.5) * 2.0
        det_c = (det - 0.5) * 2.0

        # Ridged mountains: sharper relief.
        # Classic ridge = (1 - |2n-1|)^p with n in [0..1].
        ridge = 1.0 - abs(2.0 * float(mtn) - 1.0)
        ridge = max(0.0, min(1.0, float(ridge))) ** float(self.p.ridge_power)
        ridge = 2.0 * (ridge - 0.5)  # ~[-1..+1]

        h = (
            cont_c * float(self.p.continental_amp_m)
            + mtn_c * float(self.p.mountain_amp_m)
            + det_c * float(self.p.detail_amp_m)
            + ridge * float(self.p.ridge_amp_m)
        )

        lake_val = self._lake_field_at_warped(xw, zw)
        is_lake = lake_val < float(self.p.lake_threshold)

        h_lim = 6200.0

        if is_lake:
            return float(h_lim * math.tanh((h - float(self.p.water_depth_m)) / h_lim))

        dunes = self._dunes_height_m_at(xw, zw)
        h = float(h + dunes)
        return float(h_lim * math.tanh(h / h_lim))

    def _dunes_height_m_at(self, xw: float, zw: float) -> float:
        """
        Dunes: periodic ridges (inspired by Titan dune fields).

        Implemented as bands along a fixed direction with noisy amplitude modulation.
        """

        # Ridge direction varies slightly with seed.
        ang = float(self.p.dunes_dir_rad + 0.35 * math.sin(self.seed * 0.0017))
        ca = math.cos(ang)
        sa = math.sin(ang)

        # Coordinate along ridge direction.
        u = ca * float(xw) + sa * float(zw)

        # Banded profile: |sin|^k gives plateaus and troughs.
        phase = (u / float(self.p.dunes_cell_m)) * (2.0 * math.pi)
        s = abs(math.sin(phase))
        ridge = s ** float(self.p.dunes_sharpness)
        ridge = 2.0 * (ridge - 0.5)  # ~[-1..+1]

        # Mask so dunes form fields instead of covering everything.
        mask = self._fbm(xw + 40_000.0, zw - 15_000.0, base_cell_m=18_000.0, octaves=3, lacunarity=2.0, gain=0.55, salt=401)
        mask = max(0.0, min(1.0, (mask - 0.35) / 0.35))
        mask = mask * mask

        return float(ridge * float(self.p.dunes_amp_m) * mask)

    def _lake_field_at_warped(self, xw: float, zw: float) -> float:
        """lake_field_at with pre-warped coordinates (avoids double _warp call)."""
        cont = self._value_noise(xw, zw, cell_m=22_000.0, salt=301)
        mid = self._value_noise(xw + 7_000.0, zw - 11_000.0, cell_m=6_000.0, salt=302)
        detail = (self._value_noise(xw - 3_000.0, zw + 9_000.0, cell_m=2_500.0, salt=303) - 0.5) * 0.06
        small = self._value_noise(xw + 21_000.0, zw + 17_000.0, cell_m=3_500.0, salt=304)
        gate = 1.0 - max(0.0, min(1.0, (0.22 - cont) / 0.12))
        pocket = max(0.0, min(1.0, (0.32 - small) / 0.32))
        pocket = float(pocket * pocket)
        small_term = -float(self.p.pocket_strength) * float(gate) * pocket
        return float(0.78 * cont + 0.20 * mid + float(detail) + float(small_term))

    def lake_field_at(self, x_m: float, z_m: float) -> float:
        """
        Lake-ness field (pseudo-scalar) used for thresholding.

        Goal: large connected lakes plus rare pockets, avoiding salt-and-pepper noise.
        """

        x = float(x_m)
        z = float(z_m)
        xw, zw = self._warp(x, z)

        cont = self._value_noise(xw, zw, cell_m=22_000.0, salt=301)
        mid = self._value_noise(xw + 7_000.0, zw - 11_000.0, cell_m=6_000.0, salt=302)
        detail = (self._value_noise(xw - 3_000.0, zw + 9_000.0, cell_m=2_500.0, salt=303) - 0.5) * 0.06

        small = self._value_noise(xw + 21_000.0, zw + 17_000.0, cell_m=3_500.0, salt=304)
        gate = 1.0 - max(0.0, min(1.0, (0.22 - cont) / 0.12))
        pocket = max(0.0, min(1.0, (0.32 - small) / 0.32))
        pocket = float(pocket * pocket)
        small_term = -float(self.p.pocket_strength) * float(gate) * pocket

        return float(0.78 * cont + 0.20 * mid + float(detail) + float(small_term))

    def is_lake_at(self, x_m: float, z_m: float) -> bool:
        """True if point is lake (below lake_threshold)."""

        return bool(self.lake_field_at(x_m, z_m) < float(self.p.lake_threshold))

    def surface_type_at(self, x_m: float, z_m: float) -> SurfaceType:
        """Surface type at point: land or lake."""

        return SurfaceType.LAKE if self.is_lake_at(x_m, z_m) else SurfaceType.LAND
