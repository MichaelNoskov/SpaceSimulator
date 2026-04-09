# Titan Landing Simulator (Pygame)

**Russian:** [README.md](README.md)

---

## Contents

| Section | Description |
|---------|-------------|
| [About](#about) | Purpose and scope of the model |
| [Installing](#installing-python-and-the-project) | Python, venv, dependencies |
| [Run](#run) | `main.py` and standalone binaries |
| [Controls](#controls-using-the-simulator) | Console, keys, CSV |
| [Flight program (EN)](FLIGHT_PROGRAM_EN.md) | Autopilot script (`tick(sim, ap)`) |
| [Программа полёта (RU)](FLIGHT_PROGRAM_RU.md) | Same document in Russian |
| [Key formulas](#key-formulas) | Thrust, Mach, drag, atmosphere |
| [Landing phases](#landing-phases) | Descent timeline |
| [Code map](#code-map) | Project layout |
| [Data and license](#data-and-license) | `data/`, MIT |
| [Links](#links) | ESA Huygens |

---

## About

This is an interactive educational simulator of descent and soft landing on Titan. You control the sequence of descent systems, pick a site on the map, and try to finish within load, temperature, and touchdown speed limits.

Physics lives in `digital_twin`; Pygame handles the world and cockpit so you can tune the model without touching graphics.

### What the simulation includes

| Area | Content |
|------|---------|
| **Gravity** | Constant acceleration at Titan’s surface |
| **Atmosphere** | Profile from `data/titan_atm.json`; exponential height profile if the table is unavailable |
| **Gas state** | Bulk $\rho$, $T$, $P$ vs altitude in the equations of motion |
| **Aerodynamics** | Quadratic drag; parachutes and heatshield jettison change $C_d$ and area |
| **Wind** | Height-dependent; shifts air-relative velocity |
| **Thermal** | Cabin / internal temperature model |
| **Scene** | Clouds and haze in the visualization |
| **Propulsion** | Thrust and fuel consumption |
| **Surface** | Procedural terrain, land and liquid regions |
| **Outcome** | Success / failure: overload, temperature, fuel, hard landing, surface-type mismatch vs target, terrain collision |

---

## Installing Python and the project

### 1. Python

**Python 3.10+** required.

| Platform | Steps |
|----------|--------|
| **Windows** | Installer from [python.org](https://www.python.org/downloads/), enable **Add Python to PATH**. Check: `python --version` |
| **Linux** | e.g. `sudo apt install python3 python3-venv python3-pip`. Check: `python3 --version` |
| **macOS** | e.g. `brew install python@3.12` or the python.org installer |

### 2. Project folder

```bash
cd /path/to/titan
```

### 3. Virtual environment (recommended)

```bash
python3 -m venv .venv
```

Activate:

| Shell | Command |
|-------|---------|
| **Linux / macOS** | `source .venv/bin/activate` |
| **Windows (cmd)** | `.venv\Scripts\activate.bat` |
| **Windows (PowerShell)** | `.venv\Scripts\Activate.ps1` |

### 4. Dependencies

```bash
pip install -r requirements.txt
```

---

## Run

```bash
python main.py
```

The game starts fullscreen by default. **F11** toggles fullscreen.

### Ready-built binaries (`standalone/`)

Prebuilt apps live under **`standalone/`** — no Python install required.

| Platform | File | How to run |
|----------|------|------------|
| **Linux x86_64** | `standalone/linux/TitanLandingSimulator` | From repo root: `./standalone/linux/TitanLandingSimulator` (if needed: `chmod +x standalone/linux/TitanLandingSimulator`) |
| **Windows** | `standalone/windows/TitanLandingSimulator.exe` | produced locally by `scripts\build_windows.ps1` |

Rebuild: Linux — `./scripts/build_linux.sh`, Windows — `.\scripts\build_windows.ps1`.

If the Linux one-file binary fails with **`Failed to extract libcrypto.so.3`** or a decompression error: check free space under **`/tmp`** and on the drive (PyInstaller unpacks there; a full disk breaks extraction). Run a build that matches your OS ABI (x86_64).

---

## Controls: using the simulator

The UI is an **operator console**: instruments show the flight state; levers and sliders are your hands on the vehicle; the minimap ties you to the chosen landing site.

### Minimap and target

**Click the minimap** to set a goal in world coordinates and the **expected surface type** (land vs liquid) at that point. Success requires soft speeds **and** a match between the surface under the probe and the target’s stored type.

### Levers (heatshield and parachutes)

- **Heatshield jettison** is allowed when **Mach** $M = v_{\mathrm{rel}} / a$ falls below the limit. The speed of sound $a$ uses the ideal-gas model (see [Key formulas](#key-formulas)). Blocked while hypersonic.
- **Drogue**, then **main** chute, within altitude gates; **chute jettison** near the ground only after a minimum **science descent** time under the main canopy (default 2 h simulation time; time warp shortens wall-clock wait).
- A **green lamp** on a lever means the action is **currently allowed** by the safety rules.

### Engine

**On/off** toggle and **throttle** slider (0–100 %). Fuel mass flow follows the relation in [Key formulas](#key-formulas). Running the engine out of fuel while airborne can fail the run.

### Auto vs manual

| Mode | Behavior |
|------|----------|
| **Auto** | Sequence from the editable **flight program** (`tick(sim, ap)`); default matches the classic heatshield → chutes → engine assist |
| **Manual** | You drive every decision — good for experiments and learning |

### Pause menu

**Pause** (top right), **Esc**, or **Space** open the menu: resume, auto/manual, **RU/EN**, restart, **CSV** logging, **flight program** editor, quit.

### Flight program (Auto mode)

**Auto** is driven by a restricted Python script edited from the pause menu (**Flight program**): API hints, syntax highlighting, and a dry-run **`tick()`** check on save. Full reference — **[FLIGHT_PROGRAM_EN.md](FLIGHT_PROGRAM_EN.md)**; in Russian — **[FLIGHT_PROGRAM_RU.md](FLIGHT_PROGRAM_RU.md)**.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| **R** | Quick restart |
| **A** | Toggle auto / manual |
| **F1** | Short help overlay |
| **+** / **−** | Faster / slower simulation time |
| **F11** | Fullscreen |
| **Esc**, **Space** | Pause menu |

After landing, **telemetry plots** overlay the HUD; the **Pause** control is drawn on top and stays usable.

### CSV trajectory log

From the pause menu you can enable logging under `logs/` next to the project.

---

## Key formulas

Symbols below match how quantities are used in code and on instruments.

### Fuel mass flow (engine model)

For thrust $T$:

$$
\dot{m} \;=\; \frac{T}{I_{\mathrm{sp}}\, g_0}
$$

| Symbol | Meaning | Typical units |
|--------|---------|----------------|
| $\dot{m}$ | Fuel mass flow rate | kg/s |
| $T$ | Thrust | N |
| $I_{\mathrm{sp}}$ | Specific impulse | s |
| $g_0$ | Standard gravity | 9.80665 m/s² |

### Speed of sound and Mach number

$$
a \;=\; \sqrt{\gamma\, R\, T(h)}
$$

$$
M \;=\; \frac{|\mathbf{v}_{\mathrm{rel}}|}{a}
$$

| Symbol | Meaning |
|--------|---------|
| $\gamma$ | Heat-capacity ratio |
| $R$ | Specific gas constant (mixture; N₂-dominated reference) |
| $T(h)$ | Temperature from the atmospheric profile |
| $\mathbf{v}_{\mathrm{rel}}$ | Velocity relative to the air (wind included) |

### Quadratic aerodynamic drag

$$
\mathbf{F}_d \;=\; -\tfrac{1}{2}\,\rho(h)\, C_d\, A\, \bigl|\mathbf{v}_{\mathrm{rel}}\bigr|\,\mathbf{v}_{\mathrm{rel}}
$$

| Symbol | Meaning |
|--------|---------|
| $\rho(h)$ | Air density vs altitude |
| $C_d$ | Drag coefficient (configuration-dependent) |
| $A$ | Reference area |

### Atmosphere vs altitude

At each altitude (table or exponential profile):

$$
\rho = \rho(h),\qquad T = T(h),\qquad P = P(h)
$$

---

## Landing phases

A Huygens-like descent chain in the simulator’s terms.

| # | Phase | What happens |
|---|-------|----------------|
| 1 | **Entry (`entry`)** | High speed; $\rho$ grows downward; strong aerobraking. **$\rho(h), T(h), P(h)$**; **wind** shifts $\mathbf{v}_{\mathrm{rel}}$ and $\mathbf{F}_d$. Visuals: sky gradient, haze, clouds; heatshield jettison can spawn particles. |
| 2 | **Heatshield jettison** | After $M$ drops: lower mass; new $C_d$ and area. |
| 3 | **Drogue (`drogue_chute`)** | Large $C_d$ and $A$ — strong deceleration in dense air. |
| 4 | **Main + science hold (`science_descent` / `main_chute`)** | Even larger area and $C_d$; minimum time under canopy before jettison. |
| 5 | **Chute jettison, final descent** | Back to “bare” aero; **engine** (manual or auto) near the ground. |
| 6 | **Touchdown** | Speed limits (higher vertical cap on liquid), overload, temperature, surface vs target, extreme impact / penetration under the terrain height model. |

**Terrain:** procedural height and lakes; minimap and low-altitude **profile** show **dunes** and **lakes**.

---

## Code map

| Path | Role |
|------|------|
| `main.py` | Pygame loop, physics accumulator, `Renderer.draw` |
| `digital_twin/model.py` | `PhysicsModel`: forces, integrator, landing, CSV, plot history |
| `digital_twin/dynamics.py` | Drag, thrust, fuel, thermal, Euler helpers for `SimState` |
| `digital_twin/config.py` | Body, engine, thermal, Mach and science-descent settings |
| `digital_twin/models/atmosphere.py` | JSON table and exponential profile |
| `digital_twin/models/wind.py` | Wind vs altitude |
| `digital_twin/world.py` | Terrain and surface type |
| `control/` | UI commands → model |
| `flight_program/` | Autopilot script: `runner.py` (API, validation), `highlighter.py` (editor colors) |
| `ui.py` | Instruments, levers, i18n, post-landing plots |
| `render.py` | World draw order, minimap |

---

## Data and license

| File | Role |
|------|------|
| `data/titan_atm.json` | Atmosphere profile and metadata (including open-data URLs) |
| `data/surface_map.meta.json` | Metadata and references for the surface mask |
| `LICENSE` | **MIT** |

---

## Links

- ESA: [Huygens overview](https://sci.esa.int/web/cassini-huygens/-/47052-huygens)
