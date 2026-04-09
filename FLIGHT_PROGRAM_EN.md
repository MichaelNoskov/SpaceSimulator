# Flight program

**Русский:** [FLIGHT_PROGRAM_RU.md](FLIGHT_PROGRAM_RU.md)

---

In the pause menu (**Esc** / **Space**), choose **«Flight program»**. The editor runs a restricted dialect of **Python** on **every simulation step** while **Auto** is on and the flight is still **running**.

## Entry point

Define:

```python
def tick(sim, ap):
    ...
```

- **`sim`** is read-only: altitude, speeds, stage flags, `can_*` guards.
- **`ap`** issues commands for the current step: pyro events, engine, throttle slider (0…1).

Use the **right-hand hint list** in the editor; **Tab** completes after `sim.` or `ap.`.

## Allowed language features

- Safe builtins: `abs`, `min`, `max`, `round`, `len`, `range`, `enumerate`, `zip`, `sum`, `any`, `all`, `pow`, `divmod`, and basic types (`int`, `float`, `bool`, `str`, `tuple`, `list`, `dict`, `set`).
- The **`math`** module (e.g. `math.sqrt`, `math.sin`).
- Global **`sleep(t)`** (see below).
- **`import` and file access are disabled** in the sandbox (still not a full security boundary).

## `sim` (read-only)

| Name | Meaning |
|------|---------|
| `altitude_m` | Height above local terrain, m |
| `vertical_speed_mps` | Vertical speed (+up), m/s |
| `horizontal_speed_mps` | Horizontal speed, m/s |
| `time_s` | Simulation time since start, s |
| `result` | Outcome (`running`, …) |
| `engine_on` | Engine enabled |
| `heatshield_jettisoned`, `drogue_deployed`, `main_deployed`, `chute_jettisoned` | Stage flags |
| `can_heatshield_jettison`, `can_drogue`, `can_main`, `can_chute_jettison` | Whether an action is allowed |
| `fuel_kg` | Fuel mass, kg |
| `g_load` | G-load |
| `atm_pressure_bar`, `internal_temp_c`, `heatshield_skin_temp_c` | Pressure, bay temperature, heatshield skin temperature (until jettison) |
| `distance_to_target_m` | Distance to map target, m |
| `stat(key, default)` | Extra HUD-style keys (`h_km`, `v_vert`, `dist_m`, …) |

## `ap` (actions)

| Call | Effect |
|------|--------|
| `request_heatshield_jettison()` | Request heatshield jettison |
| `request_drogue()` | Drogue chute |
| `request_main()` | Main chute |
| `request_chute_jettison()` | Jettison chute |
| `set_engine(on)` | Engine on/off |
| `set_throttle_slider(v)` | Throttle slider 0…1 |
| `clamp(x, lo, hi)` | Clamp a number |
| `sleep(t)` | Same as global `sleep(t)` (see below) |

### `sleep(t)` — simulation-time pause

**`sleep(t)`** (you may call **`sleep(t)`** or **`ap.sleep(t)`** inside `tick`) pauses the autopilot for **`int(t)`** whole seconds of **simulation** time (`PhysicsModel.time_s`, same as **`sim.time_s`**). The fractional part is discarded (like built-in **`int`**); **`t < 0`** is treated as **`0`**:

- the **current** `tick` call **ends immediately** — any code **after** `sleep` in that call **does not run**;
- until `sim.time_s` reaches the scheduled wake time, **whole `tick` calls are skipped** (no program output);
- then `tick` runs again **from the top** on each step as usual.

On **save validation**, `sleep` is a no-op. A **mission restart** clears the sleep schedule.

**Sleep does not pause the simulation clock**: while a sleep is pending, `tick` is skipped, but model time still advances (including time-warp).

## Code editor

The simulation **stays paused** while the editor is open. The hint panel links to this doc (**See FLIGHT_PROGRAM_EN.md**; Russian UI shows **FLIGHT_PROGRAM_RU.md**).

### Buttons

| Button | Action |
|--------|--------|
| **Save** | Compile, validate `tick` (below); on success close the editor and return to the pause menu |
| **Cancel** | Restore the last **successfully saved** text and close the editor (**Esc** does the same) |
| **Default** | Load the built-in sample (`DEFAULT_SCRIPT`) into the editor **without** applying it as the active flight program until you save |

### Hints and typing

- **Right-hand panel** — `sim` and `ap` members; **click** a row to insert at the caret. **Mouse wheel** scrolls the list over the panel.
- **Syntax highlighting** for Python keywords, strings, comments, numbers, and **`sim`** / **`ap`** names.
- **Tab** — complete after **`sim.`** or **`ap.`**; otherwise inserts **four spaces** (indent).

### Selection and clipboard

- **Mouse**: drag to select; **Shift+click** extends the selection from the previous caret.
- **Shift+arrows**, **Shift+Home/End** — adjust the selection.
- **Ctrl+A** — select all.
- **Ctrl+C** / **Ctrl+X** / **Ctrl+V** — copy, cut, paste (multi-line supported). Uses the system clipboard via Pygame when available; otherwise an **in-session internal** buffer.
- **Backspace** and **Delete** remove the selection first when present. **Key repeat** applies when keys are held.

### Other

- **Ctrl+S** — same as **Save**.
- With **Ctrl** (or **Cmd**) held, stray **TEXTINPUT** events are ignored so shortcuts stay reliable.

## Save and validation

**Save** runs in order:

1. **Compile** the restricted Python; on syntax error or missing `tick(sim, ap)` the previously saved program **stays** active and the error is shown in the editor.
2. **One dry-run** `tick` with a no-op `ap` (no real commands, `sleep` disabled). On exception the save is **reverted** and the editor **stays open**.
3. On success the new source becomes the **Auto** program and the editor closes.

## Errors during flight

Exceptions inside **`tick`** (other than the internal **`sleep`** mechanism) turn **Auto** off and show a toast. Avoid catching **`BaseException`** broadly in user code, or **`sleep`** handling may break.

## Code locations

- API: `flight_program/runner.py`
- Syntax highlighting: `flight_program/highlighter.py`
- Default script: `DEFAULT_SCRIPT` in `flight_program/runner.py`
