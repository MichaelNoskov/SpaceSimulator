# Архив NASA / PDS в этом каталоге

Копии открытых продуктов **Cassini–Huygens** (TAB/LBL, документация в PDF) для автономной работы с репозиторием.

Таблицы для движка симуляции (`data/titan_atm.json`, `data/titan_wind_huygens.json`, `data/huygens_velocity_telemetry.json`) при необходимости пересобираются из TAB командой:

```bash
python3 scripts/parse_pds_titan.py
```

## TAB/LBL (Planetary Data System)

| Путь | Набор |
|------|--------|
| `hasi_profiles/HASI_L4_ATMO_PROFILE_*.TAB` | HASI L4: атмосфера, вход и спуск |
| `hasi_profiles/HASI_L4_VELOCITY_PROFILE.TAB` | HASI L4: скорость зонда |
| `dwe/ZONALWIND.TAB` | DWE: зональный ветер по спуску |

Корни зеркала PDS Atmospheres Node (NMSU):

- https://atmos.nmsu.edu/PDS/data/PDS4/Huygens/hphasi_bundle/
- https://atmos.nmsu.edu/PDS/data/PDS4/Huygens/hpdwe_bundle/

Каталог PDS:

- https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=HP-SSA-HASI-2-3-4-MISSION-V1.1
- https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=HP-SSA-DWE-2-3-DESCENT-V1.0

## PDF (`pdf/`)

| Файл | Содержание |
|------|------------|
| `cassinifactsheet_nasa_science.pdf` | Cassini fact sheet (NASA Science; тот же документ, что по ссылке в README) |
| `descanso3_cassini_telecom_jpl.pdf` | JPL Descanso: телекоммуникации Cassini |
| `hasi_fulchignoni_2005_pds_atmospheres.pdf` | HASI / Титан (копия из PDS DOCUMENT) |
| `dwe_bird_nature2005_pds_atmospheres.pdf` | DWE / ветер на Титане, Nature 2005 (копия из PDS DOCUMENT) |
