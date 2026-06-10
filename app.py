"""
Dash + dash-leaflet version of the base-station placement simulator.

Run:
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import base64
import io
import json
import logging
import logging.handlers
import os
import threading
import time
import traceback
import uuid
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    ctx,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate
import dash_leaflet as dl
from dash_extensions.javascript import assign
from geopy.distance import geodesic

from environment import SyntheticEnvironment
from obstacle_sources import (
    filter_polygons,
    geojson_to_polygons,
    load_osm_polygons_with_cache,
)
from optimizers import (
    REGISTRY,
    ProblemInput,
    convert_to_geo,
    get_optimizer,
    sinr_coverage,
)
from patterns import PATTERN_CHOICES


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_optimizer_logging() -> logging.Logger:
    """optimizers 패키지 전용 로거: logs/optimizer.log (rotating) + stderr."""
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("optimizers")
    if logger.handlers:          # 재진입(reload) 방지
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # 파일 핸들러: 최대 2 MB, 백업 3개
    fh = logging.handlers.RotatingFileHandler(
        "logs/optimizer.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # 콘솔 핸들러: WARNING 이상만
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger

_opt_logger = _setup_optimizer_logging()

STATION_PIN_MARKER_ENABLED = True


# ---------------------------------------------------------------------------
# Constants / server state
# ---------------------------------------------------------------------------

DEFAULT_CENTER = [37.4979, 127.0276]
DEFAULT_ZOOM = 14

OSM_OBSTACLE_TYPE_LABELS = ["건물", "수역/물길", "도로"]
OSM_OBSTACLE_TYPE_VALUES = {
    "건물": "building",
    "수역/물길": ("water", "waterway"),
    "도로": "road",
}
OSM_OBJECT_USAGE_MODES = ["장애물로 사용", "기지국 후보로 사용"]

APP_STATE: dict[str, dict[str, Any]] = {}
_LAST_ACCESSED: dict[str, float] = {}
_SESSION_TTL = 3_600.0  # 1시간 미접근 세션 자동 삭제
_APP_STATE_LOCK = threading.Lock()


TRAFFIC_STYLE = assign(
    """
function(feature, context){
    return {
        fillColor: feature.properties.fillColor || "#ff0000",
        color: "transparent",
        weight: 0,
        fillOpacity: feature.properties.fillOpacity || 0.2,
        interactive: feature.properties.interactive || false
    };
}
"""
)

TRAFFIC_ON_EACH_FEATURE = assign(
    """
function(feature, layer, context){
    var p = feature.properties || {};
    layer.bindTooltip(
        "Traffic: " + (p.traffic ?? "-") +
        "<br>Status: " + (p.status ?? "-") +
        "<br>Area: " + (p.obstacle ?? "-"),
        {sticky: true}
    );
}
"""
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def get_session_state(session_id: str) -> dict[str, Any]:
    if not session_id:
        raise PreventUpdate
    now = time.time()
    with _APP_STATE_LOCK:
        _LAST_ACCESSED[session_id] = now
        if len(APP_STATE) > 50:
            stale = [sid for sid, ts in list(_LAST_ACCESSED.items()) if now - ts > _SESSION_TTL]
            for sid in stale:
                APP_STATE.pop(sid, None)
                _LAST_ACCESSED.pop(sid, None)
        return APP_STATE.setdefault(session_id, {})


def version_token() -> dict[str, Any]:
    return {"version": time.time()}


def normalize_triggered_bool(value: Any) -> bool:
    return bool(value)


def decode_upload_to_bytes(contents: str | None) -> io.BytesIO | None:
    if not contents:
        return None
    try:
        _, encoded = contents.split(",", 1)
        return io.BytesIO(base64.b64decode(encoded))
    except Exception:
        return None


def safe_float(value: Any, default: float) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_map_center(center: Any) -> tuple[float, float]:
    if isinstance(center, dict):
        lat = center.get("lat", center.get("latitude"))
        lon = center.get("lng", center.get("lon", center.get("longitude")))
        if lat is not None and lon is not None:
            return float(lat), float(lon)

    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])

    return float(DEFAULT_CENTER[0]), float(DEFAULT_CENTER[1])


def parse_map_bounds(bounds: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not bounds:
        return None

    if isinstance(bounds, dict):
        sw = bounds.get("_southWest") or bounds.get("southWest")
        ne = bounds.get("_northEast") or bounds.get("northEast")
        if isinstance(sw, dict) and isinstance(ne, dict):
            sw_lat = sw.get("lat")
            sw_lng = sw.get("lng", sw.get("lon"))
            ne_lat = ne.get("lat")
            ne_lng = ne.get("lng", ne.get("lon"))
            if None not in (sw_lat, sw_lng, ne_lat, ne_lng):
                return (float(sw_lat), float(sw_lng)), (float(ne_lat), float(ne_lng))

    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        sw = bounds[0]
        ne = bounds[1]
        if isinstance(sw, (list, tuple)) and isinstance(ne, (list, tuple)) and len(sw) >= 2 and len(ne) >= 2:
            return (float(sw[0]), float(sw[1])), (float(ne[0]), float(ne[1]))

    return None


def ensure_station_spec_rows(
    rows: list[dict[str, Any]] | None,
    target_count: int,
    default_radius: float,
    default_capacity: float,
    default_tx_power: float = 43.0,
) -> list[dict[str, Any]]:
    rows = rows or []
    norm_rows: list[dict[str, Any]] = []

    for i in range(max(0, int(target_count))):
        old = rows[i] if i < len(rows) and isinstance(rows[i], dict) else {}
        norm_rows.append(
            {
                "station": i + 1,
                "radius_m": safe_float(old.get("radius_m"), default_radius),
                "capacity": safe_float(old.get("capacity"), default_capacity),
                "tx_power_dbm": safe_float(old.get("tx_power_dbm"), default_tx_power),
            }
        )

    return norm_rows


def coerce_station_tx_power_array(
    rows: list[dict[str, Any]] | None,
    station_points: int,
    fallback_tx: float,
) -> np.ndarray:
    rows = rows or []
    tx_power = []

    for i in range(station_points):
        if i < len(rows) and isinstance(rows[i], dict):
            tx_power.append(safe_float(rows[i].get("tx_power_dbm"), fallback_tx))
        else:
            tx_power.append(float(fallback_tx))

    return np.asarray(tx_power, dtype=float)


def coerce_station_capacity_array(
    rows: list[dict[str, Any]] | None,
    station_points: int,
    fallback_capacity: float,
) -> np.ndarray:
    rows = rows or []
    capacity = []

    for i in range(station_points):
        if i < len(rows) and isinstance(rows[i], dict):
            capacity.append(safe_float(rows[i].get("capacity"), fallback_capacity))
        else:
            capacity.append(float(fallback_capacity))

    return np.asarray(capacity, dtype=float)


def set_station_spec_rows_from_arrays(
    radius: np.ndarray | list[float],
    capacity: np.ndarray | list[float],
    tx_power: np.ndarray | list[float],
    fallback_radius: float,
    fallback_capacity: float,
    fallback_tx: float = 43.0,
) -> list[dict[str, Any]]:
    radius_arr = np.asarray(radius, dtype=float).reshape(-1)
    capacity_arr = np.asarray(capacity, dtype=float).reshape(-1)
    tx_arr = np.asarray(tx_power, dtype=float).reshape(-1)
    target_count = max(len(radius_arr), len(capacity_arr), len(tx_arr))

    rows: list[dict[str, Any]] = []

    for idx in range(target_count):
        rows.append(
            {
                "station": idx + 1,
                "radius_m": float(radius_arr[idx]) if idx < len(radius_arr) else float(fallback_radius),
                "capacity": float(capacity_arr[idx]) if idx < len(capacity_arr) else float(fallback_capacity),
                "tx_power_dbm": float(tx_arr[idx]) if idx < len(tx_arr) else float(fallback_tx),
            }
        )

    return rows


def tx_power_for_k(
    k: int,
    hetnet_enabled: bool,
    ui_tx_power: float,
    n_macro: int,
    n_small: int,
    macro_power: float,
    small_power: float,
    spec_mode: str,
    spec_rows: list[dict[str, Any]] | None,
) -> np.ndarray:
    if k <= 0:
        return np.zeros(0, dtype=float)

    if hetnet_enabled:
        base = np.concatenate(
            [
                np.full(max(0, int(n_macro)), float(macro_power), dtype=float),
                np.full(max(0, int(n_small)), float(small_power), dtype=float),
            ]
        )
        if len(base) == 0:
            base = np.asarray([float(ui_tx_power)], dtype=float)

    elif spec_mode == "기지국별 개별" and spec_rows:
        base = coerce_station_tx_power_array(spec_rows, k, float(ui_tx_power))

    else:
        base = np.asarray([float(ui_tx_power)], dtype=float)

    if len(base) < k:
        base = np.concatenate([base, np.full(k - len(base), float(base[-1]), dtype=float)])

    return base[:k].astype(float)


def capacity_for_k(
    k: int,
    spec_mode: str,
    spec_rows: list[dict[str, Any]] | None,
    capacity_default: float,
) -> np.ndarray:
    if spec_mode == "기지국별 개별" and spec_rows:
        return coerce_station_capacity_array(spec_rows, k, float(capacity_default))
    return np.full(k, float(capacity_default), dtype=float)


def prop_params_base(
    path_loss_exponent: float,
    bandwidth_mhz: float,
    sinr_threshold_db: float,
) -> dict[str, float]:
    bandwidth_mhz = max(float(bandwidth_mhz), 1e-9)
    noise_floor_dbm = -174.0 + 10.0 * np.log10(bandwidth_mhz * 1e6) + 7.0

    return {
        "path_loss_exponent": float(path_loss_exponent),
        "path_loss_ref_db": 38.0,
        "noise_floor_dbm": float(noise_floor_dbm),
        "sinr_threshold_db": float(sinr_threshold_db),
        "bandwidth_mhz": float(bandwidth_mhz),
    }


def radius_from_tx(tx_power_dbm: np.ndarray, prop: dict[str, float]) -> np.ndarray:
    n = max(float(prop["path_loss_exponent"]), 1e-9)

    exp = (
        tx_power_dbm
        - float(prop["path_loss_ref_db"])
        - float(prop["noise_floor_dbm"])
        - float(prop["sinr_threshold_db"])
    ) / (10.0 * n)

    return np.maximum(1.0, np.power(10.0, exp))



# ---------------------------------------------------------------------------
# Obstacle loading/application
# ---------------------------------------------------------------------------

def load_map_obstacles(
    env: SyntheticEnvironment,
    source: str,
    uploaded_geojson: io.BytesIO | None,
    min_area_m2: float,
    max_obstacles: int | None,
    osm_obstacle_types: list[str] | None = None,
    osm_object_mode: str = OSM_OBJECT_USAGE_MODES[0],
):
    if source == "OSM 지도 데이터":
        if not osm_obstacle_types:
            raise ValueError("OSM 오브젝트 종류를 하나 이상 선택해주세요.")

        try:
            geo_polygons, raw_count = load_osm_polygons_with_cache(
                env.lat_min,
                env.lon_min,
                env.lat_max,
                env.lon_max,
                obstacle_types=osm_obstacle_types,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'obstacle_types'" not in str(exc):
                raise
            geo_polygons, raw_count = load_osm_polygons_with_cache(
                env.lat_min,
                env.lon_min,
                env.lat_max,
                env.lon_max,
            )

    elif source == "GeoJSON 업로드":
        if uploaded_geojson is None:
            raise ValueError("GeoJSON 파일을 먼저 업로드해주세요.")
        geo_polygons = geojson_to_polygons(uploaded_geojson.getvalue())
        raw_count = len(geo_polygons)

    else:
        return [], 0

    local_polygons = []

    for polygon in geo_polygons:
        local_polygons.extend(env.geo_to_local_polygons(polygon))

    if osm_object_mode == "기지국 후보로 사용":
        candidate_points = [
            poly.representative_point().coords[0]
            for poly in local_polygons
            if poly.area > 0
        ]
        return candidate_points, raw_count

    return filter_polygons(local_polygons, min_area_m2, max_obstacles, coordinates_are_meters=True), raw_count


def apply_obstacle_source(
    env: SyntheticEnvironment,
    source: str,
    uploaded_geojson: io.BytesIO | None,
    min_area_m2: float,
    max_obstacles: int | None,
    obstacle_pattern: str,
    num_obstacles: int,
    osm_obstacle_types: list[str] | None = None,
    osm_object_mode: str = OSM_OBJECT_USAGE_MODES[0],
    append: bool = False,
) -> tuple[int, int]:
    def apply_candidate_mode(points):
        if append:
            env.append_station_candidate_points(points)
        else:
            env.set_station_candidate_points(points)

        env.obstacles = []
        env.obstacles_geo = []
        env.remask_traffic()

    if source == "합성":
        if osm_object_mode == "기지국 후보로 사용":
            generated = SyntheticEnvironment(
                center_lat=env.center_lat,
                center_lon=env.center_lon,
                width_km=env.width_km,
                height_km=env.height_km,
                resolution_m=env.resolution_m,
            )
            generated.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
            candidate_points = [
                poly.representative_point().coords[0]
                for poly in generated.obstacles
                if poly.area > 0
            ]
            apply_candidate_mode(candidate_points)
            return len(candidate_points), num_obstacles

        if append:
            before = len(env.obstacles)
            generated = SyntheticEnvironment(
                center_lat=env.center_lat,
                center_lon=env.center_lon,
                width_km=env.width_km,
                height_km=env.height_km,
                resolution_m=env.resolution_m,
            )
            generated.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
            env.append_obstacles(generated.obstacles)
            return len(env.obstacles) - before, num_obstacles

        env.generate_obstacles(num_obstacles=num_obstacles, pattern=obstacle_pattern)
        env.remask_traffic()
        return len(env.obstacles), num_obstacles

    polygons, raw_count = load_map_obstacles(
        env,
        source,
        uploaded_geojson,
        min_area_m2,
        max_obstacles,
        osm_obstacle_types=osm_obstacle_types,
        osm_object_mode=osm_object_mode,
    )

    if osm_object_mode == "기지국 후보로 사용":
        apply_candidate_mode(polygons)
        return len(env.station_candidate_points), raw_count

    if append:
        env.append_obstacles(polygons)
    else:
        env.replace_obstacles(polygons)
        env.clear_station_candidate_points()

    return len(polygons), raw_count


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------

def env_dataframe_for_current_frame(env: SyntheticEnvironment) -> pd.DataFrame:
    raw_series = env.get_raw_traffic_series()

    if raw_series is not None:
        flat_traffic = raw_series[env.dynamic_frame_index].ravel()
    else:
        flat_traffic = env.get_raw_traffic_map().ravel()

    obstacle_mask = env.get_obstacle_mask().ravel()

    return pd.DataFrame(
        {
            "lat": env.lat_grid.ravel(),
            "lon": env.lon_grid.ravel(),
            "traffic": flat_traffic,
            "is_obstacle": obstacle_mask,
        }
    )


def compute_status_overlay(
    env: SyntheticEnvironment,
    df: pd.DataFrame,
    opt_results: dict[str, Any] | None,
    opt_stats: dict[str, Any] | None,
    station_specs: list[dict[str, Any]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not opt_results or not opt_stats:
        return np.zeros(len(df), dtype=int), np.zeros(0, dtype=float)

    stations = opt_results.get("stations_geo")

    if stations is None or len(stations) == 0:
        return np.zeros(len(df), dtype=int), np.zeros(0, dtype=float)

    station_df = pd.DataFrame(stations)

    if station_df.empty or not {"lat", "lon"}.issubset(station_df.columns):
        return np.zeros(len(df), dtype=int), np.zeros(0, dtype=float)

    station_points = station_df[["lat", "lon"]].values

    prop = opt_results.get("prop_params", {})
    fallback_tx = float(np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()[0])
    tx = coerce_station_tx_power_array(station_specs, len(station_points), fallback_tx)

    capacity_default = float(opt_stats.get("capacity_default", 1000))
    capacity = coerce_station_capacity_array(station_specs, len(station_points), capacity_default)

    traffic_mask = df["traffic"] > 0.1
    grid_points = df.loc[traffic_mask, ["lat", "lon", "traffic"]].values
    grid_indices = np.where(traffic_mask.to_numpy())[0]

    if len(grid_points) == 0:
        return np.zeros(len(df), dtype=int), np.zeros(len(station_points), dtype=float)

    x_scale = env.width_m / max(env.lon_max - env.lon_min, 1e-12)
    y_scale = env.height_m / max(env.lat_max - env.lat_min, 1e-12)

    st_x = (station_points[:, 1] - env.lon_min) * x_scale
    st_y = (station_points[:, 0] - env.lat_min) * y_scale
    st_local = np.column_stack((st_x, st_y))

    gd_x = (grid_points[:, 1] - env.lon_min) * x_scale
    gd_y = (grid_points[:, 0] - env.lat_min) * y_scale
    gd_local = np.column_stack((gd_x, gd_y))

    prop_for_radius = {
        "path_loss_ref_db": float(prop.get("path_loss_ref_db", 38.0)),
        "noise_floor_dbm": float(prop.get("noise_floor_dbm", -97.0)),
        "sinr_threshold_db": float(prop.get("sinr_threshold_db", 3.0)),
        "path_loss_exponent": float(prop.get("path_loss_exponent", 3.5)),
        "bandwidth_mhz": float(prop.get("bandwidth_mhz", 10.0)),
    }

    problem = ProblemInput(
        X=gd_local,
        weights=grid_points[:, 2],
        width_m=env.width_m,
        height_m=env.height_m,
        radius_m=radius_from_tx(tx, prop_for_radius),
        capacity=capacity,
        lat_min=env.lat_min,
        lat_max=env.lat_max,
        lon_min=env.lon_min,
        lon_max=env.lon_max,
        path_loss_exponent=prop_for_radius["path_loss_exponent"],
        path_loss_ref_db=prop_for_radius["path_loss_ref_db"],
        tx_power_dbm=tx,
        noise_floor_dbm=prop_for_radius["noise_floor_dbm"],
        sinr_threshold_db=prop_for_radius["sinr_threshold_db"],
        bandwidth_mhz=prop_for_radius["bandwidth_mhz"],
    )

    is_cov, srv_idx, sinr_db = sinr_coverage(st_local, problem)

    grid_status = np.zeros(len(df), dtype=int)
    overlay_loads = np.zeros(len(station_points), dtype=float)
    station_allocs: list[list[tuple[int, float, float]]] = [[] for _ in range(len(station_points))]

    for i in range(len(grid_points)):
        if is_cov[i]:
            station_allocs[int(srv_idx[i])].append(
                (i, -float(sinr_db[i]), float(grid_points[i, 2]))
            )

    for s_idx, allocs in enumerate(station_allocs):
        allocs.sort(key=lambda x: x[1])
        current_load = 0.0
        station_capacity = capacity[s_idx] if s_idx < len(capacity) else 0.0

        for cell_i, _, traffic in allocs:
            if station_capacity <= 0 or current_load + traffic <= station_capacity:
                current_load += traffic
                grid_status[grid_indices[cell_i]] = 1
            else:
                grid_status[grid_indices[cell_i]] = 2

        overlay_loads[s_idx] = current_load

    return grid_status, overlay_loads


def build_traffic_geojson(
    env: SyntheticEnvironment,
    df: pd.DataFrame,
    map_layer_mode: str,
    status_list: np.ndarray,
    interactive: bool,
) -> dict[str, Any]:
    lat_step = (env.lat_max - env.lat_min) / max(env.rows, 1)
    lon_step = (env.lon_max - env.lon_min) / max(env.cols, 1)

    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    traffics = df["traffic"].to_numpy()
    is_obstacles = df["is_obstacle"].to_numpy(dtype=bool)

    features = []

    for idx in range(len(df)):
        lat = float(lats[idx])
        lon = float(lons[idx])
        traffic = float(traffics[idx])
        is_obstacle = bool(is_obstacles[idx])

        color = "#ff0000"
        opacity = min(traffic / 150.0, 0.8)
        status_text = "N/A"

        if map_layer_mode == "커버리지 상태 (Status)" and len(status_list) > idx:
            status = int(status_list[idx])

            if status == 1:
                color = "#0000ff"
            elif status == 2:
                color = "#ffa500"
            else:
                color = "#ff0000"

            opacity = min(traffic / 150.0 + 0.2, 0.9)
            status_text = {0: "Uncovered", 1: "Covered", 2: "Overloaded"}.get(status, "N/A")

            if is_obstacle:
                status_text = f"{status_text} (Obstacle)"

        min_lat, max_lat = lat - lat_step / 2, lat + lat_step / 2
        min_lon, max_lon = lon - lon_step / 2, lon + lon_step / 2

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]],
                },
                "properties": {
                    "traffic": round(traffic, 2),
                    "is_obstacle": is_obstacle,
                    "obstacle": "Obstacle" if is_obstacle else "Open",
                    "status": status_text,
                    "fillColor": color,
                    "fillOpacity": float(opacity),
                    "interactive": bool(interactive),
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def station_status(load: float, capacity: float) -> tuple[str, str]:
    if capacity <= 0:
        return "용량 미설정", "#6b7280"

    usage_pct = (load / capacity) * 100.0

    if usage_pct >= 90:
        return "포화 위험", "#dc2626"
    if usage_pct >= 70:
        return "주의", "#f97316"
    return "정상", "#16a34a"


def build_station_popup(
    station_idx: int,
    lat: float,
    lon: float,
    load: float,
    usage_pct: float,
    capacity: float,
    tx_power: float,
    radius_m: float,
):
    status_text, status_color = station_status(load, capacity)
    remaining = capacity - load if capacity > 0 else 0.0

    return dl.Popup(
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.B(f"Station #{station_idx + 1}", style={"fontSize": "14px"}),
                            html.Span(
                                status_text,
                                style={
                                    "display": "inline-block",
                                    "float": "right",
                                    "background": status_color,
                                    "color": "white",
                                    "borderRadius": "999px",
                                    "padding": "2px 8px",
                                    "fontSize": "11px",
                                    "fontWeight": "700",
                                },
                            ),
                        ],
                        style={"marginBottom": "6px"},
                    ),

                    html.Div(f"Lat: {lat:.6f}", style={"color": "#555"}),
                    html.Div(f"Lon: {lon:.6f}", style={"color": "#555"}),

                    html.Hr(style={"margin": "8px 0"}),

                    html.Div(f"Load: {load:.1f}"),
                    html.Div(f"Capacity: {capacity:.1f}"),
                    html.Div(f"Usage: {usage_pct:.1f}%"),
                    html.Div(f"Remaining: {remaining:.1f}"),
                    html.Div(f"Tx Power: {tx_power:.1f} dBm"),
                    html.Div(f"예상 커버 반경: {radius_m:.0f} m"),

                    html.Hr(style={"margin": "8px 0"}),

                    html.Label("Capacity", style={"display": "block", "fontWeight": "700"}),
                    dcc.Input(
                        id={"type": "station-capacity-input", "index": station_idx},
                        type="number",
                        min=1,
                        max=1_000_000_000,
                        step=100,
                        value=float(capacity),
                        debounce=True,
                        style={
                            "width": "100%",
                            "boxSizing": "border-box",
                            "marginTop": "2px",
                            "marginBottom": "6px",
                        },
                    ),

                    html.Label("Tx Power (dBm)", style={"display": "block", "fontWeight": "700"}),
                    dcc.Input(
                        id={"type": "station-tx-input", "index": station_idx},
                        type="number",
                        min=10,
                        max=60,
                        step=1,
                        value=float(tx_power),
                        debounce=True,
                        style={
                            "width": "100%",
                            "boxSizing": "border-box",
                            "marginTop": "2px",
                        },
                    ),

                    html.Button(
                        "적용 후 지도 갱신",
                        id={"type": "station-apply", "index": station_idx},
                        n_clicks=0,
                        style={
                            "marginTop": "10px",
                            "width": "100%",
                            "padding": "6px",
                            "cursor": "pointer",
                            "background": "#2563eb",
                            "color": "white",
                            "border": "0",
                            "borderRadius": "4px",
                            "fontWeight": "700",
                        },
                    ),
                ],
                style={
                    "minWidth": "240px",
                    "fontFamily": "sans-serif",
                    "fontSize": "12px",
                    "lineHeight": "1.45",
                },
            )
        ],
        maxWidth=320,
    )


def build_station_layers(
    opt_results: dict[str, Any],
    opt_stats: dict[str, Any],
    station_specs: list[dict[str, Any]] | None,
    selected_station_idx: int | None,
    overlay_loads: np.ndarray,
) -> list[Any]:
    stations = pd.DataFrame(opt_results.get("stations_geo", []))

    if stations.empty or not {"lat", "lon"}.issubset(stations.columns):
        return []

    capacity_default = float(opt_stats.get("capacity_default", 1000))
    capacities = coerce_station_capacity_array(station_specs, len(stations), capacity_default)

    prop = opt_results.get("prop_params", {})
    fallback_tx = float(np.asarray(prop.get("tx_power_dbm", [43.0]), dtype=float).ravel()[0])
    tx = coerce_station_tx_power_array(station_specs, len(stations), fallback_tx)

    prop_for_radius = {
        "path_loss_ref_db": float(prop.get("path_loss_ref_db", 38.0)),
        "noise_floor_dbm": float(prop.get("noise_floor_dbm", -97.0)),
        "sinr_threshold_db": float(prop.get("sinr_threshold_db", 3.0)),
        "path_loss_exponent": float(prop.get("path_loss_exponent", 3.5)),
        "bandwidth_mhz": float(prop.get("bandwidth_mhz", 10.0)),
    }
    radii = radius_from_tx(tx, prop_for_radius)

    st_lats = stations["lat"].to_numpy()
    st_lons = stations["lon"].to_numpy()
    layers = []

    for i in range(len(stations)):
        lat = float(st_lats[i])
        lon = float(st_lons[i])

        load = float(overlay_loads[i]) if i < len(overlay_loads) else 0.0
        cap = float(capacities[i]) if i < len(capacities) else capacity_default
        usage_pct = (load / cap) * 100.0 if cap > 0 else 0.0

        color = "green"
        if usage_pct > 90:
            color = "red"
        elif usage_pct > 70:
            color = "orange"

        circle_color = "yellow" if selected_station_idx == i else color
        weight = 3 if selected_station_idx == i else 1
        radius_m = float(radii[i]) if i < len(radii) else 300.0
        tx_i = float(tx[i]) if i < len(tx) else fallback_tx

        # 중요:
        # 커버 반경 Circle은 넓은 면적을 차지하므로 클릭 이벤트를 가로채지 않게 interactive=False.
        layers.append(
            dl.Circle(
                center=[lat, lon],
                radius=radius_m,
                color=circle_color,
                weight=weight,
                fill=True,
                fillOpacity=0.18 if selected_station_idx == i else 0.1,
                interactive=False,
            )
        )

        # 중요:
        # Station 마커는 반드시 interactive=True.
        # n_clicks=0을 명시해 dash-leaflet이 클릭 가능한 레이어로 인식하도록 한다.
        station_marker = (
            dl.Marker(
                id={"type": "station-marker", "index": int(i)},
                position=[lat, lon],
                interactive=True,
                n_clicks=0,
                bubblingMouseEvents=False,
                children=[
                    dl.Tooltip(
                        f"Station #{i + 1}"
                        + (" (선택됨)" if selected_station_idx == i else "")
                    ),
                    build_station_popup(
                        station_idx=int(i),
                        lat=lat,
                        lon=lon,
                        load=load,
                        usage_pct=usage_pct,
                        capacity=cap,
                        tx_power=tx_i,
                        radius_m=radius_m,
                    ),
                ],
            )
            if STATION_PIN_MARKER_ENABLED
            else dl.CircleMarker(
                id={"type": "station-marker", "index": int(i)},
                center=[lat, lon],
                radius=13,
                color=color,
                weight=4,
                fill=True,
                fillColor=color,
                fillOpacity=0.95,
                interactive=True,
                n_clicks=0,
                bubblingMouseEvents=False,
                children=[
                    dl.Tooltip(
                        f"Station #{i + 1}"
                        + (" (선택됨)" if selected_station_idx == i else "")
                    ),
                    build_station_popup(
                        station_idx=int(i),
                        lat=lat,
                        lon=lon,
                        load=load,
                        usage_pct=usage_pct,
                        capacity=cap,
                        tx_power=tx_i,
                        radius_m=radius_m,
                    ),
                ],
            )
        )

        layers.append(station_marker)

    return layers


def build_candidate_layers(env: SyntheticEnvironment) -> list[Any]:
    candidate_points = env.local_points_to_geo(env.station_candidate_points)
    layers = []

    for idx, (lat, lon) in enumerate(candidate_points):
        layers.append(
            dl.CircleMarker(
                center=[float(lat), float(lon)],
                radius=5,
                color="blue",
                fill=True,
                fillColor="blue",
                fillOpacity=0.7,
                interactive=True,
                children=[dl.Tooltip(f"Station Candidate #{idx + 1}")],
            )
        )

    return layers


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def metric_card(title: str, value: str):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "12px", "color": "#666"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": "700"}),
        ],
        style={
            "background": "#f8f9fa",
            "border": "1px solid #e9ecef",
            "borderRadius": "8px",
            "padding": "12px",
            "minWidth": "130px",
        },
    )


def sidebar_layout():
    available_algos = [cls.name for cls in REGISTRY]
    default_algo = available_algos[0] if available_algos else ""

    def _disclosure(title: str, children: list[Any], open: bool = False):
        return html.Details(
            [html.Summary(html.H3(title, style={"display": "inline", "margin": 0})), *children],
            open=open,
            style={"marginBottom": "12px"},
        )

    return html.Aside(
        [
            html.H2("시뮬레이터 제어", style={"marginTop": 0}),

            _disclosure(
                "1. 환경 설정",
                [
                    html.Label("격자 크기 (m)"),
                    dcc.Input(
                        id="resolution-m",
                        type="number",
                        min=50,
                        max=500,
                        step=10,
                        value=100,
                        style={"width": "100%"},
                    ),

                    html.Details(
                        [
                            html.Summary("트래픽 세부 설정"),

                            html.Label("트래픽 패턴"),
                            dcc.Dropdown(
                                id="traffic-pattern",
                                options=[{"label": x, "value": x} for x in PATTERN_CHOICES],
                                value=PATTERN_CHOICES[0],
                            ),

                            html.Label("기초 트래픽량 (base)"),
                            dcc.Slider(
                                id="base-intensity",
                                min=0,
                                max=50,
                                step=1,
                                value=10,
                                tooltip={"placement": "bottom"},
                            ),

                            html.Label("최대 트래픽량 (max)"),
                            dcc.Slider(
                                id="max-intensity",
                                min=50,
                                max=500,
                                step=10,
                                value=100,
                                tooltip={"placement": "bottom"},
                            ),

                            html.Div(
                                [
                                    html.Label("핫스팟 개수"),
                                    dcc.Slider(
                                        id="num-hotspots",
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=5,
                                        tooltip={"placement": "bottom"},
                                    ),

                                    html.Label("핫스팟 확산 반경 (m)"),
                                    dcc.Slider(
                                        id="spread-m",
                                        min=100,
                                        max=1000,
                                        step=50,
                                        value=300,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                id="multi-hotspot-controls",
                            ),

                            dcc.Checklist(
                                id="dynamic-traffic",
                                options=[{"label": "동적 트래픽 생성", "value": "on"}],
                                value=[],
                                style={"marginTop": "8px"},
                            ),

                            html.Div(
                                [
                                    html.Label("시간 단계 수"),
                                    dcc.Slider(
                                        id="dynamic-time-steps",
                                        min=2,
                                        max=48,
                                        step=1,
                                        value=12,
                                        tooltip={"placement": "bottom"},
                                    ),

                                    html.Label("시간 변화 강도"),
                                    dcc.Slider(
                                        id="dynamic-variation",
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=0.25,
                                        tooltip={"placement": "bottom"},
                                    ),

                                    html.Label("공간 이동 범위 (m)"),
                                    dcc.Slider(
                                        id="dynamic-drift-m",
                                        min=0,
                                        max=2000,
                                        step=50,
                                        value=300,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                id="dynamic-traffic-controls",
                            ),
                        ],
                        open=True,
                    ),

                    html.Details(
                        [
                            html.Summary("오브젝트 세부 설정"),

                            html.Label("오브젝트 사용 방식"),
                            dcc.RadioItems(
                                id="osm-object-mode",
                                options=[{"label": x, "value": x} for x in OSM_OBJECT_USAGE_MODES],
                                value=OSM_OBJECT_USAGE_MODES[0],
                                inline=True,
                            ),

                            html.Label("오브젝트 소스"),
                            dcc.Dropdown(
                                id="obstacle-source",
                                options=[
                                    {"label": x, "value": x} for x in ["합성", "OSM 지도 데이터", "GeoJSON 업로드"]
                                ],
                                value="합성",
                            ),

                            html.Div(
                                [
                                    html.Label("오브젝트 생성 패턴"),
                                    dcc.Dropdown(
                                        id="obstacle-pattern",
                                        options=[
                                            {"label": x, "value": x}
                                            for x in ["mixed", "random", "circle", "strip", "grid"]
                                        ],
                                        value="mixed",
                                    ),

                                    html.Label("오브젝트 개수"),
                                    dcc.Slider(
                                        id="num-obstacles",
                                        min=0,
                                        max=10,
                                        step=1,
                                        value=3,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                id="synthetic-obstacle-controls",
                            ),

                            html.Div(
                                [
                                    html.Label("OSM 오브젝트 타입"),
                                    dcc.Checklist(
                                        id="osm-types",
                                        options=[{"label": x, "value": x} for x in OSM_OBSTACLE_TYPE_LABELS],
                                        value=OSM_OBSTACLE_TYPE_LABELS,
                                    ),
                                ],
                                id="osm-obstacle-controls",
                            ),

                            html.Div(
                                [
                                    dcc.Upload(
                                        id="geojson-upload",
                                        children=html.Div(["GeoJSON 파일을 드래그하거나 클릭해서 업로드"]),
                                        style={
                                            "border": "1px dashed #aaa",
                                            "borderRadius": "6px",
                                            "padding": "12px",
                                            "textAlign": "center",
                                            "fontSize": "13px",
                                        },
                                        multiple=False,
                                    ),

                                    html.Div(
                                        [
                                            html.Label("최소 오브젝트 면적 (m²)"),
                                            dcc.Slider(
                                                id="min-obstacle-area-m2",
                                                min=0,
                                                max=5000,
                                                step=100,
                                                value=100,
                                                tooltip={"placement": "bottom"},
                                            ),

                                            html.Label("최대 오브젝트 개수"),
                                            dcc.Slider(
                                                id="max-map-obstacles",
                                                min=1,
                                                max=500,
                                                step=10,
                                                value=100,
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        id="geojson-filter-controls",
                                    ),
                                ],
                                id="geojson-obstacle-controls",
                            ),
                        ],
                        open=True,
                    ),

                    html.Div(
                        id="custom-region-info",
                        style={
                            "display": "none",
                            "fontSize": "12px",
                            "marginTop": "8px",
                            "padding": "6px 8px",
                            "background": "#f0fdf4",
                            "border": "1px solid #86efac",
                            "borderRadius": "4px",
                            "color": "#166534",
                        },
                    ),
                    html.Button(
                        "영역 지우기",
                        id="clear-region-btn",
                        n_clicks=0,
                        style={
                            "display": "none",
                            "width": "100%",
                            "padding": "6px 12px",
                            "marginTop": "4px",
                            "cursor": "pointer",
                            "background": "#dc2626",
                            "color": "white",
                            "border": "0",
                            "borderRadius": "6px",
                            "fontSize": "12px",
                            "fontWeight": "600",
                        },
                    ),

                    html.Button(
                        "가상 데이터 생성",
                        id="create-env-btn",
                        n_clicks=0,
                        className="primary-button",
                    ),
                    html.Div(id="create-status", style={"fontSize": "13px", "marginTop": "8px"}),
                ],
                open=True,
            ),

            _disclosure(
                "2. 시각화 설정",
                [
                    dcc.RadioItems(
                        id="map-layer-mode",
                        options=[
                            {"label": "트래픽 분포 (Traffic)", "value": "트래픽 분포 (Traffic)"},
                            {"label": "커버리지 상태 (Status)", "value": "커버리지 상태 (Status)"},
                        ],
                        value="커버리지 상태 (Status)",
                    ),
                ],
                open=True,
            ),

            _disclosure(
                "3. 계산 알고리즘",
                [
                    html.Label("알고리즘 선택"),
                    dcc.Dropdown(
                        id="algo-select",
                        options=[{"label": x, "value": x} for x in available_algos],
                        value=default_algo,
                    ),
                    html.Div(id="hyperparam-controls"),

                    html.Label("기지국 개수 설정"),
                    dcc.RadioItems(
                        id="opt-mode",
                        options=[
                            {"label": "고정 개수 (Fixed)", "value": "고정 개수 (Fixed)"},
                            {"label": "범위 탐색 (Range)", "value": "범위 탐색 (Range)"},
                        ],
                        value="고정 개수 (Fixed)",
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("기지국 수"),
                                    dcc.Slider(
                                        id="n-stations",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=5,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                id="fixed-count-controls",
                            ),

                            html.Div(
                                [
                                    html.Label("최소 개수"),
                                    dcc.Input(
                                        id="k-min",
                                        type="number",
                                        min=1,
                                        max=100,
                                        step=1,
                                        value=3,
                                        style={"width": "48%", "marginRight": "4%"},
                                    ),

                                    html.Label("최대 개수"),
                                    dcc.Input(
                                        id="k-max",
                                        type="number",
                                        min=1,
                                        max=200,
                                        step=1,
                                        value=10,
                                        style={"width": "48%"},
                                    ),
                                ],
                                id="range-count-controls",
                            ),
                        ],
                        id="station-count-controls",
                    ),
                ],
                open=True,
            ),

            _disclosure(
                "기지국 스펙",
                [
                    dcc.RadioItems(
                        id="spec-mode",
                        options=[
                            {"label": "전체 동일", "value": "전체 동일"},
                            {"label": "기지국별 개별", "value": "기지국별 개별"},
                        ],
                        value="전체 동일",
                        inline=True,
                    ),

                    html.Label("기본 용량 (Traffic)"),
                    dcc.Input(
                        id="capacity-default",
                        type="number",
                        min=500,
                        max=1_000_000_000,
                        step=100,
                        value=2000,
                        style={"width": "100%"},
                    ),

                    html.Div(
                        [
                            dash_table.DataTable(
                                id="station-spec-table",
                                columns=[
                                    {"name": "station", "id": "station", "type": "numeric", "editable": False},
                                    {"name": "radius_m", "id": "radius_m", "type": "numeric", "editable": False},
                                    {"name": "capacity", "id": "capacity", "type": "numeric", "editable": True},
                                    {"name": "tx_power_dbm", "id": "tx_power_dbm", "type": "numeric", "editable": True},
                                ],
                                data=ensure_station_spec_rows([], 5, 300.0, 2000.0, 43.0),
                                editable=True,
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_cell={"fontSize": "12px", "padding": "4px"},
                            )
                        ],
                        id="spec-table-wrap",
                        style={"marginTop": "8px", "display": "none"},
                    ),
                ],
                open=True,
            ),

            _disclosure(
                "전파 모델",
                [
                    html.Label("송신 전력 (dBm)"),
                    dcc.Slider(
                        id="ui-tx-power",
                        min=20,
                        max=50,
                        step=1,
                        value=43,
                        tooltip={"placement": "bottom"},
                    ),

                    html.Label("경로 손실 지수 n"),
                    dcc.Slider(
                        id="ui-path-loss-exp",
                        min=2.0,
                        max=5.0,
                        step=0.1,
                        value=3.5,
                        tooltip={"placement": "bottom"},
                    ),

                    html.Label("대역폭 (MHz)"),
                    dcc.Slider(
                        id="ui-bandwidth-mhz",
                        min=1,
                        max=100,
                        step=1,
                        value=10,
                        tooltip={"placement": "bottom"},
                    ),

                    html.Label("SINR 임계값 (dB)"),
                    dcc.Slider(
                        id="ui-sinr-threshold",
                        min=-10,
                        max=30,
                        step=1,
                        value=3,
                        tooltip={"placement": "bottom"},
                    ),

                    html.Div(
                        id="noise-caption",
                        style={"fontSize": "12px", "color": "#555", "marginTop": "4px"},
                    ),

                    dcc.Checklist(
                        id="ui-hetnet",
                        options=[{"label": "HetNet 활성화 — 매크로 + 스몰셀 혼합 배치", "value": "on"}],
                        value=[],
                        style={"marginTop": "8px"},
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("매크로 기지국 수"),
                                    dcc.Input(
                                        id="ui-n-macro",
                                        type="number",
                                        min=0,
                                        max=20,
                                        step=1,
                                        value=2,
                                        style={"width": "48%", "marginRight": "4%"},
                                    ),

                                    html.Label("스몰셀 수"),
                                    dcc.Input(
                                        id="ui-n-small",
                                        type="number",
                                        min=0,
                                        max=50,
                                        step=1,
                                        value=3,
                                        style={"width": "48%"},
                                    ),
                                ]
                            ),
                            
                            html.Label("매크로 전력 (dBm)"),
                            dcc.Slider(
                                id="ui-macro-power",
                                min=30,
                                max=50,
                                step=1,
                                value=43,
                                tooltip={"placement": "bottom"},
                            ),

                            html.Label("스몰셀 전력 (dBm)"),
                            dcc.Slider(
                                id="ui-small-power",
                                min=20,
                                max=40,
                                step=1,
                                value=30,
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        id="hetnet-controls",
                    ),

                    html.Button(
                        "계산 실행",
                        id="optimize-btn",
                        n_clicks=0,
                        className="primary-button",
                    ),
                ],
                open=True,
            ),

            _disclosure(
                "데이터 내보내기",
                [
                    html.Button("GIS CSV", id="download-gis-btn", n_clicks=0),
                    html.Button(
                        "Local CSV",
                        id="download-local-btn",
                        n_clicks=0,
                        style={"marginLeft": "6px"},
                    ),
                ],
                open=True,
            ),
        ],
        style={
            "width": "380px",
            "minWidth": "380px",
            "height": "100vh",
            "overflowY": "auto",
            "padding": "16px",
            "background": "#ffffff",
            "borderRight": "1px solid #e5e7eb",
            "boxSizing": "border-box",
        },
    )
def serve_layout():
    session_id = str(uuid.uuid4())

    return html.Div(
        [
            dcc.Store(id="session-id", data=session_id),
            dcc.Store(id="env-meta"),
            dcc.Store(id="opt-meta"),
            dcc.Store(id="range-meta"),
            dcc.Store(id="selected-station", data=None),
            dcc.Store(id="drawn-region-store", data=None),
            dcc.Store(id="custom-region-store", data=None),
            dcc.Store(id="editcontrol-clear-count", data=0),
            dcc.Store(id="algo-history-store", data=None),
            dcc.Store(id="opt-live-store", data=None),
            dcc.Interval(id="opt-poll-interval", interval=750, disabled=True, n_intervals=0),

            dcc.Download(id="download-gis-csv"),
            dcc.Download(id="download-local-csv"),

            html.Div(
                [
                    sidebar_layout(),

                    html.Main(
                        [
                            html.H1("기지국 위치 최적화 시뮬레이터", style={"marginTop": 0}),

                            html.Div(
                                id="stats-panel",
                                style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
                            ),

                            html.Div(id="run-status", style={"margin": "12px 0"}),

                            html.Div(
                                [
                                    html.Div(
                                        id="traffic-frame-label",
                                        style={"fontWeight": "700", "marginBottom": "4px"},
                                    ),

                                    dcc.Slider(
                                        id="traffic-frame-slider",
                                        min=0,
                                        max=1,
                                        step=1,
                                        value=0,
                                        marks=None,
                                    ),

                                    html.Button(
                                        "지도 재생",
                                        id="traffic-play-btn",
                                        n_clicks=0,
                                        style={"marginRight": "6px"},
                                    ),

                                    dcc.Interval(
                                        id="traffic-frame-interval",
                                        interval=500,
                                        disabled=True,
                                        n_intervals=0,
                                    ),
                                ],
                                id="dynamic-frame-wrap",
                                style={"display": "none", "marginBottom": "12px"},
                            ),

                            html.Div(
                                [
                                    html.Div(
                                        id="algo-history-label",
                                        style={"fontWeight": "700", "marginBottom": "4px"},
                                    ),
                                    dcc.Slider(
                                        id="algo-history-slider",
                                        min=0,
                                        max=1,
                                        step=1,
                                        value=0,
                                        marks=None,
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "▶ 재생",
                                                id="algo-play-btn",
                                                n_clicks=0,
                                                style={"marginRight": "6px"},
                                            ),
                                            html.Button(
                                                "⏮ 처음",
                                                id="algo-reset-btn",
                                                n_clicks=0,
                                            ),
                                        ],
                                        style={"marginTop": "4px", "marginBottom": "8px"},
                                    ),
                                    dcc.Interval(
                                        id="algo-frame-interval",
                                        interval=300,
                                        disabled=True,
                                        n_intervals=0,
                                    ),
                                    dcc.Graph(
                                        id="algo-history-chart",
                                        style={"height": "150px"},
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                id="algo-history-wrap",
                                style={"display": "none", "marginBottom": "12px",
                                       "background": "#fff", "padding": "10px",
                                       "borderRadius": "8px"},
                            ),

                            dl.Map(
                                id="sim-map",
                                center=DEFAULT_CENTER,
                                zoom=DEFAULT_ZOOM,
                                bounds=None,
                                children=[
                                    dl.TileLayer(
                                        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                                        attribution="&copy; OpenStreetMap contributors &copy; CARTO",
                                    ),
                                    dl.LayerGroup(id="overlay-layers", children=[]),
                                    dl.FeatureGroup(
                                        id="region-draw-feature-group",
                                        children=[
                                            dl.EditControl(
                                                id="region-edit-control",
                                                draw={
                                                    "rectangle": {},
                                                    "polyline": False,
                                                    "polygon": False,
                                                    "circle": False,
                                                    "marker": False,
                                                    "circlemarker": False,
                                                },
                                                edit={"edit": False},
                                                position="topleft",
                                            )
                                        ],
                                    ),
                                ],
                                style={
                                    "width": "100%",
                                    "height": "720px",
                                    "borderRadius": "8px",
                                },
                            ),

                            html.Div(id="range-panel", style={"marginTop": "20px"}),
                        ],
                        style={"flex": "1", "padding": "18px", "minWidth": 0},
                    ),
                ],
                style={"display": "flex", "height": "100vh", "overflow": "hidden"},
            ),

            # Region selection popup overlay
            html.Div(
                id="region-popup",
                children=[
                    html.Div(
                        [
                            html.H4("트래픽 영역 설정", style={"margin": "0 0 16px 0", "color": "#111827"}),

                            html.Div(
                                [
                                    html.Label("너비 (km)", style={"fontWeight": "600", "fontSize": "13px"}),
                                    dcc.Input(
                                        id="region-width-km",
                                        type="number",
                                        min=0.1,
                                        max=200,
                                        step=0.1,
                                        style={"width": "100%", "padding": "6px", "borderRadius": "4px", "border": "1px solid #d1d5db"},
                                    ),
                                ],
                                style={"marginBottom": "12px"},
                            ),

                            html.Div(
                                [
                                    html.Label("높이 (km)", style={"fontWeight": "600", "fontSize": "13px"}),
                                    dcc.Input(
                                        id="region-height-km",
                                        type="number",
                                        min=0.1,
                                        max=200,
                                        step=0.1,
                                        style={"width": "100%", "padding": "6px", "borderRadius": "4px", "border": "1px solid #d1d5db"},
                                    ),
                                ],
                                style={"marginBottom": "20px"},
                            ),

                            html.Div(
                                [
                                    html.Button(
                                        "확인",
                                        id="region-confirm-btn",
                                        n_clicks=0,
                                        style={
                                            "padding": "8px 24px",
                                            "marginRight": "8px",
                                            "background": "#2563eb",
                                            "color": "white",
                                            "border": "0",
                                            "borderRadius": "6px",
                                            "cursor": "pointer",
                                            "fontWeight": "700",
                                        },
                                    ),
                                    html.Button(
                                        "취소",
                                        id="region-cancel-btn",
                                        n_clicks=0,
                                        style={
                                            "padding": "8px 24px",
                                            "background": "#6b7280",
                                            "color": "white",
                                            "border": "0",
                                            "borderRadius": "6px",
                                            "cursor": "pointer",
                                            "fontWeight": "700",
                                        },
                                    ),
                                ],
                            ),
                        ],
                        style={
                            "background": "white",
                            "padding": "24px",
                            "borderRadius": "10px",
                            "boxShadow": "0 8px 32px rgba(0,0,0,0.25)",
                            "minWidth": "280px",
                        },
                    ),
                ],
                style={
                    "display": "none",
                    "position": "fixed",
                    "top": 0,
                    "left": 0,
                    "width": "100vw",
                    "height": "100vh",
                    "background": "rgba(0,0,0,0.45)",
                    "zIndex": 10000,
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
        ]
    )


# ---------------------------------------------------------------------------
# App / callbacks
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.layout = serve_layout

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Base Station Simulator</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #f3f4f6;
            }
            label {
                display:block;
                margin-top: 10px;
                margin-bottom: 4px;
                font-weight: 600;
                font-size: 13px;
            }
            summary {
                cursor: pointer;
                font-weight: 700;
                margin: 10px 0;
            }
            .primary-button {
                width: 100%;
                padding: 10px 12px;
                margin-top: 12px;
                cursor: pointer;
                background: #2563eb;
                color: white;
                border: 0;
                border-radius: 6px;
                font-weight: 700;
            }
            button {
                cursor: pointer;
            }
            .leaflet-interactive {
                cursor: pointer;
            }
            /* delete-layers button is kept enabled for programmatic clear but hidden from UI */
            a.leaflet-draw-edit-remove {
                display: none !important;
            }
            .primary-button:disabled {
                background: #9ca3af !important;
                cursor: not-allowed;
                opacity: 0.7;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


@app.callback(
    Output("multi-hotspot-controls", "style"),
    Input("traffic-pattern", "value"),
)
def toggle_multi_hotspot_controls(pattern):
    return {"display": "block" if pattern == "multi_hotspot" else "none"}


@app.callback(
    Output("dynamic-traffic-controls", "style"),
    Input("dynamic-traffic", "value"),
)
def toggle_dynamic_controls(dynamic_value):
    return {"display": "block" if normalize_triggered_bool(dynamic_value) else "none"}


@app.callback(
    Output("synthetic-obstacle-controls", "style"),
    Output("osm-obstacle-controls", "style"),
    Output("geojson-obstacle-controls", "style"),
    Output("geojson-filter-controls", "style"),
    Input("obstacle-source", "value"),
    Input("osm-object-mode", "value"),
)
def toggle_obstacle_source_controls(source, object_mode):
    return (
        {"display": "block" if source == "합성" else "none"},
        {"display": "block" if source == "OSM 지도 데이터" else "none"},
        {"display": "block" if source == "GeoJSON 업로드" else "none"},
        {"display": "block" if source == "GeoJSON 업로드" and object_mode == "장애물로 사용" else "none"},
    )


@app.callback(
    Output("fixed-count-controls", "style"),
    Output("range-count-controls", "style"),
    Input("opt-mode", "value"),
)
def toggle_station_count_controls(opt_mode):
    return (
        {"display": "block" if opt_mode == "고정 개수 (Fixed)" else "none"},
        {"display": "block" if opt_mode == "범위 탐색 (Range)" else "none"},
    )


@app.callback(
    Output("hyperparam-controls", "children"),
    Input("algo-select", "value"),
)
def render_hyperparam_controls(algo):
    if not algo:
        return []

    optimizer = get_optimizer(algo)
    controls = []

    for p in optimizer.hyperparams or []:
        label = p.label or p.name
        component_id = {"type": "hyperparam", "name": p.name, "kind": p.kind}

        controls.append(html.Label(label))

        if p.kind == "int":
            if p.min is not None and p.max is not None and p.step is not None:
                controls.append(
                    dcc.Slider(
                        id=component_id,
                        min=int(p.min),
                        max=int(p.max),
                        step=int(p.step),
                        value=int(p.default),
                        tooltip={"placement": "bottom"},
                    )
                )
            else:
                controls.append(
                    dcc.Input(
                        id=component_id,
                        type="number",
                        value=int(p.default),
                        style={"width": "100%"},
                    )
                )

        elif p.kind == "float":
            if p.min is not None and p.max is not None:
                controls.append(
                    dcc.Slider(
                        id=component_id,
                        min=float(p.min),
                        max=float(p.max),
                        step=float(p.step if p.step is not None else 0.01),
                        value=float(p.default),
                        tooltip={"placement": "bottom"},
                    )
                )
            else:
                controls.append(
                    dcc.Input(
                        id=component_id,
                        type="number",
                        value=float(p.default),
                        style={"width": "100%"},
                    )
                )

        elif p.kind == "choice":
            controls.append(
                dcc.Dropdown(
                    id=component_id,
                    options=[{"label": str(x), "value": x} for x in p.choices],
                    value=p.default,
                )
            )

        elif p.kind == "bool":
            controls.append(
                dcc.Checklist(
                    id=component_id,
                    options=[{"label": "사용", "value": "on"}],
                    value=["on"] if bool(p.default) else [],
                )
            )

    return controls


@app.callback(
    Output("noise-caption", "children"),
    Input("ui-tx-power", "value"),
    Input("ui-path-loss-exp", "value"),
    Input("ui-bandwidth-mhz", "value"),
    Input("ui-sinr-threshold", "value"),
)
def update_noise_caption(tx_power, path_loss_exp, bandwidth_mhz, sinr_threshold):
    prop = prop_params_base(
        float(path_loss_exp),
        float(bandwidth_mhz),
        float(sinr_threshold),
    )
    r_eff = radius_from_tx(np.asarray([float(tx_power)], dtype=float), prop)[0]

    return (
        f"잡음 바닥: {prop['noise_floor_dbm']:.1f} dBm "
        f"| 단일 기지국 예상 커버 반경: {r_eff:.0f} m"
    )


@app.callback(
    Output("hetnet-controls", "style"),
    Input("ui-hetnet", "value"),
)
def toggle_hetnet_controls(hetnet_value):
    return {"display": "block" if normalize_triggered_bool(hetnet_value) else "none"}


@app.callback(
    Output("station-spec-table", "data"),
    Output("spec-table-wrap", "style"),
    Input("opt-mode", "value"),
    Input("spec-mode", "value"),
    Input("capacity-default", "value"),
    Input("ui-tx-power", "value"),
    Input("ui-hetnet", "value"),
    Input("n-stations", "value"),
    Input("k-max", "value"),
    Input("ui-n-macro", "value"),
    Input("ui-n-small", "value"),
    State("station-spec-table", "data"),
)
def refresh_station_spec_table(
    opt_mode,
    spec_mode,
    capacity_default,
    ui_tx_power,
    hetnet_value,
    n_stations,
    k_max,
    ui_n_macro,
    ui_n_small,
    existing_rows,
):
    hetnet_enabled = normalize_triggered_bool(hetnet_value)

    if opt_mode == "범위 탐색 (Range)":
        target_count = safe_int(k_max, 10)
    elif hetnet_enabled:
        target_count = safe_int(ui_n_macro, 0) + safe_int(ui_n_small, 0)
        target_count = max(target_count, 1)
    else:
        target_count = safe_int(n_stations, 5)

    default_tx = safe_float(ui_tx_power, 43.0)

    rows = ensure_station_spec_rows(
        existing_rows,
        target_count,
        default_radius=300.0,
        default_capacity=safe_float(capacity_default, 2000.0),
        default_tx_power=default_tx,
    )

    style = {
        "marginTop": "8px",
        "display": "block" if spec_mode == "기지국별 개별" else "none",
    }

    return rows, style


@app.callback(
    Output("env-meta", "data"),
    Output("opt-meta", "data"),
    Output("range-meta", "data"),
    Output("create-status", "children"),
    Input("create-env-btn", "n_clicks"),
    State("session-id", "data"),
    State("sim-map", "bounds"),
    State("sim-map", "center"),
    State("sim-map", "zoom"),
    State("resolution-m", "value"),
    State("traffic-pattern", "value"),
    State("base-intensity", "value"),
    State("max-intensity", "value"),
    State("dynamic-traffic", "value"),
    State("num-hotspots", "value"),
    State("spread-m", "value"),
    State("dynamic-time-steps", "value"),
    State("dynamic-variation", "value"),
    State("dynamic-drift-m", "value"),
    State("osm-object-mode", "value"),
    State("obstacle-source", "value"),
    State("obstacle-pattern", "value"),
    State("num-obstacles", "value"),
    State("osm-types", "value"),
    State("geojson-upload", "contents"),
    State("min-obstacle-area-m2", "value"),
    State("max-map-obstacles", "value"),
    State("custom-region-store", "data"),
    prevent_initial_call=True,
)
def create_environment(
    n_clicks,
    session_id,
    bounds,
    center,
    zoom,
    resolution_m,
    traffic_pattern,
    base_intensity,
    max_intensity,
    dynamic_traffic,
    num_hotspots,
    spread_m,
    dynamic_time_steps,
    dynamic_variation,
    dynamic_drift_m,
    osm_object_mode,
    obstacle_source,
    obstacle_pattern,
    num_obstacles,
    osm_types,
    geojson_contents,
    min_obstacle_area_m2,
    max_map_obstacles,
    custom_region,
):
    if not n_clicks:
        raise PreventUpdate

    state = get_session_state(session_id)

    try:
        # Custom region takes priority over map viewport bounds
        if isinstance(custom_region, dict) and custom_region.get("width_km") and custom_region.get("height_km"):
            center_lat = float(custom_region["center_lat"])
            center_lon = float(custom_region["center_lon"])
            width_km = max(float(custom_region["width_km"]), 0.1)
            height_km = max(float(custom_region["height_km"]), 0.1)
        else:
            parsed_bounds = parse_map_bounds(bounds)
            center_lat, center_lon = parse_map_center(center)

            if parsed_bounds is not None:
                sw, ne = parsed_bounds
                width_km = geodesic((sw[0], sw[1]), (sw[0], ne[1])).km
                height_km = geodesic((sw[0], sw[1]), (ne[0], sw[1])).km
                width_km = max(width_km, 0.1)
                height_km = max(height_km, 0.1)
            else:
                width_km = 2.0
                height_km = 2.0

        env = SyntheticEnvironment(
            center_lat=center_lat,
            center_lon=center_lon,
            width_km=width_km,
            height_km=height_km,
            resolution_m=safe_float(resolution_m, 100.0),
        )

        is_dynamic = normalize_triggered_bool(dynamic_traffic)

        if is_dynamic:
            pattern_params = {}

            if traffic_pattern == "multi_hotspot":
                sigma_cells = max(
                    safe_float(spread_m, 300.0) / max(safe_float(resolution_m, 100.0), 1.0),
                    1.0,
                )
                pattern_params = {
                    "n_centers": safe_int(num_hotspots, 5),
                    "sigma_x": sigma_cells,
                    "sigma_y": sigma_cells,
                }

            env.generate_dynamic_traffic_pattern(
                pattern=traffic_pattern,
                time_steps=safe_int(dynamic_time_steps, 12),
                max_intensity=safe_float(max_intensity, 100.0),
                base_intensity=safe_float(base_intensity, 10.0),
                variation=safe_float(dynamic_variation, 0.25),
                drift_m=safe_float(dynamic_drift_m, 300.0),
                params=pattern_params,
            )

        elif traffic_pattern == "multi_hotspot":
            env.generate_traffic(
                num_hotspots=safe_int(num_hotspots, 5),
                spread_m=safe_float(spread_m, 300.0),
                base_intensity=safe_float(base_intensity, 10.0),
                max_intensity=safe_float(max_intensity, 100.0),
            )

        else:
            env.generate_traffic_pattern(
                pattern=traffic_pattern,
                max_intensity=safe_float(max_intensity, 100.0),
                base_intensity=safe_float(base_intensity, 10.0),
            )

        selected_osm_types = osm_types or []
        osm_obstacle_types: list[str] = []

        for osm_type in selected_osm_types:
            value = OSM_OBSTACLE_TYPE_VALUES[osm_type]
            if isinstance(value, tuple):
                osm_obstacle_types.extend(value)
            else:
                osm_obstacle_types.append(value)

        osm_obstacle_types = list(dict.fromkeys(osm_obstacle_types))
        uploaded_geojson = decode_upload_to_bytes(geojson_contents)

        applied_count, raw_count = apply_obstacle_source(
            env,
            source=obstacle_source or "합성",
            uploaded_geojson=uploaded_geojson,
            min_area_m2=safe_float(min_obstacle_area_m2, 100.0),
            max_obstacles=safe_int(max_map_obstacles, 100) if max_map_obstacles is not None else None,
            obstacle_pattern=obstacle_pattern or "mixed",
            num_obstacles=safe_int(num_obstacles, 3),
            osm_obstacle_types=osm_obstacle_types,
            osm_object_mode=osm_object_mode or OSM_OBJECT_USAGE_MODES[0],
            append=False,
        )

        state["env"] = env
        state.pop("opt_results", None)
        state.pop("opt_stats", None)
        state.pop("range_results", None)
        state.pop("station_overlay_loads", None)

        applied_type = "기지국 후보" if osm_object_mode == "기지국 후보로 사용" else "장애물"

        msg = (
            f"가상 환경 생성 완료 | 영역: {width_km:.2f} km × {height_km:.2f} km | "
            f"{obstacle_source}({applied_type}): 원본 {raw_count}개 중 {applied_count}개 적용"
        )

        return version_token(), None, None, html.Div(msg, style={"color": "#166534"})

    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        return (
            no_update,
            no_update,
            no_update,
            html.Div(
                f"생성 실패: {exc}\n{tb}",
                style={"color": "#b91c1c", "whiteSpace": "pre-wrap"},
            ),
        )


@app.callback(
    Output("dynamic-frame-wrap", "style"),
    Output("traffic-frame-slider", "max"),
    Output("traffic-frame-slider", "value"),
    Output("traffic-frame-label", "children"),
    Input("env-meta", "data"),
    State("session-id", "data"),
)
def refresh_dynamic_frame_controls(env_meta, session_id):
    state = get_session_state(session_id)
    env = state.get("env")
    series = getattr(env, "traffic_series", None) if env is not None else None

    if series is None or getattr(series, "shape", [0])[0] <= 1:
        return {"display": "none"}, 1, 0, ""

    current = int(getattr(env, "dynamic_frame_index", 0))
    max_frame = int(series.shape[0] - 1)
    current = max(0, min(current, max_frame))

    return (
        {
            "display": "block",
            "marginBottom": "12px",
            "background": "#fff",
            "padding": "10px",
            "borderRadius": "8px",
        },
        max_frame,
        current,
        f"동적 트래픽 프레임: {current} / {max_frame}",
    )


@app.callback(
    Output("traffic-frame-interval", "disabled"),
    Output("traffic-play-btn", "children"),
    Input("traffic-play-btn", "n_clicks"),
    State("traffic-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_traffic_playback(n_clicks, disabled):
    next_disabled = not bool(disabled)
    return next_disabled, "지도 재생" if next_disabled else "정지"


@app.callback(
    Output("traffic-frame-slider", "value", allow_duplicate=True),
    Input("traffic-frame-interval", "n_intervals"),
    State("traffic-frame-slider", "value"),
    State("traffic-frame-slider", "max"),
    State("traffic-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def advance_traffic_frame(n_intervals, current_value, max_value, disabled):
    if disabled:
        raise PreventUpdate

    current = safe_int(current_value, 0)
    max_frame = max(0, safe_int(max_value, 0))

    if max_frame <= 0:
        raise PreventUpdate

    return (current + 1) % (max_frame + 1)


@app.callback(
    Output("env-meta", "data", allow_duplicate=True),
    Input("traffic-frame-slider", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def set_dynamic_traffic_frame(frame_idx, session_id):
    state = get_session_state(session_id)
    env = state.get("env")

    if env is None or getattr(env, "traffic_series", None) is None:
        raise PreventUpdate

    max_frame = int(env.traffic_series.shape[0] - 1)
    frame = max(0, min(safe_int(frame_idx, 0), max_frame))
    env.set_traffic_frame(frame)
    state["env"] = env

    return version_token()


@app.callback(
    Output("overlay-layers", "children"),
    Input("env-meta", "data"),
    Input("opt-meta", "data"),
    Input("station-spec-table", "data"),
    Input("selected-station", "data"),
    Input("map-layer-mode", "value"),
    Input("custom-region-store", "data"),
    Input("algo-history-store", "data"),
    Input("algo-history-slider", "value"),
    Input("opt-live-store", "data"),
    State("session-id", "data"),
)
def update_map_layers(
    env_meta,
    opt_meta,
    station_specs,
    selected_station_idx,
    map_layer_mode,
    custom_region,
    algo_history,
    history_frame_idx,
    opt_live,
    session_id,
):
    children: list = []

    # Show confirmed custom region boundary
    if isinstance(custom_region, dict):
        try:
            sw = [custom_region["center_lat"] - custom_region["height_km"] / 2 / 110.574,
                  custom_region["center_lon"] - custom_region["width_km"] / 2 / (111.32 * np.cos(np.radians(custom_region["center_lat"])))]
            ne = [custom_region["center_lat"] + custom_region["height_km"] / 2 / 110.574,
                  custom_region["center_lon"] + custom_region["width_km"] / 2 / (111.32 * np.cos(np.radians(custom_region["center_lat"])))]
            children.append(
                dl.Rectangle(
                    bounds=[sw, ne],
                    pathOptions={"color": "#f59e0b", "weight": 2, "fillOpacity": 0.05, "dashArray": "6 4"},
                )
            )
        except (KeyError, TypeError, ZeroDivisionError):
            pass

    state = get_session_state(session_id)
    env = state.get("env")

    if env is None:
        return children

    opt_results = state.get("opt_results")
    opt_stats = state.get("opt_stats")

    df = env_dataframe_for_current_frame(env)
    status_list, overlay_loads = compute_status_overlay(
        env,
        df,
        opt_results,
        opt_stats,
        station_specs,
    )
    state["station_overlay_loads"] = overlay_loads

    # 최적화 결과가 있을 때는 트래픽 격자 GeoJSON이 Station 클릭 이벤트를 먹지 않도록 비활성화한다.
    traffic_interactive = not bool(opt_results and opt_stats)

    traffic_geojson = build_traffic_geojson(
        env,
        df,
        map_layer_mode,
        status_list,
        interactive=traffic_interactive,
    )

    traffic_options = {
        "style": TRAFFIC_STYLE,
        "interactive": traffic_interactive,
    }

    if traffic_interactive:
        traffic_options["onEachFeature"] = TRAFFIC_ON_EACH_FEATURE

    children.append(
        dl.GeoJSON(
            id="traffic-geojson",
            data=traffic_geojson,
            options=traffic_options,
            interactive=traffic_interactive,
        )
    )

    candidate_layers = build_candidate_layers(env)

    if candidate_layers:
        children.append(
            dl.LayerGroup(
                candidate_layers,
                id="candidate-layer",
            )
        )

    # Live optimization preview (during background thread execution)
    live_progress = state.get("opt_progress", {})
    if live_progress.get("running") and live_progress.get("stations_geo"):
        for lat, lon in live_progress["stations_geo"]:
            children.append(
                dl.CircleMarker(
                    center=[lat, lon],
                    radius=10,
                    pathOptions={
                        "color": "#ea580c",
                        "fillColor": "#f97316",
                        "fillOpacity": 0.85,
                        "weight": 2,
                    },
                )
            )
        return children

    # History replay: show intermediate station positions (orange markers)
    history_active = False
    if isinstance(algo_history, dict) and algo_history.get("frames"):
        frames = algo_history["frames"]
        n_frames = len(frames)
        idx = min(safe_int(history_frame_idx, 0), n_frames - 1)
        if idx < n_frames - 1:
            history_active = True
            frame = frames[idx]
            stations_geo = frame.get("stations_geo", [])

            # Fading trail from previous snapshots
            trail_start = max(0, idx - 4)
            for ti in range(trail_start, idx):
                alpha = 0.15 + 0.15 * (ti - trail_start + 1)
                for lat, lon in frames[ti].get("stations_geo", []):
                    children.append(
                        dl.CircleMarker(
                            center=[lat, lon],
                            radius=5,
                            pathOptions={
                                "color": "#6b7280",
                                "fillColor": "#9ca3af",
                                "fillOpacity": alpha,
                                "weight": 1,
                            },
                        )
                    )

            # Current frame stations (orange)
            for lat, lon in stations_geo:
                children.append(
                    dl.CircleMarker(
                        center=[lat, lon],
                        radius=10,
                        pathOptions={
                            "color": "#ea580c",
                            "fillColor": "#f97316",
                            "fillOpacity": 0.85,
                            "weight": 2,
                        },
                    )
                )

    if not history_active and opt_results and opt_stats:
        children.append(
            dl.LayerGroup(
                build_station_layers(
                    opt_results,
                    opt_stats,
                    station_specs,
                    selected_station_idx if isinstance(selected_station_idx, int) else None,
                    overlay_loads,
                ),
                id="station-layer",
            )
        )

    return children


@app.callback(
    Output("station-spec-table", "data", allow_duplicate=True),
    Output("selected-station", "data", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Input({"type": "station-apply", "index": ALL}, "n_clicks"),
    State({"type": "station-capacity-input", "index": ALL}, "value"),
    State({"type": "station-tx-input", "index": ALL}, "value"),
    State({"type": "station-apply", "index": ALL}, "id"),
    State("station-spec-table", "data"),
    prevent_initial_call=True,
)
def apply_station_popup_edit(n_clicks, capacity_values, tx_values, apply_ids, rows):
    triggered = ctx.triggered_id

    if not triggered or not isinstance(triggered, dict):
        raise PreventUpdate

    if not n_clicks or max([int(x or 0) for x in n_clicks], default=0) <= 0:
        raise PreventUpdate

    station_idx = int(triggered["index"])

    pos = None
    for j, id_obj in enumerate(apply_ids):
        if int(id_obj.get("index")) == station_idx:
            pos = j
            break

    if pos is None:
        raise PreventUpdate

    rows = ensure_station_spec_rows(
        rows,
        max(station_idx + 1, len(rows or [])),
        300.0,
        2000.0,
        43.0,
    )

    old_capacity = rows[station_idx]["capacity"]
    old_tx = rows[station_idx]["tx_power_dbm"]

    rows[station_idx]["capacity"] = safe_float(capacity_values[pos], old_capacity)
    rows[station_idx]["tx_power_dbm"] = safe_float(tx_values[pos], old_tx)

    status = html.Div(
        (
            f"Station #{station_idx + 1} 스펙 적용 완료 | "
            f"Capacity={rows[station_idx]['capacity']:.1f}, "
            f"Tx Power={rows[station_idx]['tx_power_dbm']:.1f} dBm"
        ),
        style={"color": "#166534"},
    )

    return rows, station_idx, status


# ---------------------------------------------------------------------------
# Optimization: helpers + background thread + callbacks
# ---------------------------------------------------------------------------

def _parse_hyperparams(hp_values, hp_ids, hp_defaults: dict) -> dict[str, Any]:
    """UI hyperparam widgets → typed dict."""
    hyperparams: dict[str, Any] = {}
    for value, id_obj in zip(hp_values or [], hp_ids or []):
        name = id_obj.get("name")
        kind = id_obj.get("kind")
        default = hp_defaults.get(name, 0)
        if kind == "bool":
            hyperparams[name] = bool(value) if value is not None else bool(default)
        elif kind == "int":
            hyperparams[name] = safe_int(value, int(default))
        elif kind == "float":
            hyperparams[name] = safe_float(value, float(default))
        else:
            hyperparams[name] = value if value is not None else default
    return hyperparams


def _build_k_list(opt_mode: str, hetnet_enabled: bool, n_stations, k_min, k_max,
                  ui_n_macro, ui_n_small) -> list[int]:
    if opt_mode == "고정 개수 (Fixed)":
        if hetnet_enabled:
            k = max(safe_int(ui_n_macro, 0) + safe_int(ui_n_small, 0), 1)
            return [k]
        return [safe_int(n_stations, 5)]
    k0 = safe_int(k_min, 3)
    k1 = max(k0, safe_int(k_max, 10))
    return list(range(k0, k1 + 1))


def _run_optimization_thread(
    session_id: str,
    algo: str,
    hyperparams: dict,
    k_list: list[int],
    prop: dict,
    spec_mode: str,
    capacity_default,
    station_specs,
    ui_tx_power, ui_hetnet, ui_n_macro, ui_n_small, ui_macro_power, ui_small_power,
) -> None:
    """백그라운드 스레드: 최적화 실행 후 세션 상태에 결과 저장."""
    try:
        state = get_session_state(session_id)
        env = state.get("env")
        if env is None:
            state["opt_progress"] = {"running": False, "done": False,
                                     "error": "env가 없습니다. 먼저 데이터를 생성하세요."}
            return

        start_time = time.time()
        optimizer = get_optimizer(algo)
        hetnet_enabled = normalize_triggered_bool(ui_hetnet)
        range_results = []

        for k_idx, k in enumerate(k_list):
            cap_k = capacity_for_k(k, spec_mode, station_specs,
                                   safe_float(capacity_default, 2000.0))
            tx_k = tx_power_for_k(
                k,
                hetnet_enabled=hetnet_enabled,
                ui_tx_power=safe_float(ui_tx_power, 43.0),
                n_macro=safe_int(ui_n_macro, 0),
                n_small=safe_int(ui_n_small, 0),
                macro_power=safe_float(ui_macro_power, 43.0),
                small_power=safe_float(ui_small_power, 30.0),
                spec_mode=spec_mode,
                spec_rows=station_specs,
            )
            radius_k = radius_from_tx(tx_k, prop)
            problem = ProblemInput.from_env(
                env,
                radius_m=radius_k,
                capacity=cap_k,
                station_candidate_points=env.station_candidate_points,
                path_loss_exponent=prop["path_loss_exponent"],
                path_loss_ref_db=prop["path_loss_ref_db"],
                tx_power_dbm=tx_k,
                noise_floor_dbm=prop["noise_floor_dbm"],
                sinr_threshold_db=prop["sinr_threshold_db"],
                bandwidth_mhz=prop["bandwidth_mhz"],
            )

            def _progress_cb(it, total, best_stations_local, best_score,
                             _k_idx=k_idx, _problem=problem):
                geo = convert_to_geo(best_stations_local, _problem)
                state["opt_progress"] = {
                    "running": True, "done": False, "error": None,
                    "algo": algo,
                    "k_current": _k_idx + 1, "k_total": len(k_list),
                    "iter": int(it), "total": int(total),
                    "best_score": float(best_score),
                    "stations_geo": geo.tolist(),
                }

            result = optimizer.optimize(problem, n_stations=k,
                                        callback=_progress_cb, **hyperparams)

            stations_geo = convert_to_geo(result.stations, problem)
            stations_df = pd.DataFrame(stations_geo, columns=["lat", "lon"])
            stats_out = dict(result.metrics)
            stats_out["n_stations"] = k
            stats_out["capacity_default"] = (
                float(cap_k[0]) if len(cap_k) > 0
                else safe_float(capacity_default, 2000.0)
            )
            result_pack = {
                "k": k,
                "score": float(result.score),
                "covered_traffic": float(result.metrics.get("covered_traffic", 0)),
                "covered_area": float(result.metrics.get("covered_area", 0)),
                "opt_results": {
                    "algo": algo,
                    "score": float(result.score),
                    "stations_geo": stations_df.to_dict("records"),
                    "capacity": cap_k.tolist(),
                    "history": result.history,
                    "capacity_default": (
                        float(cap_k[0]) if len(cap_k) > 0
                        else safe_float(capacity_default, 2000.0)
                    ),
                    "capacity_per_station": cap_k.tolist(),
                    "prop_params": {**prop, "tx_power_dbm": tx_k.tolist()},
                },
                "stats": stats_out,
            }
            range_results.append(result_pack)

        best_res = max(range_results, key=lambda x: x["score"])
        best_opt = best_res["opt_results"]
        best_stats = best_res["stats"]
        best_tx = np.asarray(
            best_opt.get("prop_params", {}).get("tx_power_dbm", [43.0]), dtype=float)
        best_rows = set_station_spec_rows_from_arrays(
            np.full(len(best_opt.get("stations_geo", [])), 300.0, dtype=float),
            np.asarray(best_opt.get("capacity", best_stats.get("capacity_default", 1000)),
                       dtype=float),
            best_tx,
            300.0,
            float(best_stats.get("capacity_default", 1000)),
            float(best_tx[0]) if len(best_tx) > 0 else 43.0,
        )
        elapsed = time.time() - start_time
        _opt_logger.info("opt_thread done: algo=%s best_k=%s score=%.4f elapsed=%.2fs",
                         algo, best_res["k"], best_res["score"], elapsed)

        # 결과를 먼저 저장한 뒤 done=True 신호 (순서 보장)
        state["range_results"] = range_results
        state["opt_results"] = best_opt
        state["opt_stats"] = best_stats
        state["opt_progress"] = {
            "running": False, "done": True, "error": None,
            "best_rows": best_rows,
            "elapsed": elapsed,
            "best_k": best_res["k"],
            "best_score": best_res["score"],
            "k_total": len(k_list),
        }

    except Exception:
        tb = traceback.format_exc(limit=6)
        _opt_logger.error("opt_thread FAILED: algo=%s\n%s", algo, tb)
        try:
            state = get_session_state(session_id)
            state["opt_progress"] = {"running": False, "done": False, "error": tb}
        except Exception:
            pass


def _make_progress_html(algo: str, k_cur: int, k_tot: int,
                        it: int, total: int, best_score: float) -> html.Div:
    if total > 0:
        pct = min(100.0, it / total * 100)
        overall_pct = ((k_cur - 1) / k_tot + pct / 100.0 / k_tot) * 100.0
        label = f"[{algo}] k {k_cur}/{k_tot} · iter {it}/{total} ({pct:.0f}%) · score {best_score:.2f}"
    else:
        overall_pct = 50.0
        label = f"[{algo}] k {k_cur}/{k_tot} · 계산 중... · score {best_score:.2f}"
    return html.Div(
        [
            html.Div(label, style={"fontSize": "13px", "marginBottom": "4px"}),
            html.Div(
                html.Div(
                    style={
                        "height": "8px",
                        "width": f"{overall_pct:.1f}%",
                        "background": "#2563eb",
                        "borderRadius": "4px",
                        "transition": "width 0.4s ease",
                    }
                ),
                style={
                    "background": "#e5e7eb",
                    "borderRadius": "4px",
                    "overflow": "hidden",
                    "height": "8px",
                },
            ),
        ],
        style={"marginTop": "4px"},
    )


@app.callback(
    Output("optimize-btn", "disabled"),
    Output("opt-poll-interval", "disabled"),
    Output("run-status", "children"),
    Input("optimize-btn", "n_clicks"),
    State("session-id", "data"),
    State("algo-select", "value"),
    State({"type": "hyperparam", "name": ALL, "kind": ALL}, "value"),
    State({"type": "hyperparam", "name": ALL, "kind": ALL}, "id"),
    State("opt-mode", "value"),
    State("n-stations", "value"),
    State("k-min", "value"),
    State("k-max", "value"),
    State("spec-mode", "value"),
    State("capacity-default", "value"),
    State("station-spec-table", "data"),
    State("ui-tx-power", "value"),
    State("ui-path-loss-exp", "value"),
    State("ui-bandwidth-mhz", "value"),
    State("ui-sinr-threshold", "value"),
    State("ui-hetnet", "value"),
    State("ui-n-macro", "value"),
    State("ui-n-small", "value"),
    State("ui-macro-power", "value"),
    State("ui-small-power", "value"),
    prevent_initial_call=True,
)
def start_optimization_job(
    n_clicks, session_id, algo,
    hp_values, hp_ids,
    opt_mode, n_stations, k_min, k_max,
    spec_mode, capacity_default, station_specs,
    ui_tx_power, ui_path_loss_exp, ui_bandwidth_mhz, ui_sinr_threshold,
    ui_hetnet, ui_n_macro, ui_n_small, ui_macro_power, ui_small_power,
):
    if not n_clicks:
        raise PreventUpdate

    state = get_session_state(session_id)

    if state.get("opt_progress", {}).get("running"):
        return False, True, html.Div("이미 계산 중입니다.", style={"color": "#dc2626"})

    if state.get("env") is None:
        return False, True, html.Div("먼저 데이터를 생성해주세요.", style={"color": "#b91c1c"})

    optimizer = get_optimizer(algo)
    hp_defaults = {p.name: p.default for p in optimizer.hyperparams}
    hyperparams = _parse_hyperparams(hp_values, hp_ids, hp_defaults)
    hetnet_enabled = normalize_triggered_bool(ui_hetnet)
    k_list = _build_k_list(opt_mode, hetnet_enabled, n_stations, k_min, k_max,
                           ui_n_macro, ui_n_small)
    prop = prop_params_base(
        path_loss_exponent=safe_float(ui_path_loss_exp, 3.5),
        bandwidth_mhz=safe_float(ui_bandwidth_mhz, 10.0),
        sinr_threshold_db=safe_float(ui_sinr_threshold, 3.0),
    )

    _opt_logger.info("opt_job start: algo=%s k_list=%s hp=%s", algo, k_list, hyperparams)

    state["opt_progress"] = {
        "running": True, "done": False, "error": None,
        "algo": algo,
        "k_current": 1, "k_total": len(k_list),
        "iter": 0, "total": 0, "best_score": 0.0, "stations_geo": [],
    }

    threading.Thread(
        target=_run_optimization_thread,
        args=(session_id, algo, hyperparams, k_list, prop,
              spec_mode, capacity_default, station_specs,
              ui_tx_power, ui_hetnet, ui_n_macro, ui_n_small,
              ui_macro_power, ui_small_power),
        daemon=True,
    ).start()

    status = _make_progress_html(algo, 1, len(k_list), 0, 0, 0.0)
    return True, False, status


@app.callback(
    Output("run-status", "children", allow_duplicate=True),
    Output("opt-live-store", "data"),
    Output("opt-meta", "data", allow_duplicate=True),
    Output("range-meta", "data", allow_duplicate=True),
    Output("station-spec-table", "data", allow_duplicate=True),
    Output("optimize-btn", "disabled", allow_duplicate=True),
    Output("opt-poll-interval", "disabled", allow_duplicate=True),
    Input("opt-poll-interval", "n_intervals"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def poll_optimization_progress(n_intervals, session_id):
    state = get_session_state(session_id)
    progress = state.get("opt_progress")

    if not progress:
        raise PreventUpdate

    # Error branch
    if progress.get("error") and not progress.get("running"):
        tb = progress["error"]
        state["opt_progress"] = {}
        return (
            html.Div(f"계산 실패:\n{tb}",
                     style={"color": "#b91c1c", "whiteSpace": "pre-wrap"}),
            no_update, no_update, no_update, no_update,
            False, True,
        )

    # Done branch
    if progress.get("done"):
        best_rows = progress.get("best_rows")
        elapsed = progress.get("elapsed", 0.0)
        k_total = progress.get("k_total", 1)
        best_k = progress.get("best_k", "?")
        best_score = progress.get("best_score", 0.0)
        state["opt_progress"] = {}
        _opt_logger.info("poll: done — best_k=%s score=%.4f elapsed=%.2fs",
                         best_k, best_score, elapsed)
        status = html.Div(
            f"계산 완료: {k_total}개 시나리오, 최고 k={best_k}, "
            f"score={best_score:.2f}, 소요 {elapsed:.2f}초",
            style={"color": "#166534"},
        )
        return status, no_update, version_token(), version_token(), best_rows, False, True

    # Running branch
    if not progress.get("running"):
        raise PreventUpdate

    algo = progress.get("algo", "")
    k_cur = progress.get("k_current", 1)
    k_tot = progress.get("k_total", 1)
    it = progress.get("iter", 0)
    total = progress.get("total", 0)
    best_score = progress.get("best_score", 0.0)

    status = _make_progress_html(algo, k_cur, k_tot, it, total, best_score)
    return status, version_token(), no_update, no_update, no_update, no_update, no_update


@app.callback(
    Output("stats-panel", "children"),
    Input("opt-meta", "data"),
    State("session-id", "data"),
)
def render_stats_panel(opt_meta, session_id):
    state = get_session_state(session_id)
    stats = state.get("opt_stats")

    if not stats:
        return []

    total_t = float(stats.get("total_traffic", 0))
    cov_t = float(stats.get("covered_traffic", 0))
    total_a = float(stats.get("total_area", 0))
    cov_a = float(stats.get("covered_area", 0))

    traffic_cov_pct = (cov_t / total_t) * 100 if total_t > 0 else 0
    area_cov_pct = (cov_a / total_a) * 100 if total_a > 0 else 0

    mean_sinr = stats.get("mean_sinr_db")

    return [
        metric_card("총 트래픽", f"{int(total_t)}"),
        metric_card("커버된 트래픽", f"{int(cov_t)} ({traffic_cov_pct:.1f}%)"),
        metric_card("커버된 면적", f"{int(cov_a)} 격자 ({area_cov_pct:.1f}%)"),
        metric_card("평균 SINR", f"{mean_sinr:.1f} dB" if mean_sinr is not None else "-"),
        metric_card("기지국 수", f"{stats.get('n_stations', '-')}"),
    ]


@app.callback(
    Output("range-panel", "children"),
    Input("range-meta", "data"),
    State("session-id", "data"),
)
def render_range_panel(range_meta, session_id):
    state = get_session_state(session_id)
    results = state.get("range_results")

    if not results:
        return []

    df_res = pd.DataFrame(
        [
            {
                "k": r["k"],
                "score": r["score"],
                "covered_traffic": r["covered_traffic"],
                "covered_area": r["covered_area"],
            }
            for r in results
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_res["k"],
            y=df_res["covered_traffic"],
            mode="lines+markers",
            name="Covered Traffic",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_res["k"],
            y=df_res["score"],
            mode="lines+markers",
            name="Score",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="범위 탐색 결과",
        xaxis_title="Number of Stations",
        yaxis=dict(title="Covered Traffic"),
        yaxis2=dict(title="Score", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return html.Div(
        [
            html.H2("보고서"),

            dcc.Graph(figure=fig),

            html.Div(
                [
                    html.Label("기지국 개수 선택"),

                    dcc.Dropdown(
                        id="range-k-dropdown",
                        options=[
                            {"label": f"k={int(k)}", "value": int(k)}
                            for k in df_res["k"]
                        ],
                        value=int(df_res.loc[df_res["score"].idxmax(), "k"]),
                        clearable=False,
                        style={
                            "width": "220px",
                            "display": "inline-block",
                            "marginRight": "8px",
                        },
                    ),

                    html.Button("선택 결과 적용", id="apply-k-btn", n_clicks=0),
                ],
                style={"margin": "8px 0"},
            ),

            dash_table.DataTable(
                data=df_res.round(3).to_dict("records"),
                columns=[{"name": c, "id": c} for c in df_res.columns],
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": "13px", "padding": "6px"},
            ),
        ]
    )


@app.callback(
    Output("opt-meta", "data", allow_duplicate=True),
    Output("station-spec-table", "data", allow_duplicate=True),
    Output("run-status", "children", allow_duplicate=True),
    Input("apply-k-btn", "n_clicks"),
    State("range-k-dropdown", "value"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def apply_range_selection(n_clicks, selected_k, session_id):
    if not n_clicks or selected_k is None:
        raise PreventUpdate

    state = get_session_state(session_id)
    results = state.get("range_results") or []

    selected = next((r for r in results if int(r["k"]) == int(selected_k)), None)

    if selected is None:
        raise PreventUpdate

    state["opt_results"] = selected["opt_results"]
    state["opt_stats"] = selected["stats"]

    opt = selected["opt_results"]
    stats = selected["stats"]

    tx = np.asarray(
        opt.get("prop_params", {}).get("tx_power_dbm", [43.0]),
        dtype=float,
    )

    rows = set_station_spec_rows_from_arrays(
        np.full(len(opt.get("stations_geo", [])), 300.0, dtype=float),
        np.asarray(opt.get("capacity", stats.get("capacity_default", 1000)), dtype=float),
        tx,
        300.0,
        float(stats.get("capacity_default", 1000)),
        float(tx[0]) if len(tx) > 0 else 43.0,
    )

    return (
        version_token(),
        rows,
        html.Div(f"k={selected_k} 결과를 지도에 적용했습니다.", style={"color": "#166534"}),
    )


@app.callback(
    Output("download-gis-csv", "data"),
    Input("download-gis-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_gis_csv(n_clicks, session_id):
    state = get_session_state(session_id)
    env = state.get("env")

    if env is None:
        raise PreventUpdate

    df = env.get_dataframe()
    return dcc.send_data_frame(df.to_csv, "traffic_geo.csv", index=False)


@app.callback(
    Output("download-local-csv", "data"),
    Input("download-local-btn", "n_clicks"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def download_local_csv(n_clicks, session_id):
    state = get_session_state(session_id)
    env = state.get("env")

    if env is None:
        raise PreventUpdate

    local_data = env.get_local_data_top_left()
    df = pd.DataFrame(local_data, columns=["x", "y", "traffic"])

    return dcc.send_data_frame(df.to_csv, "traffic_local.csv", index=False)


# ---------------------------------------------------------------------------
# Region selection callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("drawn-region-store", "data"),
    Input("region-edit-control", "geojson"),
    prevent_initial_call=True,
)
def handle_drawn_region(geojson):
    """Capture bounding box of the most recently drawn rectangle."""
    if not geojson or not isinstance(geojson, dict):
        raise PreventUpdate
    features = geojson.get("features", [])
    if not features:
        raise PreventUpdate
    feature = features[-1]
    coords = feature.get("geometry", {}).get("coordinates", [[]])[0]
    if not coords or len(coords) < 3:
        raise PreventUpdate
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    south, north = float(min(lats)), float(max(lats))
    west, east = float(min(lons)), float(max(lons))
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    width_km = geodesic((south, west), (south, east)).km
    height_km = geodesic((south, west), (north, west)).km
    return {
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "width_km": round(width_km, 3),
        "height_km": round(height_km, 3),
    }


@app.callback(
    Output("region-popup", "style"),
    Output("region-width-km", "value"),
    Output("region-height-km", "value"),
    Input("drawn-region-store", "data"),
)
def show_region_popup(region_data):
    """Show/hide the dimension-adjustment popup based on drawn region."""
    base_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "width": "100vw",
        "height": "100vh",
        "background": "rgba(0,0,0,0.45)",
        "zIndex": 10000,
        "alignItems": "center",
        "justifyContent": "center",
    }
    if not region_data or not isinstance(region_data, dict):
        return {**base_style, "display": "none"}, no_update, no_update
    w = round(region_data.get("width_km", 2.0), 2)
    h = round(region_data.get("height_km", 2.0), 2)
    return {**base_style, "display": "flex"}, w, h


@app.callback(
    Output("custom-region-store", "data"),
    Output("drawn-region-store", "data", allow_duplicate=True),
    Output("editcontrol-clear-count", "data"),
    Output("custom-region-info", "children"),
    Output("custom-region-info", "style"),
    Output("clear-region-btn", "style"),
    Input("region-confirm-btn", "n_clicks"),
    State("drawn-region-store", "data"),
    State("region-width-km", "value"),
    State("region-height-km", "value"),
    State("editcontrol-clear-count", "data"),
    prevent_initial_call=True,
)
def apply_region(n_clicks, region_data, width_km, height_km, clear_count):
    """Store confirmed custom region and clear the temporary drawn shape."""
    if not n_clicks or not isinstance(region_data, dict):
        raise PreventUpdate
    w = safe_float(width_km, region_data.get("width_km", 2.0))
    h = safe_float(height_km, region_data.get("height_km", 2.0))
    w = max(w, 0.1)
    h = max(h, 0.1)
    custom = {
        "center_lat": region_data["center_lat"],
        "center_lon": region_data["center_lon"],
        "width_km": w,
        "height_km": h,
    }
    info_text = f"선택 영역: {w:.2f} km × {h:.2f} km"
    info_style = {
        "display": "block",
        "fontSize": "12px",
        "marginTop": "8px",
        "padding": "6px 8px",
        "background": "#f0fdf4",
        "border": "1px solid #86efac",
        "borderRadius": "4px",
        "color": "#166534",
    }
    clear_style = {
        "display": "block",
        "width": "100%",
        "padding": "6px 12px",
        "marginTop": "4px",
        "cursor": "pointer",
        "background": "#dc2626",
        "color": "white",
        "border": "0",
        "borderRadius": "6px",
        "fontSize": "12px",
        "fontWeight": "600",
    }
    return custom, None, int(clear_count or 0) + 1, info_text, info_style, clear_style


@app.callback(
    Output("drawn-region-store", "data", allow_duplicate=True),
    Output("editcontrol-clear-count", "data", allow_duplicate=True),
    Input("region-cancel-btn", "n_clicks"),
    State("editcontrol-clear-count", "data"),
    prevent_initial_call=True,
)
def cancel_region(n_clicks, clear_count):
    """Dismiss the popup and clear the drawn shape."""
    if not n_clicks:
        raise PreventUpdate
    return None, int(clear_count or 0) + 1


@app.callback(
    Output("region-edit-control", "editToolbar"),
    Input("editcontrol-clear-count", "data"),
    prevent_initial_call=True,
)
def sync_editcontrol_clear(count):
    """Programmatically clear all drawn shapes whenever the counter increments."""
    if not count:
        raise PreventUpdate
    return {"mode": "remove", "action": "clear all", "n_clicks": int(count)}


@app.callback(
    Output("custom-region-store", "data", allow_duplicate=True),
    Output("custom-region-info", "children", allow_duplicate=True),
    Output("custom-region-info", "style", allow_duplicate=True),
    Output("clear-region-btn", "style", allow_duplicate=True),
    Input("clear-region-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_custom_region(n_clicks):
    """Remove the confirmed custom region."""
    if not n_clicks:
        raise PreventUpdate
    hidden_info = {
        "display": "none",
        "fontSize": "12px",
        "marginTop": "8px",
        "padding": "6px 8px",
        "background": "#f0fdf4",
        "border": "1px solid #86efac",
        "borderRadius": "4px",
        "color": "#166534",
    }
    hidden_btn = {
        "display": "none",
        "width": "100%",
        "padding": "6px 12px",
        "marginTop": "4px",
        "cursor": "pointer",
        "background": "#dc2626",
        "color": "white",
        "border": "0",
        "borderRadius": "6px",
        "fontSize": "12px",
        "fontWeight": "600",
    }
    return None, "", hidden_info, hidden_btn


# ---------------------------------------------------------------------------
# Algorithm history visualization callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("algo-history-store", "data"),
    Output("algo-history-slider", "max"),
    Output("algo-history-slider", "value"),
    Output("algo-history-wrap", "style"),
    Input("opt-meta", "data"),
    State("session-id", "data"),
    prevent_initial_call=True,
)
def populate_history_store(opt_meta, session_id):
    hidden_style = {
        "display": "none", "marginBottom": "12px",
        "background": "#fff", "padding": "10px", "borderRadius": "8px",
    }
    visible_style = {
        "display": "block", "marginBottom": "12px",
        "background": "#fff", "padding": "10px", "borderRadius": "8px",
    }

    state = get_session_state(session_id)
    opt_results = state.get("opt_results")
    env = state.get("env")

    if not opt_results or not env:
        return None, 1, 0, hidden_style

    history = opt_results.get("history") or []
    algo = opt_results.get("algo", "")

    snapshot_entries = [e for e in history if "stations" in e]
    if len(snapshot_entries) < 2:
        return None, 1, 0, hidden_style

    x_scale = (env.lon_max - env.lon_min) / env.width_m
    y_scale = (env.lat_max - env.lat_min) / env.height_m

    frames = []
    for entry in snapshot_entries:
        local = np.array(entry["stations"], dtype=float)
        if local.ndim != 2 or local.shape[1] != 2:
            continue
        lon = env.lon_min + local[:, 0] * x_scale
        lat = env.lat_min + local[:, 1] * y_scale
        frames.append({
            "iter": int(entry["iter"]),
            "best_score": float(entry.get("best_score", 0)),
            "stations_geo": [[float(la), float(lo)] for la, lo in zip(lat, lon)],
        })

    if not frames:
        return None, 1, 0, hidden_style

    score_series = [
        {
            "iter": int(e["iter"]),
            "best_score": float(e.get("best_score", 0)),
            "gen_score": float(e.get("gen_best_score", e.get("current_score", e.get("best_score", 0)))),
        }
        for e in history
    ]

    algo_history_data = {"algo": algo, "frames": frames, "score_series": score_series}
    n_frames = len(frames)
    return algo_history_data, n_frames - 1, n_frames - 1, visible_style


@app.callback(
    Output("algo-history-label", "children"),
    Input("algo-history-slider", "value"),
    Input("algo-history-store", "data"),
)
def update_algo_history_label(frame_idx, algo_history):
    if not isinstance(algo_history, dict) or not algo_history.get("frames"):
        return ""
    frames = algo_history["frames"]
    n_frames = len(frames)
    idx = min(safe_int(frame_idx, 0), n_frames - 1)
    frame = frames[idx]
    algo = algo_history.get("algo", "알고리즘")
    suffix = " (최종 결과)" if idx == n_frames - 1 else ""
    return f"{algo} 수렴 과정{suffix}: {idx + 1}/{n_frames} | score: {frame['best_score']:.4f}"


@app.callback(
    Output("algo-history-chart", "figure"),
    Input("algo-history-slider", "value"),
    Input("algo-history-store", "data"),
)
def update_algo_history_chart(frame_idx, algo_history):
    if not isinstance(algo_history, dict) or not algo_history.get("score_series"):
        empty = go.Figure()
        empty.update_layout(margin={"l": 30, "r": 10, "t": 20, "b": 30}, height=150)
        return empty

    series = algo_history["score_series"]
    iters = [e["iter"] for e in series]
    best_scores = [e["best_score"] for e in series]
    gen_scores = [e.get("gen_score") for e in series]

    frames = algo_history.get("frames", [])
    n_frames = len(frames)
    idx = min(safe_int(frame_idx, 0), n_frames - 1) if n_frames > 0 else 0
    current_iter = frames[idx]["iter"] if frames else 0

    fig = go.Figure()

    if any(g is not None for g in gen_scores):
        fig.add_trace(go.Scatter(
            x=iters,
            y=gen_scores,
            mode="lines",
            name="Gen Score",
            line={"color": "#9ca3af", "width": 1, "dash": "dot"},
        ))

    fig.add_trace(go.Scatter(
        x=iters,
        y=best_scores,
        mode="lines",
        name="Best Score",
        line={"color": "#2563eb", "width": 2},
    ))

    fig.add_vline(x=current_iter, line_color="#ea580c", line_width=2, line_dash="dash")

    fig.update_layout(
        margin={"l": 30, "r": 10, "t": 10, "b": 30},
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="#f9fafb",
        xaxis={"title": "Iteration", "tickfont": {"size": 10}},
        yaxis={"title": "Score", "tickfont": {"size": 10}},
        height=150,
    )
    return fig


@app.callback(
    Output("algo-frame-interval", "disabled"),
    Output("algo-play-btn", "children"),
    Input("algo-play-btn", "n_clicks"),
    State("algo-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_algo_playback(n_clicks, disabled):
    next_disabled = not bool(disabled)
    return next_disabled, "▶ 재생" if next_disabled else "⏹ 정지"


@app.callback(
    Output("algo-history-slider", "value", allow_duplicate=True),
    Input("algo-frame-interval", "n_intervals"),
    State("algo-history-slider", "value"),
    State("algo-history-slider", "max"),
    State("algo-frame-interval", "disabled"),
    prevent_initial_call=True,
)
def advance_algo_frame(n_intervals, current_value, max_value, disabled):
    if disabled:
        raise PreventUpdate
    current = safe_int(current_value, 0)
    max_frame = max(0, safe_int(max_value, 0))
    if max_frame <= 0:
        raise PreventUpdate
    return (current + 1) % (max_frame + 1)


@app.callback(
    Output("algo-history-slider", "value", allow_duplicate=True),
    Input("algo-reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def reset_algo_frame(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return 0


if __name__ == "__main__":
    test_port = os.environ.get("DASH_PORT")
    if test_port:
        app.run(debug=False, port=int(test_port), host="127.0.0.1", use_reloader=False)
    else:
        app.run(debug=True)
