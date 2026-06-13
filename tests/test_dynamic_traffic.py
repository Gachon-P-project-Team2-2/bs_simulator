from __future__ import annotations

import numpy as np

from app import compute_dynamic_scenario_summary, compute_frame_metrics
from environment import SyntheticEnvironment


def _make_env() -> SyntheticEnvironment:
    return SyntheticEnvironment(width_km=1.0, height_km=1.0, resolution_m=100)


def test_dynamic_traffic_types_create_time_series():
    for dynamic_type in ("fixed_variation", "moving_hotspot", "switching_locations"):
        env = _make_env()
        series = env.generate_dynamic_traffic_pattern_density(
            area_demand_mbps_km2=150,
            pattern="multi_hotspot",
            time_steps=6,
            variation=0.5,
            drift_m=200,
            dynamic_type=dynamic_type,
            params={
                "centers": [(3, 3), (7, 7)],
                "n_centers": 2,
                "sigma_x": 1.5,
                "sigma_y": 1.5,
                "noise_std": 0.0,
            },
        )

        assert series.shape == (6, env.rows, env.cols)
        assert env.traffic_series is not None
        assert np.isfinite(series).all()
        assert series.max() > series.min()


def test_fixed_variation_keeps_peak_location_stable():
    env = _make_env()
    series = env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=5,
        variation=0.4,
        dynamic_type="fixed_variation",
        params={
            "centers": [(4, 6)],
            "n_centers": 1,
            "sigma_x": 1.2,
            "sigma_y": 1.2,
            "noise_std": 0.0,
        },
    )

    peak_locations = [np.unravel_index(int(np.argmax(frame)), frame.shape) for frame in series]
    assert len(set(peak_locations)) == 1


def test_switching_locations_changes_peak_location():
    env = _make_env()
    series = env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="center_hotspot",
        time_steps=8,
        variation=0.8,
        dynamic_type="switching_locations",
        params={"noise_std": 0.0},
    )

    peak_locations = {np.unravel_index(int(np.argmax(frame)), frame.shape) for frame in series}
    assert len(peak_locations) > 1


def test_frame_metrics_change_with_dynamic_frame():
    env = _make_env()
    env.generate_dynamic_traffic_pattern_density(
        area_demand_mbps_km2=150,
        pattern="multi_hotspot",
        time_steps=5,
        variation=0.6,
        dynamic_type="fixed_variation",
        params={
            "centers": [(5, 5)],
            "n_centers": 1,
            "sigma_x": 1.5,
            "sigma_y": 1.5,
            "noise_std": 0.0,
        },
    )
    station_lat, station_lon = env.local_points_to_geo(np.array([[500.0, 500.0]]))[0]
    opt_results = {
        "stations_geo": [{"lat": float(station_lat), "lon": float(station_lon)}],
        "prop_params": {
            "path_loss_ref_db": 38.0,
            "path_loss_exponent": 3.5,
            "sinr_threshold_db": 3.0,
            "bandwidth_mhz": 10.0,
            "noise_floor_dbm": -97.0,
            "max_coord_stations": 1,
            "tx_power_dbm": [43.0],
        },
    }

    frame0 = compute_frame_metrics(env, opt_results, None, frame_index=0)
    frame1 = compute_frame_metrics(env, opt_results, None, frame_index=1)
    summary = compute_dynamic_scenario_summary(env, opt_results, None)

    assert frame0 is not None
    assert frame1 is not None
    assert frame0["total_traffic"] != frame1["total_traffic"]
    assert summary is not None
    assert summary["avg_traffic_coverage_pct"] >= summary["worst_traffic_coverage_pct"]
