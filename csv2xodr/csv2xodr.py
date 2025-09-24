import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from csv2xodr.ingest.loader import load_all
from csv2xodr.normalize.core import (
    build_centerline,
    build_curvature_profile,
    build_elevation_profile,
    build_elevation_profile_from_slopes,
    build_geometry_segments,
    build_offset_mapper,
    build_shoulder_profile,
    build_slope_profile,
    build_superelevation_profile,
)
from csv2xodr.line_geometry import build_line_geometry_lookup
from csv2xodr.topology.core import make_sections, build_lane_topology
from csv2xodr.writer.xodr_writer import write_xodr
from csv2xodr.lane_spec import build_lane_spec

def main():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        yaml = None
        from csv2xodr.mini_yaml import load as mini_yaml_load
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory containing CSVs")
    ap.add_argument("--output", required=True, help="path to output .xodr")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    if yaml is not None:
        with open(args.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = mini_yaml_load(args.config)
    dfs = load_all(args.input, cfg)

    # planView (centerline) from line geometry + geo origin
    center, (lat0, lon0) = build_centerline(dfs["line_geometry"], dfs["map_base_point"])
    offset_mapper = build_offset_mapper(center)

    # lane sections (from offsets)
    sections = make_sections(center, dfs["lane_link"], dfs["lane_division"], offset_mapper=offset_mapper)

    # lane topology hints (no center id in the guess)
    lane_topo = build_lane_topology(dfs["lane_link"], offset_mapper=offset_mapper)

    # per-section spec (width/roadMark/topology flags)
    line_geometry_lookup = build_line_geometry_lookup(
        dfs["line_geometry"], offset_mapper=offset_mapper, lat0=lat0, lon0=lon0
    )
    lane_specs = build_lane_spec(
        sections,
        lane_topo,
        cfg.get("defaults", {}),
        dfs["lane_division"],
        line_geometry_lookup=line_geometry_lookup,
        offset_mapper=offset_mapper,
    )

    curvature_profile = build_curvature_profile(dfs.get("curvature"), offset_mapper=offset_mapper)

    geometry_cfg_raw = cfg.get("geometry") or {}
    if not isinstance(geometry_cfg_raw, dict):
        raise TypeError("geometry configuration must be a mapping if provided")

    max_endpoint_deviation_cfg = geometry_cfg_raw.get("max_endpoint_deviation_m", 0.5)
    try:
        max_endpoint_deviation = float(max_endpoint_deviation_cfg)
    except (TypeError, ValueError) as exc:
        raise TypeError("geometry.max_endpoint_deviation_m must be a number") from exc

    geometry_segments = build_geometry_segments(
        center,
        curvature_profile,
        max_endpoint_deviation=max_endpoint_deviation,
    )

    slope_profiles = build_slope_profile(dfs.get("slope"), offset_mapper=offset_mapper)
    longitudinal = slope_profiles.get("longitudinal", [])
    if longitudinal:
        elevation_profile = build_elevation_profile_from_slopes(longitudinal)
    else:
        elevation_profile = build_elevation_profile(dfs["line_geometry"], offset_mapper=offset_mapper)

    superelevation_profile = build_superelevation_profile(slope_profiles.get("superelevation", []))

    shoulder_profile = build_shoulder_profile(dfs.get("shoulder_width"), offset_mapper=offset_mapper)
    from csv2xodr.lane_spec import apply_shoulder_profile  # local import to avoid cycle

    apply_shoulder_profile(lane_specs, shoulder_profile, defaults=cfg.get("defaults", {}))

    # write xodr
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    geo_ref = f"LOCAL_XY origin={lat0},{lon0}"
    road_cfg_raw = cfg.get("road") or {}
    if not isinstance(road_cfg_raw, dict):
        raise TypeError("road configuration must be a mapping if provided")

    road_cfg = {
        "type": road_cfg_raw.get("type", "town"),
    }
    if "speed" in road_cfg_raw:
        road_cfg["speed"] = road_cfg_raw["speed"]
    else:
        road_cfg["speed"] = {"max": 50 / 3.6, "unit": "m/s"}

    output_path = write_xodr(
        center,
        sections,
        lane_specs,
        args.output,
        geo_ref=geo_ref,
        elevation_profile=elevation_profile,
        geometry_segments=geometry_segments,
        superelevation_profile=superelevation_profile,
        road_metadata=road_cfg,
    )

    output_path = Path(output_path)
    try:
        file_size = output_path.stat().st_size
    except FileNotFoundError:
        file_size = 0
    try:
        with output_path.open("rb") as fh:
            line_count = sum(1 for _ in fh)
    except FileNotFoundError:
        line_count = 0

    # stats log
    stats = {
        "input_counts": {k: (0 if v is None else len(v)) for k, v in dfs.items()},
        "output_counts": {
            "roads": 1,
            "laneSections": len(lane_specs),
            "lanes_total": sum(
                1 + len(sec.get("left", [])) + len(sec.get("right", [])) for sec in lane_specs
            ),
        },
        "road_length_m": float(center["s"].iloc[-1]) if len(center["s"]) else 0.0,
        "xodr_file": {
            "path": str(output_path.resolve()),
            "size_bytes": file_size,
            "line_count": line_count,
        },
    }
    log_path = os.path.join(os.path.dirname(args.output), "report.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
