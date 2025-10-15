import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

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
    filter_dataframe_by_path,
    select_best_path_id,
)
from csv2xodr.line_geometry import build_line_geometry_lookup
from csv2xodr.topology.core import make_sections, build_lane_topology
from csv2xodr.writer.xodr_writer import write_xodr
from csv2xodr.signals import generate_signals
from csv2xodr.lane_spec import build_lane_spec, normalize_lane_ids


def _apply_dataset_overrides(dfs, cfg) -> None:
    """Adjust loaded CSV tables based on optional configuration hints."""

    lane_link = dfs.get("lane_link")
    lane_width = dfs.get("lane_width")

    if lane_link is None or lane_width is None:
        return

    try:
        from csv2xodr.normalize.us_adapters import merge_lane_width_into_links
    except Exception:  # pragma: no cover - optional helper
        return

    enriched = merge_lane_width_into_links(lane_link, lane_width)
    if enriched is not None:
        dfs["lane_link"] = enriched


def _detect_country(sign_filename: Optional[str]) -> str:
    if not sign_filename:
        return "JPN"
    upper = sign_filename.upper()
    if "US" in upper:
        return "US"
    if "JPN" in upper:
        return "JPN"
    return "JPN"


def convert_dataset(input_dir: str, output_path: str, config_path: str) -> dict:
    """Convert a directory of CSV files into an OpenDRIVE file.

    Parameters
    ----------
    input_dir:
        Directory containing the CSV files that describe the road network.
    output_path:
        Target path for the generated ``.xodr`` file.
    config_path:
        Path to the YAML configuration that describes how to read ``input_dir``.

    Returns
    -------
    dict
        Statistics about the conversion, including input counts and metadata about
        the generated OpenDRIVE file.
    """
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        yaml = None
        from csv2xodr.mini_yaml import load as mini_yaml_load
    if yaml is not None:
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    else:
        cfg = mini_yaml_load(config_path)
    dfs = load_all(input_dir, cfg)
    primary_path = select_best_path_id(dfs.get("line_geometry"))
    if primary_path is not None:
        for key, table in list(dfs.items()):
            filtered = filter_dataframe_by_path(table, primary_path)
            if filtered is not None:
                dfs[key] = filtered
    _apply_dataset_overrides(dfs, cfg)

    # planView (centerline) from line geometry + geo origin
    center, (lat0, lon0) = build_centerline(dfs["line_geometry"], dfs["map_base_point"])
    offset_mapper = build_offset_mapper(center)

    # lane sections (from offsets)
    sections = make_sections(center, dfs["lane_link"], dfs["lane_division"], offset_mapper=offset_mapper)

    # lane topology hints (no center id in the guess)
    lane_topo = build_lane_topology(dfs["lane_link"], offset_mapper=offset_mapper)

    # per-section spec (width/roadMark/topology flags)
    curvature_profile, curvature_samples = build_curvature_profile(
        dfs.get("curvature"),
        offset_mapper=offset_mapper,
        centerline=center,
        geo_origin=(lat0, lon0),
        lane_geometry_df=dfs.get("lanes_geometry"),
    )

    line_geometry_lookup = build_line_geometry_lookup(
        dfs["line_geometry"],
        offset_mapper=offset_mapper,
        lat0=lat0,
        lon0=lon0,
        curvature_samples=curvature_samples,
    )
    lane_specs = build_lane_spec(
        sections,
        lane_topo,
        cfg.get("defaults", {}),
        dfs["lane_division"],
        line_geometry_lookup=line_geometry_lookup,
        offset_mapper=offset_mapper,
        lanes_geometry_df=dfs.get("lanes_geometry"),
        centerline=center,
        geo_origin=(lat0, lon0),
    )

    geometry_cfg_raw = cfg.get("geometry") or {}
    if not isinstance(geometry_cfg_raw, dict):
        raise TypeError("geometry configuration must be a mapping if provided")

    max_endpoint_deviation_cfg = geometry_cfg_raw.get("max_endpoint_deviation_m", 0.5)
    try:
        max_endpoint_deviation = float(max_endpoint_deviation_cfg)
    except (TypeError, ValueError) as exc:
        raise TypeError("geometry.max_endpoint_deviation_m must be a number") from exc

    max_segment_length_cfg = geometry_cfg_raw.get("max_segment_length_m", 2.0)
    try:
        max_segment_length = float(max_segment_length_cfg)
    except (TypeError, ValueError) as exc:
        raise TypeError("geometry.max_segment_length_m must be a number") from exc

    geometry_segments = build_geometry_segments(
        center,
        curvature_profile,
        max_endpoint_deviation=max_endpoint_deviation,
        max_segment_length=max_segment_length,
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

    files_cfg = cfg.get("files") or {}
    sign_filename = files_cfg.get("sign") if isinstance(files_cfg, dict) else None
    signal_export = generate_signals(
        dfs.get("sign"),
        country=_detect_country(sign_filename),
        offset_mapper=offset_mapper,
        sign_filename=sign_filename,
    )
    signals = signal_export.signals
    signal_objects = signal_export.objects

    apply_shoulder_profile(lane_specs, shoulder_profile, defaults=cfg.get("defaults", {}))
    normalize_lane_ids(lane_specs)

    # write xodr
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
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

    output_file_path = write_xodr(
        center,
        sections,
        lane_specs,
        str(output_path),
        geo_ref=geo_ref,
        elevation_profile=elevation_profile,
        geometry_segments=geometry_segments,
        superelevation_profile=superelevation_profile,
        signals=signals,
        objects=signal_objects,
        road_metadata=road_cfg,
    )

    output_file_path = Path(output_file_path)
    try:
        file_size = output_file_path.stat().st_size
    except FileNotFoundError:
        file_size = 0
    try:
        with output_file_path.open("rb") as fh:
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
            "signals": len(signals),
            "signalObjects": len(signal_objects),
        },
        "road_length_m": float(center["s"].iloc[-1]) if len(center["s"]) else 0.0,
        "xodr_file": {
            "path": str(output_file_path.resolve()),
            "size_bytes": file_size,
            "line_count": line_count,
        },
    }
    log_path = output_path.parent / "report.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory containing CSVs")
    ap.add_argument("--output", required=True, help="path to output .xodr")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    stats = convert_dataset(args.input, args.output, args.config)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
