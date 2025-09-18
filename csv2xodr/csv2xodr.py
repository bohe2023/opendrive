import argparse
import json
import os

from csv2xodr.ingest.loader import load_all
from csv2xodr.normalize.core import build_centerline
from csv2xodr.topology.core import make_sections, build_lane_topology
from csv2xodr.writer.xodr_writer import write_xodr
from csv2xodr.lane_spec import build_lane_spec

def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory containing CSVs")
    ap.add_argument("--output", required=True, help="path to output .xodr")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    dfs = load_all(args.input, cfg)

    # planView (centerline) from line geometry + geo origin
    center, (lat0, lon0) = build_centerline(dfs["line_geometry"], dfs["map_base_point"])

    # lane sections (from offsets)
    sections = make_sections(center, dfs["lane_link"], dfs["lane_division"])

    # lane topology hints (no center id in the guess)
    lane_topo, uniq_lane_ids = build_lane_topology(dfs["lane_link"])

    # per-section spec (width/roadMark/topology flags)
    lane_specs = build_lane_spec(sections, lane_topo, cfg.get("defaults", {}), dfs["lane_division"])

    # write xodr
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    geo_ref = f"LOCAL_XY origin={lat0},{lon0}"
    write_xodr(center, sections, lane_specs, args.output, geo_ref=geo_ref)

    # stats log
    stats = {
        "input_counts": {k: (0 if v is None else len(v)) for k, v in dfs.items()},
        "output_counts": {
            "roads": 1,
            "laneSections": len(lane_specs),
            # +1 center per section; lanes list excludes center(0)
            "lanes_total": sum(len(sec["lanes"]) + 1 for sec in lane_specs)
        }
    }
    log_path = os.path.join(os.path.dirname(args.output), "report.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
