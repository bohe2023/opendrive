import argparse
import json
import os
import yaml

from ingest.loader import load_all
from normalize.core import build_centerline
from topology.core import make_sections, build_lane_topology
from mapping.core import mark_type_from_division_row
from writer.xodr_writer import write_xodr

def build_lane_spec(sections, lane_topo, defaults, lane_div_df):
    """
    Build simple per-section lane spec:
      - lanes: from topology hint (no center(0); writer adds it)
      - width: default constant for now
      - roadMark: one type per section (solid/broken basic)
    """
    lanes_guess = lane_topo.get("lanes_guess") or [-1, 1]

    roadmark_type = "solid"
    if lane_div_df is not None and len(lane_div_df) > 0:
        roadmark_type = mark_type_from_division_row(lane_div_df.iloc[0])

    out = []
    for sec in sections:
        out.append({
            "s0": sec["s0"], "s1": sec["s1"],
            "lanes": lanes_guess,
            "lane_width": defaults.get("lane_width_m", 3.5),
            "roadMark": roadmark_type,
            "predecessor": True,
            "successor": True,
        })
    return out

def main():
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
