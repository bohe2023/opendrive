from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom


def _format_float(value: float, precision: int = 6) -> str:
    return f"{float(value):.{precision}f}"


def _pretty(elem: Element) -> bytes:
    rough = tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")

def write_xodr(centerline, sections, lane_spec_per_section, out_path, geo_ref=None, elevation_profile=None):
    # root + header
    odr = Element("OpenDRIVE")
    header = SubElement(odr, "header", {
        "revMajor": "1",
        "revMinor": "4",
        "name": "csv2xodr",
        "version": "1.00",
        "date": "2025-09-16",
    })
    if geo_ref:
        SubElement(header, "geoReference").text = geo_ref

    # single road
    length = float(centerline["s"].iloc[-1])
    road = SubElement(
        odr,
        "road",
        {"name": "road_1", "length": f"{length:.3f}", "id": "1", "junction": "-1"},
    )

    # planView with piecewise lines
    plan = SubElement(road, "planView")
    for i in range(len(centerline) - 1):
        s = float(centerline["s"].iloc[i])
        x = float(centerline["x"].iloc[i])
        y = float(centerline["y"].iloc[i])
        hdg = float(centerline["hdg"].iloc[i])
        x2 = float(centerline["x"].iloc[i + 1])
        y2 = float(centerline["y"].iloc[i + 1])
        seg_len = ((x2 - x) ** 2 + (y2 - y) ** 2) ** 0.5
        geom = SubElement(
            plan,
            "geometry",
            {
                "s": f"{s:.3f}",
                "x": f"{x:.3f}",
                "y": f"{y:.3f}",
                "hdg": f"{hdg:.6f}",
                "length": f"{seg_len:.3f}",
            },
        )
        SubElement(geom, "line")

    if elevation_profile:
        elev = SubElement(road, "elevationProfile")
        for entry in elevation_profile:
            attrs = {
                "s": f"{entry['s']:.3f}",
                "a": _format_float(entry.get("a", 0.0)),
                "b": _format_float(entry.get("b", 0.0)),
                "c": _format_float(entry.get("c", 0.0)),
                "d": _format_float(entry.get("d", 0.0)),
            }
            SubElement(elev, "elevation", attrs)

    # lanes
    lanes = SubElement(road, "lanes")
    for sec in lane_spec_per_section:
        ls = SubElement(lanes, "laneSection", {"s": f"{sec['s0']:.3f}"})

        center_el = SubElement(ls, "center")
        SubElement(center_el, "lane", {"id": "0", "type": "driving", "level": "false"})

        left_el = SubElement(ls, "left")
        right_el = SubElement(ls, "right")

        def _write_lane(parent, lane_data):
            lane_id = lane_data["id"]
            ln = SubElement(parent, "lane", {"id": str(lane_id), "type": "driving", "level": "false"})
            width = float(lane_data.get("width", 3.5))
            SubElement(ln, "width", {"sOffset": "0.0", "a": f"{width:.3f}", "b": "0", "c": "0", "d": "0"})
            road_mark = lane_data.get("roadMark") or {"type": "solid", "width": 0.12, "laneChange": "both"}
            rm_attrs = {
                "sOffset": "0.0",
                "type": str(road_mark.get("type", "solid")),
                "weight": str(road_mark.get("weight", "standard")),
                "width": f"{float(road_mark.get('width', 0.12)):.3f}",
                "color": str(road_mark.get("color", "standard")),
                "laneChange": str(road_mark.get("laneChange", "both")),
            }
            SubElement(ln, "roadMark", rm_attrs)

            predecessors = lane_data.get("predecessors") or []
            successors = lane_data.get("successors") or []
            if predecessors or successors:
                link = SubElement(ln, "link")
                for pid in predecessors:
                    SubElement(link, "predecessor", {"id": str(pid)})
                for sid in successors:
                    SubElement(link, "successor", {"id": str(sid)})

        for lane_data in sec.get("left", []):
            _write_lane(left_el, lane_data)

        for lane_data in sec.get("right", []):
            _write_lane(right_el, lane_data)

    with open(out_path, "wb") as f:
        f.write(_pretty(odr))
    return out_path
