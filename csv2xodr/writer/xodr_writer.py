from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom


def _format_float(value: float, precision: int = 6) -> str:
    return f"{float(value):.{precision}f}"


def _pretty(elem: Element) -> bytes:
    rough = tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")

def write_xodr(
    centerline,
    sections,
    lane_spec_per_section,
    out_path,
    geo_ref=None,
    elevation_profile=None,
    geometry_segments=None,
    superelevation_profile=None,
    road_metadata=None,
):
    road_metadata = road_metadata or {}

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

    road_type = str(road_metadata.get("type", "town"))
    type_el = SubElement(road, "type", {"s": "0.0", "type": road_type})

    speed = road_metadata.get("speed")
    if speed is not None:
        if isinstance(speed, (int, float)):
            speed_attrs = {"max": _format_float(speed), "unit": "m/s"}
        elif isinstance(speed, dict):
            speed_attrs = {str(k): str(v) for k, v in speed.items()}
            if "max" in speed_attrs:
                try:
                    speed_attrs["max"] = _format_float(float(speed_attrs["max"]))
                except (TypeError, ValueError):
                    pass
        else:
            raise TypeError("speed must be a number or a mapping of attributes")

        SubElement(type_el, "speed", speed_attrs)

    # planView with piecewise lines
    plan = SubElement(road, "planView")
    if geometry_segments:
        for seg in geometry_segments:
            geom = SubElement(
                plan,
                "geometry",
                {
                    "s": _format_float(seg["s"], precision=6),
                    "x": _format_float(seg["x"], precision=6),
                    "y": _format_float(seg["y"], precision=6),
                    "hdg": _format_float(seg["hdg"], precision=9),
                    "length": _format_float(seg["length"], precision=6),
                },
            )
            curvature = float(seg.get("curvature", 0.0))
            if abs(curvature) > 1e-9:
                SubElement(geom, "arc", {"curvature": _format_float(curvature, precision=9)})
            else:
                SubElement(geom, "line")
    else:
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
                    "s": _format_float(s, precision=6),
                    "x": _format_float(x, precision=6),
                    "y": _format_float(y, precision=6),
                    "hdg": _format_float(hdg, precision=9),
                    "length": _format_float(seg_len, precision=6),
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

    if superelevation_profile:
        lateral = SubElement(road, "lateralProfile")
        for entry in superelevation_profile:
            attrs = {
                "s": f"{entry['s']:.3f}",
                "a": _format_float(entry.get("a", 0.0)),
                "b": _format_float(entry.get("b", 0.0)),
                "c": _format_float(entry.get("c", 0.0)),
                "d": _format_float(entry.get("d", 0.0)),
            }
            SubElement(lateral, "superelevation", attrs)

    # lanes
    lanes = SubElement(road, "lanes")
    for sec in lane_spec_per_section:
        attrs = {"s": _format_float(sec["s0"], precision=6)}
        has_left = bool(sec.get("left"))
        has_right = bool(sec.get("right"))
        if has_left != has_right:
            attrs["singleSide"] = "true"

        ls = SubElement(lanes, "laneSection", attrs)

        center_el = SubElement(ls, "center")
        SubElement(center_el, "lane", {"id": "0", "type": "none", "level": "false"})

        left_el = SubElement(ls, "left") if has_left else None
        right_el = SubElement(ls, "right") if has_right else None

        def _write_lane(parent, lane_data):
            lane_id = lane_data["id"]
            lane_type = lane_data.get("type", "driving")
            ln = SubElement(
                parent,
                "lane",
                {
                    "id": str(lane_id),
                    "type": lane_type,
                    "level": str(lane_data.get("level", "false")).lower(),
                },
            )
            width = float(lane_data.get("width", 3.5))
            SubElement(ln, "width", {"sOffset": "0.0", "a": f"{width:.3f}", "b": "0", "c": "0", "d": "0"})
            road_mark = lane_data.get("roadMark")
            if road_mark is None and lane_type != "shoulder":
                road_mark = {"type": "solid", "width": 0.12, "laneChange": "both"}

            if road_mark is not None:
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
            if left_el is None:
                left_el = SubElement(ls, "left")
            _write_lane(left_el, lane_data)

        for lane_data in sec.get("right", []):
            if right_el is None:
                right_el = SubElement(ls, "right")
            _write_lane(right_el, lane_data)

    with open(out_path, "wb") as f:
        f.write(_pretty(odr))
    return out_path
