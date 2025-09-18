from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom as minidom

def _pretty(elem: Element) -> bytes:
    rough = tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8")

def write_xodr(centerline, sections, lane_spec_per_section, out_path, geo_ref=None):
    # root + header
    odr = Element("OpenDRIVE")
    header = SubElement(odr, "header", {
        "revMajor": "1", "revMinor": "4",
        "name": "csv2xodr", "version": "1.00", "date": "2025-09-16"
    })
    if geo_ref:
        SubElement(header, "geoReference").text = geo_ref

    # single road
    length = float(centerline["s"].iloc[-1])
    road = SubElement(odr, "road", {"name": "road_1", "length": f"{length:.3f}", "id": "1", "junction": "-1"})

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
        geom = SubElement(plan, "geometry", {
            "s": f"{s:.3f}", "x": f"{x:.3f}", "y": f"{y:.3f}",
            "hdg": f"{hdg:.6f}", "length": f"{seg_len:.3f}"
        })
        SubElement(geom, "line")

    # lanes
    lanes = SubElement(road, "lanes")
    for sec in lane_spec_per_section:
        ls = SubElement(lanes, "laneSection", {"s": f"{sec['s0']:.3f}"})

        # center lane (id=0) – create only once per section
        center_el = SubElement(ls, "center")
        SubElement(center_el, "lane", {"id": "0", "type": "driving", "level": "false"})

        left_el = SubElement(ls, "left")
        right_el = SubElement(ls, "right")

        # left lanes: id negative, from outer to inner (e.g., -3,-2,-1)
        for lane_id in sorted([l for l in sec["lanes"] if l < 0]):
            ln = SubElement(left_el, "lane", {"id": str(lane_id), "type": "driving", "level": "false"})
            SubElement(ln, "width", {"sOffset": "0.0", "a": f"{sec['lane_width']:.2f}", "b": "0", "c": "0", "d": "0"})
            SubElement(ln, "roadMark", {"sOffset": "0.0", "type": sec.get("roadMark", "solid"),
                                         "weight": "standard", "width": "0.12", "color": "standard",
                                         "laneChange": "both"})
            link = SubElement(ln, "link")
            if sec.get("predecessor"):
                SubElement(link, "predecessor", {"id": str(lane_id)})
            if sec.get("successor"):
                SubElement(link, "successor", {"id": str(lane_id)})

        # right lanes: id positive, from inner to outer (1..N)
        for lane_id in sorted([l for l in sec["lanes"] if l > 0]):
            ln = SubElement(right_el, "lane", {"id": str(lane_id), "type": "driving", "level": "false"})
            SubElement(ln, "width", {"sOffset": "0.0", "a": f"{sec['lane_width']:.2f}", "b": "0", "c": "0", "d": "0"})
            SubElement(ln, "roadMark", {"sOffset": "0.0", "type": sec.get("roadMark", "solid"),
                                         "weight": "standard", "width": "0.12", "color": "standard",
                                         "laneChange": "both"})
            link = SubElement(ln, "link")
            if sec.get("predecessor"):
                SubElement(link, "predecessor", {"id": str(lane_id)})
            if sec.get("successor"):
                SubElement(link, "successor", {"id": str(lane_id)})

    with open(out_path, "wb") as f:
        f.write(_pretty(odr))
    return out_path
