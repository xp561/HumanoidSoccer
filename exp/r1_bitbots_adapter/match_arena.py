"""Build an independent Small-field MJCF around the frozen R1 model.

Only scene visuals, static goals, and the explicitly selected ball change.
Robot, actuators, timestep and ground contact parameters are inherited.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .match_geometry import DEFAULT_CONFIG, bitbots_parameters, load_config


FORMAL = Path("logs/r1_logic_fix/r1_mujoco_formal_20261017")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(values):
    return " ".join(f"{value:.12g}" for value in values)


def field_to_world(point, heading):
    c, s = math.cos(heading), math.sin(heading)
    return [c * point[0] - s * point[1], s * point[0] + c * point[1], point[2]]


def world_to_field(point, heading):
    return field_to_world(point, -heading)


def reference_heading():
    import numpy as np

    meta = json.loads((FORMAL / "artifacts/metadata.json").read_text())
    with np.load(
        Path("motions/r1-corrected") / (meta["motion_names"][0] + ".npz")
    ) as motion:
        q = motion["body_quat_w"][0, list(motion["body_names"]).index("pelvis_link")]
    return float(
        math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
    )


def build_scene(source, output, config, *, ball_profile, heading):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source.parent == output.parent:
        raise ValueError("Arena must be generated outside the frozen asset directory")
    if not math.isfinite(heading):
        raise ValueError("Nonfinite field heading")
    ball = config["ball_profiles"][ball_profile]
    tree = ET.parse(source)
    root = tree.getroot()
    # Make any existing external asset references independent of output location.
    compiler = root.find("compiler")
    for attr in ("meshdir", "texturedir", "assetdir"):
        value = compiler.get(attr)
        if value and not Path(value).is_absolute():
            compiler.set(attr, str((source.parent / value).resolve()))
    world = root.find("worldbody")
    if world.find("body[@name='match_arena']") is not None:
        raise ValueError("Source already contains an arena")
    geom = world.find("body[@name='soccer_ball']/geom[@name='soccer_ball_geom']")
    if geom is None:
        raise ValueError("Expected frozen soccer ball")
    geom.set("size", str(ball["radius"]))
    geom.set("mass", str(ball["mass"]))
    geom.set("rgba", "0.95 0.95 0.95 1")
    body = ET.SubElement(
        world, "body", name="match_arena", euler=numbers([0, 0, heading])
    )
    length, width = config["field_length"], config["field_width"]
    half_x, half_y = length / 2, width / 2
    line_width = config["line_width"]
    visual_geoms = []

    def box(name, centre, half_size, rgba, *, collision=False, angle=0):
        attrs = dict(
            name=name,
            type="box",
            pos=numbers(centre),
            size=numbers(half_size),
            rgba=rgba,
        )
        if not collision:
            attrs.update(contype="0", conaffinity="0", group="1")
            visual_geoms.append(name)
        if angle:
            attrs["euler"] = numbers([0, 0, angle])
        return ET.SubElement(body, "geom", **attrs)

    def line(name, start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        box(
            name,
            [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, 0.001],
            [math.hypot(dx, dy) / 2, line_width / 2, 0.0004],
            "1 1 1 1",
            angle=math.atan2(dy, dx),
        )

    box(
        "arena_carpet_visual",
        [0, 0, 0.0002],
        [half_x + config["padding"], half_y + config["padding"], 0.0001],
        "0.12 0.36 0.16 1",
    )
    # Outside edges, not centres, are the specified 9 x 6 m bounds.
    for sign in (-1, 1):
        x, y = sign * (half_x - line_width / 2), sign * (half_y - line_width / 2)
        line(f"arena_endline_{sign}", [x, -half_y], [x, half_y])
        line(f"arena_touchline_{sign}", [-half_x, y], [half_x, y])
    line("arena_halfway", [0, -half_y], [0, half_y])
    for area in ("goal_area", "penalty_area"):
        depth, span = config[area + "_depth"], config[area + "_width"]
        for sign in (-1, 1):
            x = sign * (half_x - depth + line_width / 2)
            y = span / 2 - line_width / 2
            line(f"arena_{area}_front_{sign}", [x, -span / 2], [x, span / 2])
            for edge in (-1, 1):
                line(
                    f"arena_{area}_side_{sign}_{edge}",
                    [sign * (half_x - depth), edge * y],
                    [sign * half_x, edge * y],
                )
    radius = (config["center_circle_diameter"] - line_width) / 2
    for i in range(128):
        a, b = i * 2 * math.pi / 128, (i + 1) * 2 * math.pi / 128
        line(
            f"arena_circle_{i}",
            [radius * math.cos(a), radius * math.sin(a)],
            [radius * math.cos(b), radius * math.sin(b)],
        )
    for i, x in enumerate(
        (
            0,
            half_x - config["penalty_mark_distance"],
            -half_x + config["penalty_mark_distance"],
        )
    ):
        name = f"arena_mark_{i}"
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="cylinder",
            pos=numbers([x, 0, 0.001]),
            size=numbers([config["mark_diameter"] / 2, 0.0004]),
            rgba="1 1 1 1",
            contype="0",
            conaffinity="0",
            group="1",
        )
        visual_geoms.append(name)
    r, opening, height, depth = (
        config[k]
        for k in ("post_radius", "goal_clear_width", "goal_clear_height", "goal_depth")
    )
    for sign in (-1, 1):
        # Back of front posts/crossbar is flush with the goal-side line edge.
        x = sign * (half_x - r)
        for edge in (-1, 1):
            y = edge * (opening / 2 + r)
            ET.SubElement(
                body,
                "geom",
                name=f"arena_post_{sign}_{edge}",
                type="cylinder",
                fromto=numbers([x, y, 0, x, y, height + r]),
                size=str(r),
                rgba="1 1 1 1",
            )
        ET.SubElement(
            body,
            "geom",
            name=f"arena_bar_{sign}",
            type="cylinder",
            fromto=numbers(
                [x, -opening / 2 - r, height + r, x, opening / 2 + r, height + r]
            ),
            size=str(r),
            rgba="1 1 1 1",
        )
        box(
            f"arena_backnet_{sign}",
            [sign * (half_x + depth + 0.01), 0, height / 2],
            [0.01, opening / 2 + 2 * r, height / 2],
            ".85 .85 .85 .25",
            collision=True,
        )
        for edge in (-1, 1):
            box(
                f"arena_sidenet_{sign}_{edge}",
                [
                    sign * (half_x + depth / 2),
                    edge * (opening / 2 + 2 * r + 0.01),
                    height / 2,
                ],
                [depth / 2, 0.01, height / 2],
                ".85 .85 .85 .2",
                collision=True,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode")
    manifest = dict(
        config=config,
        ball_profile=ball_profile,
        ball=ball,
        source_model=str(source),
        source_model_sha256=sha256(source),
        generated_model=str(output),
        generated_model_sha256=sha256(output),
        field_heading_in_odom_rad=heading,
        field_frame="map aligned to frozen reference forward; origin at world origin",
        goal_width_convention="clear opening; Bit-Bots parameter is post-centre spacing",
        goal_plane_x=[-half_x, half_x],
        visual_only_geometries=visual_geoms,
        physics="original flat ground contact; turf compliance and rolling not calibrated",
        scope="scene/geometry baseline, not match referee, perception or robot eligibility approval",
    )
    (output.parent / "arena_manifest.json").write_text(json.dumps(manifest, indent=2))
    parameters = bitbots_parameters(config)
    lines = ["parameter_blackboard:", "  ros__parameters:"]
    lines.extend(
        "    " + key + ": " + json.dumps(value) for key, value in parameters.items()
    )
    (output.parent / "bitbots_field.yaml").write_text("\n".join(lines) + "\n")
    (output.parent / "navigation_field.yaml").write_text(
        "navigation:\n  map.ball_diameter: " + str(2 * ball["radius"]) + "\n"
    )
    return manifest


def render_map(output, config):
    """Metric line occupancy image and readable engineering field overview."""
    from PIL import Image, ImageDraw

    output = Path(output)
    scale = 100
    hx, hy, pad = (
        config["field_length"] / 2,
        config["field_width"] / 2,
        config["padding"],
    )
    size = (round((2 * hx + 2 * pad) * scale), round((2 * hy + 2 * pad) * scale))
    image = Image.new("RGB", size, (27, 75, 36))
    mask = Image.new("L", size, 0)
    draw, md = ImageDraw.Draw(image), ImageDraw.Draw(mask)

    def point(x, y):
        return (round((x + hx + pad) * scale), round((hy + pad - y) * scale))

    stroke = round(config["line_width"] * scale)

    def rect(a, b):
        bounds = (*point(a[0], b[1]), *point(b[0], a[1]))
        for d, color in ((draw, "white"), (md, 255)):
            d.rectangle(bounds, outline=color, width=stroke)

    rect((-hx, -hy), (hx, hy))
    for d, color in ((draw, "white"), (md, 255)):
        d.line([point(0, -hy), point(0, hy)], fill=color, width=stroke)
        r = config["center_circle_diameter"] / 2
        d.ellipse((*point(-r, r), *point(r, -r)), outline=color, width=stroke)
    for sign in (-1, 1):
        for area in ("goal_area", "penalty_area"):
            x1, x2 = sorted([sign * hx, sign * (hx - config[area + "_depth"])])
            rect((x1, -config[area + "_width"] / 2), (x2, config[area + "_width"] / 2))
        x1, x2 = sorted([sign * hx, sign * (hx + config["goal_depth"])])
        draw.rectangle(
            (
                *point(x1, config["goal_clear_width"] / 2),
                *point(x2, -config["goal_clear_width"] / 2),
            ),
            outline=(210, 210, 210),
            width=2,
        )
    for x in (
        0,
        hx - config["penalty_mark_distance"],
        -hx + config["penalty_mark_distance"],
    ):
        r = config["mark_diameter"] / 2
        for d, color in ((draw, "white"), (md, 255)):
            d.ellipse((*point(x - r, r), *point(x + r, -r)), fill=color)
    mask.save(output / "lines.png")
    draw.text(
        (20, 15),
        "HSL Small engineering baseline | 9.00 x 6.00 m | 1.00 m border",
        fill="white",
    )
    draw.text(
        (20, size[1] - 40),
        "Clear goal 1.80 x 1.25 m, depth 0.60 m | Candidate, event specification pending",
        fill="white",
    )
    image.save(output / "field_topdown.png")
    (output / "lines.yaml").write_text(
        f"image: lines.png\nresolution: 0.01\norigin: [{-hx - pad}, {-hy - pad}, 0.0]\nnegate: 1\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--ball-profile",
        choices=["small_candidate", "frozen_reference"],
        default="small_candidate",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = build_scene(
        FORMAL / "model.xml",
        args.output / "model.xml",
        config,
        ball_profile=args.ball_profile,
        heading=reference_heading(),
    )
    render_map(args.output, config)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("generated_model", "generated_model_sha256", "ball_profile")
            }
        )
    )


if __name__ == "__main__":
    main()
