"""Exact quotas and held-out geometries for the expanded collection plan."""
import numpy as np

from . import diverse_patterns as motion

VERSION = "expanded-v2"
PLANES = dict(motion.PLANES, **{
    "XY tilted +30 deg about Y": ((np.sqrt(3)/2, 0, -0.5), (0, 1, 0)),
    "XY tilted -30 deg about Y": ((np.sqrt(3)/2, 0, 0.5), (0, 1, 0)),
})
PRIMARY = {
    "line": (25, 30, 35, 45, 50, 60, 75, 85),
    "diagonal": (35, 45, 55, 65),
    "circle": (18, 20, 23, 26, 29, 32, 36, 40),
    "ellipse": (24, 30, 36, 42),
    "helix": (18, 22, 26, 30),
    "zigzag": (36, 48, 60, 72),
    "arc": (20, 26, 32, 40),
    "figure_eight": (18, 24, 30, 36),
}
TEST = {
    "line": (27, 38, 57, 82), "circle": (19, 24, 34, 42),
    "diagonal": (40, 50, 60, 70), "ellipse": (27, 33, 39, 45),
    "helix": (20, 24, 28, 32), "zigzag": (42, 54, 66, 78),
    "arc": (23, 29, 35, 43), "figure_eight": (21, 27, 33, 39),
}
NOVEL = {
    "circle_line_circle": ("sequential", ("circle", "line", "circle")),
    "ellipse_helix": ("sequential", ("ellipse", "helix")),
    "figure_eight_slide": ("synchronous", ("figure_eight", "line")),
    "zigzag_delayed_lift": ("offset", ("zigzag", "lift")),
}
SPEEDS = (50, 70, 100)
STARTS = [(x, y, z) for x in (470, 488, 505, 522, 540)
          for y in (-45, -22, 0, 22, 45) for z in (240, 260, 280, 300)]


def timings(kind, mode, index, rng):
    if kind == "circle_line_circle":
        return [(0, .35), (.35, .65), (.65, 1)]
    if kind == "ellipse_helix":
        return [(0, .4), (.4, 1)]
    if kind == "zigzag_delayed_lift":
        return [(0, 1), ((.25, .85), (.35, .95))[index]]
    if kind == "arc_line_arc":
        a, b = ((.30, .70), (.40, .65), (.25, .60))[index % 3]
        return [(0, a), (a, b), (b, 1)]
    if mode == "sequential":
        cut = {
            "line_circle": (.3, .4, .5), "circle_line": (.5, .6, .7),
            "circle_helix": (.45, .35, .25), "helix_circle": (.55, .65, .75),
            "zigzag_arc": (.35, .45, .55),
        }[kind][index % 3]
        return [(0, cut), (cut, 1)]
    if mode == "offset":
        return [(0, 1), ((.2, .8), (.25, .9), (.35, .9))[index % 3]]
    return None


def profile(t, style):
    if style == "smooth":
        return t*t*(3-2*t)
    d = .1
    return np.where(t < d, t*t/(2*d), np.where(t > 1-d,
                    1-d-(1-t)**2/(2*d), t-d/2))/(1-d)


def displacement(t, item):
    kind = item["kind"]
    if kind not in ("diagonal", "zigzag", "figure_eight", "wave"):
        return motion.component(t, item)
    u, v = np.asarray(PLANES[item["plane"]])
    turn = item["axis_quarter_turns"]
    u, v = ((u,v), (v,-u), (-u,-v), (-v,u))[turn]
    n = np.cross(u,v)
    size, sign = item["radius_or_scale_mm"], item["direction"]
    if kind == "diagonal":
        angle = np.deg2rad(item["heading_deg"])
        delta = size*(np.cos(angle)*u+np.sin(angle)*v) + sign*item["height_mm"]*n
        return t[:,None]*delta, f"move diagonally by {np.round(delta,2).tolist()} mm in base XYZ"
    if kind == "zigzag":
        width = item["width_mm"]
        x = size*t
        y = np.interp(t, (0,1/3,2/3,1), (0,sign*width,-sign*width,0))
        description = f"trace a three-section zigzag {size:g} mm long with {width:g} mm lateral amplitude"
    elif kind == "figure_eight":
        x = size*np.sin(2*np.pi*t)
        y = sign*size*item["figure_ratio"]*np.sin(4*np.pi*t)
        description = f"trace a figure eight with local half-spans {size:g} and {size*item['figure_ratio']:g} mm"
    else:
        x = item["wave_length_mm"]*t
        y = sign*size*np.sin(2*np.pi*t)
        description = f"trace a {item['wave_length_mm']} mm sinusoidal wave with amplitude {size:g} mm"
    return x[:,None]*u+y[:,None]*v, (f"{description} in the {item['plane']} plane with local U at "
        f"{turn*90} degrees, first moving toward {'positive' if sign==1 else 'negative'} local V")


def make_entry(split, kind, plane, direction, speed, size, index, version, start, rng):
    mode, kinds = (motion.COMPOUNDS | NOVEL).get(kind, ("single", (kind,)))
    components = [motion.parameters(part, plane, size, direction, rng) for part in kinds]
    for j, item in enumerate(components):
        k = index + j
        part = item["kind"]
        item["axis_quarter_turns"] = k % 4
        if j:
            item["plane"] = plane if version == 0 else tuple(PLANES)[(tuple(PLANES).index(plane)+1+k) % 7]
            if item["plane"] == plane and version:
                item["plane"] = tuple(PLANES)[(tuple(PLANES).index(plane)+1) % 7]
            item["radius_or_scale_mm"] = PRIMARY.get(part, (15, 25, 35, 45))[k % len(PRIMARY.get(part, (15, 25, 35, 45)))]
            item["direction"] = direction * (-1 if k % 2 else 1)
        item["ellipse_ratio"] = (.5, .6, .7, .8)[k % 4]
        item["rise_mm"] = (20, 30, 40, 50)[k % 4]
        item["arc_degrees"] = (60, 90, 120, 180)[k % 4]
        item["turns"] = 1 + (k % 2)
        if part == "diagonal":
            item.update(heading_deg=(30,60)[k%2], height_mm=(15,25,35,45)[k%4])
        if part == "zigzag":
            item["width_mm"] = (10,15,20,25)[k%4]
        if part == "figure_eight":
            item["figure_ratio"] = (.4,.5,.6,.7)[k%4]
        if part == "wave":
            item["wave_length_mm"] = (20,30,40,50)[k%4]
    if kind == "helix":
        components[0]["turns"] = 1+version
    if kind == "arc_line_arc":
        components[-1]["direction"] = direction * (1 if version == 0 else -1)
    if kind in ("circle_helix", "helix_circle", "arc_line_arc", "orbit_slide", "ellipse_wave", "arc_helix_overlap"):
        ratios = {"circle_helix": (.5,.65,.8,.95), "helix_circle": (.65,.8,1,1.2),
                  "arc_line_arc": (.5,.75,1,1.25), "orbit_slide": (.5,.75,1,1.25),
                  "ellipse_wave": (.2,.3,.4,.5), "arc_helix_overlap": (.4,.55,.7,.85)}
        components[-1]["radius_or_scale_mm"] = size*ratios[kind][index % 4]
    if kind in NOVEL:
        sizes = {"circle_line_circle": ((26,45,19.5),(36,60,45)),
                 "ellipse_helix": ((30,20),(40,26)),
                 "figure_eight_slide": ((24,20),(33,35)),
                 "zigzag_delayed_lift": ((48,20),(66,40))}[kind][version]
        for item, radius in zip(components, sizes):
            item["radius_or_scale_mm"] = radius
        if kind == "ellipse_helix":
            components[0]["ellipse_ratio"] = (.6,.75)[version]
            components[1]["rise_mm"] = (30,45)[version]
        if kind == "figure_eight_slide":
            components[1]["axis_quarter_turns"] = (components[0]["axis_quarter_turns"]+1) % 4
            components[1]["plane"] = plane
        if kind == "zigzag_delayed_lift":
            components[1]["direction"] = 1
    windows = timings(kind, mode, version if kind in NOVEL else index, rng)
    windows = windows or [(0, 1)]*len(kinds)
    style = ("ramp", "smooth")[version if kind not in ("helix", "arc_line_arc") else index % 2]

    def sample(count):
        t = np.linspace(0, 1, count)
        path = np.zeros((count, 3))
        labels = []
        for item, (a,b) in zip(components, windows):
            u = profile(np.clip((t-a)/(b-a), 0, 1), style)
            delta, label = displacement(u, item)
            path += delta
            labels.append(f"{label} from {a:.0%} to {b:.0%}")
        return path, labels

    dense, _ = sample(1025)
    # Measure the largest point-to-point distance in the complete motion,
    # including simultaneous components. Preserve shape and component ratios.
    probes = dense[::16]
    diameter = np.linalg.norm(probes[:, None] - probes[None, :], axis=-1).max()
    nominal_sizes = sorted(set(PRIMARY[kinds[0]]) | set(TEST[kinds[0]]))
    rank = np.interp(components[0]["radius_or_scale_mm"], nominal_sizes,
                     np.linspace(100, 200, len(nominal_sizes)))
    scale = rank / diameter
    for item in components:
        for field in ("radius_or_scale_mm", "rise_mm", "height_mm", "width_mm", "wave_length_mm"):
            if field in item:
                item[field] *= scale
    dense, _ = sample(1025)
    count = int(np.ceil(np.linalg.norm(np.diff(dense, axis=0), axis=1).max()*1024*20/speed))+1
    path, labels = sample(max(16, count))
    low, high = np.array([290, -260, 95]), np.array([710, 260, 470])
    requested_start = start
    start = np.clip(start, low - path.min(axis=0), high - path.max(axis=0))
    path += start
    if not ((path >= low - 1e-6) & (path <= high + 1e-6)).all():
        raise ValueError(f"Workspace exceeded for {kind}, index {index}")
    metadata = dict(id=f"{VERSION}-{split}-{kind}-{index:04d}", split=split, kind=kind,
                    segments=list(kinds), size_mm=float(rank), frame="panda_base", start_xyz_mm=list(start),
                    requested_start_xyz_mm=list(requested_start), shape_scale=float(scale),
                    composition=mode, components=components, component_windows=windows,
                    phase_profile=style, target_speed_mm_s=speed, motion_samples=len(path),
                    peak_target_speed_mm_s=float(np.linalg.norm(np.diff(path,axis=0),axis=1).max()*20),
                    language=f"From the current position, {mode} motion: " + ", ".join(labels)
                    + f". Follow the {style} speed profile with a peak of {speed} mm/s.")
    return path.astype(np.float32)/1000, metadata


def examples(split, seed, pilot=0):
    motion.PLANES.update(PLANES)
    rng = np.random.default_rng(seed + {"train":0,"validation":1,"test":2}[split])
    starts = [STARTS[i] for i in rng.permutation(len(STARTS))]
    entries = []
    generated = 0
    kinds = (*motion.BASE, *motion.COMPOUNDS, *(NOVEL if split=="test" else ()))
    for kind in kinds:
        primary = (motion.COMPOUNDS | NOVEL).get(kind, ("single",(kind,)))[1][0]
        train_sizes = PRIMARY[primary]
        if kind=="orbit_slide": train_sizes=(20,26,32,40)
        candidates = []
        for group, (plane, direction) in enumerate((p,d) for p in PLANES for d in (1,-1)):
            if split=="train" or kind in NOVEL:
                sizes = (train_sizes[0], train_sizes[-1]) if kind in NOVEL else train_sizes
                candidates.extend((plane,direction,speed,size,v) for speed in SPEEDS for size in sizes
                                  for v in ((0,) if kind in NOVEL else (0,1)))
            else:
                sizes = TEST[primary] if split=="test" else train_sizes[4*(group%2):4*(group%2)+4] if len(train_sizes)==8 else train_sizes
                candidates.extend((plane,direction,SPEEDS[(group+i)%3],size,(group+i)%2)
                                  for i,size in enumerate(sizes))
        selected = set(np.linspace(0, len(candidates)-1, min(pilot, len(candidates)), dtype=int)) if pilot else None
        for i,(plane,direction,speed,size,version) in enumerate(candidates):
            if kind in NOVEL: version=i%2
            start = starts[generated % len(starts)]
            entry = make_entry(split,kind,plane,direction,speed,size,i,version,start,rng)
            generated += 1
            if selected is None or i in selected:
                entries.append(entry)
    return entries

