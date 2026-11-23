#!/usr/bin/env python3
"""Flatten safe outline SVGs into one M/L-only path; write CSV manifests.

This purposely *rejects* SVGs involving transforms, <use>, text, fills, clips,
images, masks, and CSS-dependent geometry instead of silently changing the
visible result. For large-scale use add a tested full SVG renderer/vectorizer.
"""
import argparse
import csv
import hashlib
import io
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

from svgpathtools import svg2paths2

SVG_NS = 'http://www.w3.org/2000/svg'
SHAPES = {'path','rect','circle','ellipse','line','polyline','polygon'}
ALLOWED_WRAPPERS = {'svg','g','title','desc','metadata'}
UNSUPPORTED = {'use','text','textPath','image','foreignObject','symbol',
               'clipPath','mask','filter','pattern','linearGradient',
               'radialGradient','marker','style','script'}
GEOM_ATTRS = {'d','x','y','width','height','rx','ry','cx','cy','r',
              'x1','x2','y1','y2','points'}


def local(tag):
    return tag.rsplit('}', 1)[-1]


def css_attrs(style):
    obj = {}
    for chunk in style.split(';'):
        if ':' in chunk:
            k, v = chunk.split(':', 1)
            obj[k.strip()] = v.strip()
    return obj


def extract_paths(source):
    if source.stat().st_size > 1024 * 1024:
        raise ValueError('oversize-svg')
    raw = source.read_text(encoding='utf-8')
    if '<!DOCTYPE' in raw.upper() or '<!ENTITY' in raw.upper():
        raise ValueError('doctype-or-entity')
    original = ET.fromstring(raw)
    if local(original.tag) != 'svg':
        raise ValueError('not-svg-root')
    root = ET.Element('svg', {'xmlns': SVG_NS})
    count = 0

    def walk(node, inherited, transformed=False):
        nonlocal count
        tag = local(node.tag)
        if tag in UNSUPPORTED:
            # Clip/mask/use/text would require a full SVG rendering engine.
            if tag in {'use','text','textPath','image','foreignObject',
                       'clipPath','mask','filter','pattern','symbol'}:
                raise ValueError('unsupported-' + tag)
            return
        if tag not in SHAPES and tag not in ALLOWED_WRAPPERS:
            raise ValueError('unsupported-tag-' + tag)
        attrs = dict(inherited)
        for key in ('fill','stroke','display','visibility','opacity','clip-path','mask'):
            if key in node.attrib:
                attrs[key] = node.attrib[key]
        attrs.update(css_attrs(node.attrib.get('style','')))
        if attrs.get('display') == 'none' or attrs.get('visibility') == 'hidden':
            return
        if float(attrs.get('opacity','1')) == 0:
            return
        transformed = transformed or ('transform' in node.attrib)
        if tag in SHAPES:
            if transformed:
                raise ValueError('transform-needs-preprocessing')
            if attrs.get('clip-path','none') != 'none' or attrs.get('mask','none') != 'none':
                raise ValueError('clip-or-mask')
            # For a pen trajectory we only accept visible outlines.
            if attrs.get('stroke','none').strip().lower() == 'none':
                return
            if attrs.get('fill','black').strip().lower() != 'none':
                raise ValueError('filled-shape')
            geom = {k:v for k,v in node.attrib.items() if local(k) in GEOM_ATTRS}
            ET.SubElement(root, tag, geom)
            count += 1
        else:
            for child in node:
                walk(child, attrs, transformed)
    walk(original, {'stroke':'none','fill':'black'})
    if count == 0:
        raise ValueError('no-visible-outline')
    # Convert elements *in original DOM order*: svg2paths2 on a whole
    # document would group all paths first, then circles, lines, etc.
    paths = []
    for child in root:
        one = ET.Element('svg', {'xmlns': SVG_NS})
        one.append(child)
        one_xml = ET.tostring(one, encoding='unicode')
        converted, _, _ = svg2paths2(io.StringIO(one_xml))
        paths.extend(converted)
    if not paths:
        raise ValueError('no-convertible-path')
    return paths


def approx_length(segment):
    try:
        val = float(segment.length(error=1e-3))
        if math.isfinite(val):
            return val
    except (TypeError, AttributeError, ValueError):
        pass
    pts = [complex(segment.point(k/32)) for k in range(33)]
    return sum(abs(b-a) for a,b in zip(pts, pts[1:]))


def flatten(paths, canvas, margin, sample_step):
    boxes = [p.bbox() for p in paths if len(p)]
    if not boxes:
        raise ValueError('empty-geometry')
    xmin, xmax = min(b[0] for b in boxes), max(b[1] for b in boxes)
    ymin, ymax = min(b[2] for b in boxes), max(b[3] for b in boxes)
    if not all(math.isfinite(x) for x in [xmin,xmax,ymin,ymax]):
        raise ValueError('nonfinite-bounds')
    w, h = xmax - xmin, ymax - ymin
    if max(w,h) < 1e-8:
        raise ValueError('point-only')
    scale = (canvas - 2*margin) / max(w,h)
    dx, dy = (canvas - w*scale)/2, (canvas - h*scale)/2

    def mapped(z):
        z = complex(z)
        return ((z.real-xmin)*scale+dx, (z.imag-ymin)*scale+dy)

    strokes = []
    for path in paths:
        current, prev = [], None
        for seg in path:
            a, b = seg.start, seg.end
            if a is None or b is None:
                continue
            if not all(math.isfinite(v) for v in (a.real,a.imag,b.real,b.imag)):
                raise ValueError('nonfinite-segment')
            if prev is None or abs(a-prev) > 1e-6:
                if len(current) > 1:
                    strokes.append(current)
                current = [mapped(a)]
            # A straight segment requires only one endpoint; curves are sampled.
            if type(seg).__name__ in ('Line','Close'):
                n = 1
            else:
                n = min(512,max(2,int(math.ceil(approx_length(seg)*scale/sample_step))))
            for i in range(1,n+1):
                xy = mapped(seg.point(i/n))
                if math.dist(current[-1],xy) > 1e-5:
                    current.append(xy)
            prev = b
        if len(current) > 1:
            strokes.append(current)
    if not strokes:
        raise ValueError('no-strokes')
    length = sum(math.dist(a,b) for s in strokes for a,b in zip(s,s[1:]))
    vertices = sum(len(s) for s in strokes)
    return strokes, vertices, length


def write_outline(out_path, strokes, canvas):
    d = []
    for s in strokes:
        d.append(f'M {s[0][0]:.3f},{s[0][1]:.3f}')
        d.extend(f'L {x:.3f},{y:.3f}' for x,y in s[1:])
    d = ' '.join(d)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    out_path.write_text(
        f'<svg xmlns="{SVG_NS}" width="{canvas}" height="{canvas}" '
        f'viewBox="0 0 {canvas} {canvas}" fill="none" stroke="black" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">\n'
        f'  <path d="{escape(d)}"/>\n</svg>\n', encoding='utf-8')
    return hashlib.sha256(d.encode('utf-8')).hexdigest()


def split(key):
    v = int(hashlib.sha256(('vla-bench-2026|'+key).encode()).hexdigest()[:8],16) / 0x100000000
    return 'train' if v < .8 else ('val' if v < .9 else 'test')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.home()/'svg_vla_bench')
    parser.add_argument('--canvas', type=int, default=256)
    parser.add_argument('--margin', type=float, default=12)
    parser.add_argument('--curve-step', type=float, default=2)
    parser.add_argument('--max-strokes', type=int, default=40)
    parser.add_argument('--max-vertices', type=int, default=2000)
    parser.add_argument('--max-length', type=float, default=6000)
    args = parser.parse_args()
    root = args.root
    if args.canvas <= 2*args.margin or args.curve_step <= 0:
        parser.error('canvas must exceed 2 * margin; curve-step must be positive')
    src_root = root/'raw'
    files = sorted(src_root.rglob('*.svg'))
    if not files:
        parser.error(f'no input SVG found under {src_root}')
    processed, rejects = [], []
    seen_shapes = set()
    for f in files:
        relative = f.relative_to(src_root)
        try:
            if relative.parts[0] not in {'lucide','tabler','iconoir','quickdraw','geometry'}:
                raise ValueError('unrecognized-source')
            paths = extract_paths(f)
            strokes, vertices, length = flatten(paths,args.canvas,args.margin,args.curve_step)
            n_strokes = len(strokes)
            if n_strokes > args.max_strokes:
                raise ValueError('too-many-strokes')
            if vertices > args.max_vertices:
                raise ValueError('too-many-vertices')
            if length > args.max_length:
                raise ValueError('too-long')
            # SHA based on normalized strokes, exact duplicates (not perceptual).
            normalized = '|'.join(','.join(f'{x:.2f}:{y:.2f}' for x,y in s) for s in strokes)
            geom_hash = hashlib.sha256(normalized.encode()).hexdigest()
            if geom_hash in seen_shapes:
                raise ValueError('exact-normalized-duplicate')
            seen_shapes.add(geom_hash)
            source = relative.parts[0]
            label = relative.parts[1].replace('-', ' ') if source in {'quickdraw','geometry'} else f.stem.replace('-',' ')
            group = re.sub(r'\s+',' ',label.lower()).strip()
            difficulty = ('easy' if n_strokes<=3 and vertices<=100 else
                          'medium' if n_strokes<=12 and vertices<=400 else 'hard')
            destination = root/'clean'/relative
            sha = write_outline(destination,strokes,args.canvas)
            processed.append(dict(
                file=str(destination.relative_to(root)), raw_file=str(f.relative_to(root)),
                source=source, label=label, stroke_count=n_strokes,
                vertex_count=vertices, length_canvas_px=f'{length:.2f}',
                difficulty=difficulty, normalized_hash=geom_hash,
                iid_split=split(str(relative)), class_ood_split=split(group),
            ))
        except Exception as exc:
            rejects.append(dict(raw_file=str(f.relative_to(root)), reason=str(exc)))
    root.mkdir(parents=True,exist_ok=True)
    for filename, rows, fields in [
        ('manifest.csv', processed, ['file','raw_file','source','label','stroke_count',
            'vertex_count','length_canvas_px','difficulty','normalized_hash','iid_split','class_ood_split']),
        ('rejected.csv', rejects, ['raw_file','reason'])]:
        with (root/filename).open('w',newline='',encoding='utf-8') as out:
            writer=csv.DictWriter(out,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
    print(f'Processed: {len(processed)}, rejected: {len(rejects)}')
    print('Clean SVG:',root/'clean')
    print('Manifest:',root/'manifest.csv','Rejected:',root/'rejected.csv')
    print('IMPORTANT: iid_split is NOT near-duplicate-safe; class_ood_split groups only identical labels.')


if __name__=='__main__':
    main()
