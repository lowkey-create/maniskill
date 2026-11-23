#!/usr/bin/env python3
"""Generate diverse, centered elementary outline shapes as raw SVG assets.

They are shape *assets*, not instruction-conditioned placement/order tasks.
Use a separate task-synthesis step after normalization to vary location/scale.
"""
import argparse
import math
import random
from pathlib import Path


def closed_polygon(n, radius, rotation, cx=128, cy=128, aspect=1):
    pts = [(cx + radius*math.cos(rotation+2*math.pi*i/n)*aspect,
            cy + radius*math.sin(rotation+2*math.pi*i/n)) for i in range(n)]
    return [pts + pts[:1]]


def shape_strokes(name, rng):
    rot = rng.uniform(-math.pi,math.pi)
    rad = rng.uniform(65,92)
    if name=='circle':
        n=64
        aspect=rng.uniform(.87,1.13)
        return closed_polygon(n,rad,rot,aspect=aspect)
    if name in {'square','triangle','pentagon'}:
        return closed_polygon({'square':4,'triangle':3,'pentagon':5}[name],rad,rot)
    if name=='star':
        pts=[]
        inner=rad*rng.uniform(.35,.55)
        for i in range(10):
            ang=rot+math.pi*i/5
            r=rad if i%2==0 else inner
            pts.append((128+r*math.cos(ang),128+r*math.sin(ang)))
        return [pts+pts[:1]]
    if name=='spiral':
        turns=rng.uniform(1.5,3.7)
        pts=[]
        for i in range(101):
            t=i/100
            ang=rot+turns*2*math.pi*t
            r=5+(rad-5)*t
            pts.append((128+r*math.cos(ang),128+r*math.sin(ang)))
        return [pts]
    if name=='wave':
        cycles=rng.randint(2,5)
        amp=rng.uniform(20,46)
        return [[(36+184*i/99,128+amp*math.sin(rot+cycles*2*math.pi*i/99)) for i in range(100)]]
    if name=='zigzag':
        bends=rng.randint(5,14)
        amp=rng.uniform(27,68)
        return [[(36+184*i/bends,128 + amp*(-1 if i%2 else 1)) for i in range(bends+1)]]
    if name=='heart':
        # 16 sin^3 t, 13cos t - 5cos 2t - 2cos 3t - cos 4t
        pts=[]
        for i in range(101):
            t=2*math.pi*i/100
            x=16*math.sin(t)**3
            y=13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
            pts.append((128+rad*x/18, 128-rad*y/18))
        return [pts]
    if name=='cross':
        c=128
        return [[(c-rad,c),(c+rad,c)],[(c,c-rad),(c,c+rad)]]
    raise ValueError(name)


def serialize(strokes):
    d=[]
    for stroke in strokes:
        if len(stroke)<2:
            continue
        d.append('M '+' '.join(('L ' if i else '') + f'{x:.3f},{y:.3f}' for i,(x,y) in enumerate(stroke)))
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" '
            'viewBox="0 0 256 256" fill="none" stroke="black" '
            'stroke-width="2" stroke-linecap="round">\n'
            '<path d="'+' '.join(d)+'"/>\n</svg>\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path.home()/'svg_vla_bench')
    p.add_argument('--per-class',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    rng=random.Random(args.seed)
    names=['circle','square','triangle','pentagon','star','spiral','wave','zigzag','heart','cross']
    for name in names:
        target=args.root/'raw'/'geometry'/name
        target.mkdir(parents=True,exist_ok=True)
        for i in range(args.per_class):
            (target/f'{name}_{i:04d}.svg').write_text(
                serialize(shape_strokes(name,rng)),encoding='utf-8')
        print(name,args.per_class)
    print('Saved',args.per_class*len(names),'raw geometric SVGs')


if __name__=='__main__':
    main()
