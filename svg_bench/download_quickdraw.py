#!/usr/bin/env python3
"""Sample recognized vector strokes from Quick, Draw!'s simplified NDJSON.

Uses streaming reservoir sampling in the first --scan-limit *recognized* rows,
so it does NOT download entire 50M-drawing dataset or claim global randomness.
"""
import argparse
import json
import random
import re
from pathlib import Path
from urllib.parse import quote

import requests

DEFAULT_CATEGORIES = [
    'cat', 'dog', 'bird', 'fish', 'butterfly', 'flower', 'tree',
    'house', 'bicycle', 'car', 'airplane', 'boat', 'cup',
    'chair', 'table', 'lamp', 'clock', 'umbrella', 'apple',
    'mushroom', 'star', 'circle', 'square', 'triangle', 'sun',
]


def slugify(s):
    return re.sub(r'[^a-z0-9]+', '-', s.strip().lower()).strip('-')


def to_svg(drawing):
    chunks = []
    for stroke in drawing:
        if not isinstance(stroke, list) or len(stroke) < 2:
            continue
        xs, ys = stroke[:2]
        if not xs or len(xs) != len(ys):
            continue
        pairs = [(float(x), float(y)) for x, y in zip(xs, ys)]
        if len(pairs) < 2:
            continue
        chunks.append('M ' + ' '.join(
            f'{"L " if i else ""}{x:.2f},{y:.2f}' for i, (x, y) in enumerate(pairs)
        ))
    if not chunks:
        return None
    d = ' '.join(chunks)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" '
            'width="256" height="256" fill="none" stroke="black" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">\n'
            f'<path d="{d}"/>\n</svg>\n')


def sample_category(session, category, count, scan_limit, seed):
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/simplified/' + quote(category, safe='') + '.ndjson'
    rng = random.Random(seed)
    seen, reservoir = 0, []
    with session.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if not data.get('recognized', False):
                continue
            if not data.get('drawing'):
                continue
            seen += 1
            if len(reservoir) < count:
                reservoir.append(data)
            else:
                j = rng.randrange(seen)
                if j < count:
                    reservoir[j] = data
            if seen >= scan_limit:
                break
    return reservoir, seen, url


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path.home() / 'svg_vla_bench')
    p.add_argument('--categories', default=','.join(DEFAULT_CATEGORIES),
                   help='Comma-separated Quick, Draw! category names')
    p.add_argument('--per-category', type=int, default=80)
    p.add_argument('--scan-limit', type=int, default=1500)
    p.add_argument('--seed', type=int, default=2026)
    a = p.parse_args()
    if a.per_category < 1 or a.scan_limit < a.per_category:
        p.error('--scan-limit must be >= --per-category >= 1')
    categories = list(dict.fromkeys(x.strip() for x in a.categories.split(',') if x.strip()))
    out = a.root / 'raw' / 'quickdraw'
    out.mkdir(parents=True, exist_ok=True)
    provenance = a.root / 'quickdraw_sources.jsonl'
    with requests.Session() as session, provenance.open('w', encoding='utf-8') as manifest:
        for i, cat in enumerate(categories):
            try:
                items, inspected, url = sample_category(
                    session, cat, a.per_category, a.scan_limit, a.seed + i)
                target = out / slugify(cat)
                target.mkdir(parents=True, exist_ok=True)
                saved = 0
                for j, item in enumerate(items):
                    svg = to_svg(item['drawing'])
                    if not svg:
                        continue
                    key_id = re.sub(r'[^A-Za-z0-9_-]', '', str(item.get('key_id', j)))
                    path = target / f'{slugify(cat)}_{key_id}.svg'
                    path.write_text(svg, encoding='utf-8')
                    manifest.write(json.dumps({
                        'file': str(path.relative_to(a.root)), 'category': cat,
                        'key_id': str(item.get('key_id', '')), 'source_url': url,
                        'sampled_from_first_recognized_rows': inspected,
                    }, ensure_ascii=False) + '\n')
                    saved += 1
                print(f'{cat}: saved {saved} drawings (sampled from first {inspected} recognized records)')
            except Exception as exc:
                print(f'[ERROR] {cat}: {exc}')
    print('Finished. Raw SVG:', out)


if __name__ == '__main__':
    main()
