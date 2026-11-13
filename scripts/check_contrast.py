#!/usr/bin/env python3
"""WCAG contrast check of a site's theme tokens (sites/<site>/theme.css).

  python3 scripts/check_contrast.py --site olga
  python3 scripts/check_contrast.py --site sveta

Text tokens must reach 4.5:1 (WCAG AA, normal text) on every surface they
sit on; the focus ring (2px outline in --accent, see :focus-visible in
index.html) must reach 3:1 against the page background. Exit 1 on any
failure. Run it after every theme change, before building.
Standard library only.
"""
import argparse, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (foreground token, background tokens, minimum ratio, what it is)
CHECKS = [
    ('--text',           ['--bg', '--surface', '--paper'], 4.5, 'body text'),
    ('--text-secondary', ['--bg', '--surface', '--paper'], 4.5, 'secondary text'),
    ('--text-muted',     ['--bg', '--surface', '--paper'], 4.5, 'notes, captions, counts'),
    ('--text-faint',     ['--bg', '--surface', '--paper'], 4.5, 'faint labels'),
    ('--accent',         ['--bg', '--surface', '--paper'], 4.5, 'links and buttons'),
    ('--accent',         ['--bg', '--surface'],            3.0, 'focus ring (non-text)'),
    ('--paper',          ['--accent'],                     4.5, 'text on accent (selection)'),
]


def tokens(css):
    """Resolve #hex tokens, following var(--x) aliases."""
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    raw = dict(re.findall(r'(--[\w-]+)\s*:\s*([^;]+);', css))
    def resolve(v, depth=0):
        v = v.strip()
        m = re.fullmatch(r'var\((--[\w-]+)\)', v)
        return resolve(raw[m.group(1)], depth + 1) if m and depth < 5 else v
    return {k: resolve(v) for k, v in raw.items()}


def rgb(h):
    h = h.lstrip('#')
    if len(h) == 3:
        h = ''.join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def luminance(h):
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb(h)]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--site', required=True)
    args = ap.parse_args()
    t = tokens((REPO / 'sites' / args.site / 'theme.css').read_text())
    fails = 0
    for fg, bgs, need, what in CHECKS:
        for bg in bgs:
            a, b = t.get(fg, ''), t.get(bg, '')
            if not (a.startswith('#') and b.startswith('#')):
                print(f'  FAIL  {fg} on {bg}: not a plain #hex colour ({a!r} / {b!r})')
                fails += 1
                continue
            r = ratio(a, b)
            ok = r >= need
            fails += not ok
            print(f'  {"ok  " if ok else "FAIL"}  {r:5.2f} ≥ {need}  {fg} {a} on {bg} {b}  ({what})')
    print(f'{args.site}: ' + ('all pass' if not fails else f'{fails} below the minimum'))
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
