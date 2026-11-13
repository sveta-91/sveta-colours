#!/usr/bin/env python3
"""Build one artist's site from the shared index.html (docs/olgazai-plan.md).

index.html is shared code plus one block between <!-- SITE:BEGIN --> and
<!-- SITE:END --> that holds everything belonging to one artist. The block is
rendered from sites/<name>/site.json (head tags + the SITE object) and
sites/<name>/theme.css (colour and font tokens).

  build_site.py sveta --root      rewrite the block in the root index.html
                                  (GitHub Pages serves the root as is)
  build_site.py olga              write dist/ for Cloudflare Pages
  build_site.py olga --strict     the same, but fail while a required link
                                  is still empty (the Cloudflare build uses this)
  build_site.py --check           fail if the root block is not what
                                  sites/sveta renders (run before committing)

Options for local testing:
  --out DIR        output folder (default dist/)
  --dev-csv FILE   copy this CSV into the build and read paintings from it

Standard library only, so the Cloudflare build image needs no installs.
"""
import argparse, html, json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BEGIN, END = '<!-- SITE:BEGIN', '<!-- SITE:END -->'
ROOT_SITE = 'sveta'          # the site the root index.html carries
# A real deploy needs these; an empty one means the artist hasn't sent it yet.
REQUIRED = ['contactEmail', 'sheetCsvUrl', 'formEndpoint']


def a(v):
    """Attribute-safe text (keeps & entities the source already uses)."""
    return html.escape(str(v), quote=True).replace('&amp;', '&')


def render_block(cfg, theme):
    h = cfg['head']
    icons = '/' + cfg['files'].get('icons', 'icons').strip('/') + '/'
    og_image = h.get('ogImage')
    lines = [
        '<!-- SITE:BEGIN ===========================================================',
        '     Everything that belongs to one artist lives between these markers:',
        '     identity meta, fonts, theme tokens and the SITE object. The rest of',
        '     the page is shared code. Do not edit this block by hand: it is',
        '     rendered from sites/<site>/ by scripts/build_site.py (the root',
        '     index.html carries sites/sveta, so GitHub Pages needs no build).',
        '     ===================================================================== -->',
        f'<title>{a(h["title"])}</title>',
        '',
        '<!-- SEO -->',
        f'<meta name="description" content="{a(h["description"])}">',
        f'<meta name="author" content="{a(h["author"])}">',
        '',
        '<!-- Open Graph -->',
        f'<meta property="og:title" content="{a(h["ogTitle"])}">',
        f'<meta property="og:description" content="{a(h["ogDescription"])}">',
        '<meta property="og:type" content="website">',
        f'<meta property="og:url" content="{a(h["url"])}">',
        f'<meta property="og:site_name" content="{a(h["siteName"])}">',
        f'<link rel="canonical" href="{a(h["url"])}">',
    ]
    if og_image:
        lines += [
            '<!-- Static on purpose: messenger and social crawlers don\'t run JS. -->',
            f'<meta property="og:image" content="{a(og_image)}">',
            f'<meta property="og:image:width" content="{h["ogImageWidth"]}">',
            f'<meta property="og:image:height" content="{h["ogImageHeight"]}">',
            f'<meta property="og:image:alt" content="{a(h["ogImageAlt"])}">',
        ]
    lines += [
        '',
        '<!-- Twitter Card -->',
        f'<meta name="twitter:card" content="{"summary_large_image" if og_image else "summary"}">',
        f'<meta name="twitter:title" content="{a(h["twitterTitle"])}">',
        f'<meta name="twitter:description" content="{a(h["twitterDescription"])}">',
    ]
    if og_image:
        lines.append(f'<meta name="twitter:image" content="{a(og_image)}">')
    lines += [
        '',
        f'<link rel="icon" type="image/svg+xml" href="{icons}favicon.svg">',
        f'<link rel="icon" type="image/png" sizes="32x32" href="{icons}favicon-32.png">',
        f'<link rel="icon" type="image/png" sizes="16x16" href="{icons}favicon-16.png">',
        f'<link rel="apple-touch-icon" sizes="180x180" href="{icons}apple-touch-icon.png">',
        '',
        '<!-- Fonts -->',
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        f'<link rel="preload" as="style" href="{a(h["fontsHref"])}">',
        f'<link href="{a(h["fontsHref"])}" rel="stylesheet">',
        '',
        '<!-- JSON-LD: injected per painting by JS; this is a fallback for the site itself -->',
        '<script type="application/ld+json">',
        json.dumps(h['jsonLd'], indent=2, ensure_ascii=False),
        '</script>',
        '',
        '<style id="site-theme">',
        theme.rstrip('\n'),
        '</style>',
        '',
        '<script>',
        'const SITE = ' + json.dumps(cfg['site'], indent=2, ensure_ascii=False).replace('</', '<\\/') + ';',
        '</script>',
        END,
    ]
    return '\n'.join(lines) + '\n'


def swap_block(page, block):
    i, j = page.index(BEGIN), page.index(END)
    return page[:i] + block + page[j + len(END) + 1:]


def fill_static(page, site):
    """Put SITE values into the static HTML that carries data-site-* markers,
    so crawlers and no-JS readers see this artist's text too."""
    strings = site.get('strings', {})
    look = lambda k: site.get(k) if isinstance(site.get(k), str) else strings.get(k, '')
    page = re.sub(r'(data-site-text="(\w+)">)[^<]*(<)',
                  lambda m: m.group(1) + html.escape(look(m.group(2)), quote=False) + m.group(3), page)
    page = re.sub(r'(data-site-html="(\w+)">).*?(</)',
                  lambda m: m.group(1) + strings.get(m.group(2), '') + m.group(3), page, flags=re.S)
    page = re.sub(r'aria-label="[^"]*"( data-site-aria="(\w+)")',
                  lambda m: f'aria-label="{a(strings.get(m.group(2), ""))}"' + m.group(1), page)
    return page


def load(name):
    d = REPO / 'sites' / name
    cfg = json.loads((d / 'site.json').read_text())
    return cfg, (d / 'theme.css').read_text(), d


def missing(cfg):
    s = cfg['site']
    return [k for k in REQUIRED if not s.get(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('site', nargs='?')
    ap.add_argument('--root', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--out', default=str(REPO / 'dist'))
    ap.add_argument('--dev-csv')
    args = ap.parse_args()

    shared = (REPO / 'index.html').read_text()

    if args.check:
        cfg, theme, _ = load(ROOT_SITE)
        want = fill_static(swap_block(shared, render_block(cfg, theme)), cfg['site'])
        if want != shared:
            sys.exit(f'index.html block differs from sites/{ROOT_SITE}: run scripts/build_site.py {ROOT_SITE} --root')
        print(f'ok: index.html carries sites/{ROOT_SITE}')
        return

    if not args.site:
        ap.error('name the site (sveta, olga, …); there is no default')
    if not (REPO / 'sites' / args.site / 'site.json').exists():
        ap.error(f'no sites/{args.site}/site.json')
    cfg, theme, src = load(args.site)

    if args.root:
        if args.site != ROOT_SITE:
            ap.error(f'only sites/{ROOT_SITE} goes into the root index.html')
        (REPO / 'index.html').write_text(fill_static(swap_block(shared, render_block(cfg, theme)), cfg['site']))
        print(f'index.html block rewritten from sites/{ROOT_SITE}')
        return

    out = Path(args.out)
    if args.dev_csv:
        cfg['site']['sheetCsvUrl'] = 'data/paintings.csv'
    gaps = missing(cfg)
    if gaps:
        msg = f'sites/{args.site}: still empty: {", ".join(gaps)}'
        if args.strict:
            sys.exit('refusing to build: ' + msg)
        print('warning: ' + msg)

    page = fill_static(swap_block(shared, render_block(cfg, theme)), cfg['site'])
    if args.site != ROOT_SITE:
        # Source notes in HTML comments are the root artist's working notes;
        # another artist's page source shouldn't carry them.
        page = re.sub(r'<!--(?!\s*SITE:).*?-->', '', page, flags=re.S)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / 'index.html').write_text(page)
    for key, path in cfg['files'].items():
        p = REPO / path
        dest = out / ('favicon.ico' if key == 'favicon' else key)
        if p.is_dir():
            shutil.copytree(p, dest)
        elif p.is_file():
            shutil.copy2(p, dest)
        else:
            print(f'warning: files.{key}: {path} not found')
    if args.dev_csv:
        (out / 'data').mkdir()
        shutil.copy2(args.dev_csv, out / 'data' / 'paintings.csv')
    print(f'built sites/{args.site} → {out}')


if __name__ == '__main__':
    main()
