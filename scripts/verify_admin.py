#!/usr/bin/env python3
"""Selenium checks for the admin page (admin/index.html) and the hidden contact email.

Local suites (default) serve the repo and a fresh `build_site.py olga` build on
free ports, map www.sveta-colours.com / olgazai.com onto them inside Chrome, and
stub every OpenAI call: no key, no cost, no Google writes.

  hosts   the domain picks the site; sign-in gate on live hosts only; noindex;
          key per tab on live hosts; voice files from GitHub off localhost
  prep    Straighten + crop on a synthetic framed, skewed 16x20 photo with known
          corners: corner order, zoom agreement and red flags, drag, 4:5 output,
          no frame/wall on the edges, JPEG named after the title, undo
  dialog  three options with Use n / title only / description only; numbered and
          **bold** labels; a revision gets one button; corrections.md entry
  form    Collection checkboxes (required, ":n" kept, Greeting cards → 5x7),
          Medium and Size dropdowns from the sheet's values plus Other…
  email   public pages: the address is not in the source or the DOM, links read
          "Email <site name>", mailto built on click, Copy email copies it

  --live  the deployed /admin/ pages on all four hosts (config, gate, new form,
          no console errors) and that Google's sign-in window opens without
          origin_mismatch (nothing is signed in)

Usage (from the repo root, with the image-pipeline venv for Selenium + Pillow):
  tools/image_pipeline/.venv/bin/python scripts/verify_admin.py
  tools/image_pipeline/.venv/bin/python scripts/verify_admin.py --only prep,form
  tools/image_pipeline/.venv/bin/python scripts/verify_admin.py --live
  ... --show       run Chrome with a window;  --shots DIR  save screenshots

Exit status 1 when any check fails.
"""
import argparse
import functools
import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'testdata'))
import make_skewed  # noqa: E402

SUITES = ['hosts', 'prep', 'dialog', 'form', 'email']
FAILS = []


def check(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  {info}' if info != '' else ''))
    if not cond:
        FAILS.append(name)


def serve(directory):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    handler = functools.partial(Quiet, directory=directory)
    srv = socketserver.ThreadingTCPServer(('127.0.0.1', 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def chrome(show, maps=''):
    o = Options()
    if not show:
        o.add_argument('--headless=new')
    o.add_argument('--window-size=1400,1000')
    if maps:
        o.add_argument('--host-resolver-rules=' + maps)
        o.add_argument('--disable-features=HttpsUpgrades,HttpsFirstBalancedModeAutoEnable,HttpsFirstModeV2ForTypicallySecureUsers')
    o.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
    return webdriver.Chrome(options=o)


def errors(d):
    """Severe console errors, minus Google sign-in library noise (it warns whenever a
    popup can't be checked) and missing favicons."""
    return [e['message'] for e in d.get_log('browser') if e['level'] == 'SEVERE'
            and 'accounts.google.com' not in e['message'] and 'favicon' not in e['message']]


def wait_for(d, js, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if d.execute_script('return ' + js):
            return True
        time.sleep(0.25)
    return False


def load_admin(d, base, photo=None):
    d.get(f'{base}/admin/?site=sveta')
    wait_for(d, '!!window.SITE_CFG && !!document.getElementById("fSizeSel")')
    if photo:
        d.find_element('id', 'fileInput').send_keys(photo)
        wait_for(d, '!!state.origImg')
        time.sleep(0.5)


def shot(d, ctx, name):
    if ctx['shots']:
        d.save_screenshot(os.path.join(ctx['shots'], name + '.png'))


# ── suites ────────────────────────────────────────────────────────────────

def suite_hosts(d, ctx):
    for label, url, live, site in [
            ('local sveta', f"http://localhost:{ctx['p_root']}/admin/?site=sveta", False, 'Sveta Colours'),
            ('live sveta', 'http://www.sveta-colours.com/admin/?site=olga', True, 'Sveta Colours'),
            ('live olga', 'http://olgazai.com/admin/', True, 'Olga Zai')]:
        d.get(url)
        wait_for(d, '!!window.SITE_CFG')
        time.sleep(0.5)
        i = d.execute_script("""const g=document.getElementById('signInGate'), bar=document.getElementById('siteBar');
          return {title:document.title, gate:!!g, bar:bar?bar.textContent:'', block:!!document.getElementById('siteBlock'),
                  ph:document.getElementById('aiKey').placeholder, voice:AI_VOICE_FILES,
                  robots:(document.querySelector('meta[name=robots]')||{}).content}""")
        check(f'hosts: {label} site', i['title'].startswith(site), i['title'])
        check(f'hosts: {label} no setup block', not i['block'])
        check(f'hosts: {label} gate only on live', i['gate'] == live)
        check(f'hosts: {label} switch link only local', ('switch site' in i['bar']) == (not live))
        check(f'hosts: {label} key lasts one tab on live', ('tab closes' in i['ph']) == live)
        check(f'hosts: {label} noindex', i['robots'] == 'noindex, nofollow')
        if i['voice']:
            check(f'hosts: {label} voice files from GitHub only when live',
                  i['voice'][0].startswith('https://raw.githubusercontent.com') == live)
        check(f'hosts: {label} no console errors', not errors(d), errors(d))


def suite_prep(d, ctx):
    truth = ctx['truth']
    W, H = truth['size']
    tc = truth['painted_corners']
    load_admin(d, f"http://localhost:{ctx['p_root']}", ctx['photo'])
    first = [[x + 40, y + 60] for x, y in tc]                 # a first guess off by 40/60 px
    shuffled = [first[2], first[0], first[3], first[1]]
    d.execute_script("""
      const T = %s, F = %s, W = %d, H = %d;
      localStorage.setItem('pm-openai-key', 'sk-test');
      $('fTitle').value = 'Skewed Test'; $('fSize').value = '16x20 in'; sizeRender();
      const rf = window.fetch; window._sent = null;
      window.fetch = async (u, opt) => {
        if (String(u).includes('api.openai.com')) { window._sent = JSON.parse(opt.body);
          return new Response(JSON.stringify({choices:[{message:{content: JSON.stringify({corners: F.map(([x,y]) => [x/W, y/H])})}}]}), {status:200}); }
        return rf(u, opt);
      };
      // zoom answers: corner 1 two agree, corner 2 all disagree, corners 3-4 exact
      const plan = {0: [[1,1],[2,-1],[80,0]], 1: [[80,0],[-90,0],null], 2: [[0,0],[0,0],[0,0]], 3: [[0,0],[0,0],[0,0]]};
      window._zoom = 0;
      window.prepZoomOnce = async (i, pt, frac) => { window._zoom++;
        const off = plan[i][[0.12,0.15,0.18].indexOf(frac)];
        return off ? [T[i][0] + off[0], T[i][1] + off[1]] : null; };
    """ % (json.dumps(tc), json.dumps(shuffled), W, H))
    d.find_element('id', 'prepFind').click()
    wait_for(d, '!!(state.prep && state.prep.flags)')
    r = d.execute_script("return {c: state.prep.corners, f: state.prep.flags, n: window._zoom, st: $('prepStatus').textContent, s: window._sent}")
    s = r['s'] or {}
    check('prep: corner request is JSON with a high-detail image',
          s.get('response_format', {}).get('type') == 'json_object'
          and s['messages'][1]['content'][1]['image_url']['detail'] == 'high')
    err = [round(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5, 1) for a, b in zip(r['c'], tc)]
    check('prep: 12 zoom calls (4 corners x 3 crops)', r['n'] == 12, r['n'])
    check('prep: agreeing pair averaged, exact kept', err[0] < 2 and err[2] < 0.5 and err[3] < 0.5, err)
    check('prep: disagreeing corner keeps the first guess', abs(r['c'][1][0] - first[1][0]) < 0.5, r['c'][1])
    check('prep: only that corner is red', r['f'] == [False, True, False, False], r['f'])
    check('prep: status names the red corner', '1 corner in red' in r['st'], r['st'])
    shot(d, ctx, 'prep_flagged')
    # drag corner 1 (index 1, the red one) a little: it moves and turns yellow
    rect = d.execute_script('const r=canvas.getBoundingClientRect(); return {w:r.width,h:r.height,fw:state.fullCanvas.width}')
    k = rect['w'] / rect['fw']
    c1 = r['c'][1]
    el = d.find_element('id', 'preview-canvas')
    ActionChains(d).move_to_element_with_offset(el, c1[0] * k - rect['w'] / 2, c1[1] * k - rect['h'] / 2) \
        .click_and_hold().move_by_offset(-10, 8).perform()
    time.sleep(0.2)
    shot(d, ctx, 'prep_dragging')
    ActionChains(d).release().perform()
    moved = d.execute_script('return [state.prep.corners[1], state.prep.flags[1]]')
    check('prep: drag moves the corner and clears red', abs(moved[0][0] - c1[0]) > 5 and moved[1] is False, moved)
    d.execute_script('state.prep.corners = %s; drawPreview();' % json.dumps(tc))
    d.find_element('id', 'prepApply').click()
    wait_for(d, '!state.prep', 30)
    res = d.execute_script("""
      const c=state.fullCanvas, x=c.getContext('2d'), w=c.width, h=c.height;
      const px=(a,b)=>Array.from(x.getImageData(a,b,1,1).data.slice(0,3));
      return {w, h, name: state.file.name, type: state.file.type, tl: px(30,30), br: px(w-30,h-30),
        edges: [px(w>>1,3), px(3,h>>1), px(w-4,h>>1), px(w>>1,h-4)], aspect: $('aspectInfo').textContent,
        undo: !$('prepUndoRow').hidden}""")
    check('prep: output is 4:5', abs(res['w'] / res['h'] - 0.8) < 0.002, f"{res['w']}x{res['h']}")
    check('prep: red mark top-left, green bottom-right',
          res['tl'][0] > 180 and res['tl'][1] < 90 and res['br'][1] > 120 and res['br'][0] < 90)
    brown = lambda p: abs(p[0] - 70) < 25 and abs(p[1] - 45) < 25 and abs(p[2] - 25) < 25
    wall = lambda p: all(abs(p[i] - v) < 15 for i, v in enumerate((205, 200, 190)))
    check('prep: no frame or wall on the edges', not any(brown(p) or wall(p) for p in res['edges']), res['edges'])
    check('prep: JPEG named after the title', res['name'] == 'Skewed Test.jpeg' and res['type'] == 'image/jpeg')
    check('prep: aspect check says it matches', 'Matches a 16' in res['aspect'], res['aspect'])
    shot(d, ctx, 'prep_result')
    d.execute_script('prepUndo()')
    check('prep: undo restores the photo', d.execute_script('return state.fullCanvas.width') == W)
    check('prep: no console errors', not errors(d), errors(d))
    d.execute_script("localStorage.removeItem('pm-openai-key')")


THREE = """Three ways in: the rocks, the towers, the stones.

OPTION 1
TITLE: Under Vancouver Sky
DESCRIPTION: A bit of sun got through and landed on the rocks.
TITLE_RU: Под небом Ванкувера
DESCRIPTION_RU: Немного солнца пробилось и упало на камни.
OPTION 2
**TITLE:** Towers in Cloud
**DESCRIPTION:** Across the water, the towers stand under all that cloud.
**TITLE_RU:** Башни в облаках
**DESCRIPTION_RU:** За водой башни стоят под всеми этими облаками.
OPTION 3
1. TITLE: Older Stones
1. DESCRIPTION: The stones were there long before the city.
1. TITLE_RU: Старые камни
1. DESCRIPTION_RU: Камни были здесь задолго до города."""
ONE = 'Shorter:\nTITLE: Towers\nDESCRIPTION: Towers under cloud.\nTITLE_RU: Башни\nDESCRIPTION_RU: Башни под облаком.'


def suite_dialog(d, ctx):
    load_admin(d, f"http://localhost:{ctx['p_root']}", ctx['photo'])
    d.execute_script("""
      localStorage.setItem('pm-openai-key','sk-test');
      window._replies = %s; window._bodies = [];
      const rf = window.fetch;
      window.fetch = async (u, opt) => {
        if (String(u).includes('api.openai.com')) { window._bodies.push(JSON.parse(opt.body));
          return new Response(JSON.stringify({choices:[{message:{content: window._replies.shift()}}]}), {status:200}); }
        return rf(u, opt);
      };""" % json.dumps([THREE, ONE]))
    d.find_element('id', 'aiSend').click()
    wait_for(d, "document.querySelectorAll('#aiLog .ai-opt').length > 0")
    b0 = d.execute_script('return window._bodies[0]')
    check('dialog: system prompt asks for three', 'exactly three candidates' in b0['messages'][0]['content'])
    check('dialog: first message asks for three', 'three different' in b0['messages'][1]['content'][0]['text'])
    main = d.execute_script("return [...document.querySelectorAll('#aiLog .ai-opt button.primary')].map(b=>b.textContent)")
    check('dialog: three Use buttons (plain, **bold**, numbered labels)',
          main == ['Use 1: Under Vancouver Sky', 'Use 2: Towers in Cloud', 'Use 3: Older Stones'], main)
    opts = d.execute_script("return document.querySelectorAll('#aiLog .ai-opt')")
    d.execute_script("arguments[0].querySelectorAll('button')[0].click()", opts[1])
    f = d.execute_script("return [$('fTitle').value,$('fDesc').value,$('fTitleRu').value,$('fDescRu').value]")
    check('dialog: Use 2 fills all four fields',
          f == ['Towers in Cloud', 'Across the water, the towers stand under all that cloud.', 'Башни в облаках',
                'За водой башни стоят под всеми этими облаками.'], f)
    d.execute_script("$('fTitle').value='x'; $('fDesc').value='y'; $('fTitleRu').value=''; $('fDescRu').value='';")
    d.execute_script("arguments[0].querySelectorAll('button')[1].click()", opts[0])   # title only from 1
    d.execute_script("arguments[0].querySelectorAll('button')[2].click()", opts[2])   # description only from 3
    f = d.execute_script("return [$('fTitle').value,$('fDesc').value,$('fTitleRu').value,$('fDescRu').value]")
    check('dialog: title from 1 + description from 3',
          f == ['Under Vancouver Sky', 'The stones were there long before the city.', 'Под небом Ванкувера',
                'Камни были здесь задолго до города.'], f)
    d.execute_script("$('aiInput').value='shorter please'")
    d.find_element('id', 'aiSend').click()
    wait_for(d, "document.querySelectorAll('#aiLog .ai-proposal').length > 1")
    last = d.execute_script("const p=[...document.querySelectorAll('#aiLog .ai-proposal')].pop(); return [...p.querySelectorAll('button.primary')].map(b=>b.textContent)")
    check('dialog: a revision has one Use this', last == ['Use this'], last)
    d.execute_script("window._clip=null; navigator.clipboard.writeText = t => { window._clip=t; return Promise.resolve(); }; aiCopyCorrection();")
    time.sleep(0.3)
    clip = d.execute_script('return window._clip') or ''
    check('dialog: corrections entry lists all three first drafts', '(1) Under Vancouver Sky' in clip and '(3) Older Stones' in clip)
    shot(d, ctx, 'dialog')
    d.execute_script("localStorage.removeItem('pm-openai-key')")


def suite_form(d, ctx):
    load_admin(d, f"http://localhost:{ctx['p_root']}", ctx['photo'])
    boxes = lambda: d.execute_script("return [...document.querySelectorAll('#fCollBoxes input')].map(b=>[b.value,b.checked])")
    site = d.execute_script('return SITE_CFG.tools.collections')
    check('form: collection boxes = site.json tools.collections', [b[0] for b in boxes()] == site, boxes())
    d.execute_script("""state.header=['title','medium','collection','size','price'];
      const R=[state.header]; const add=(n,m,s)=>{for(let i=0;i<n;i++)R.push(['p'+R.length,m,'Forest',s,'400']);};
      add(17,'acrylic','16x20 in'); add(3,'acrylic','11x14 in'); add(3,'watercolor','5x7 in'); add(2,'acrylic','11x14'); add(1,'Watercolor','16x20');
      window._rows = R; $('fTitle').value='Brand New'; state.filledFor=null; resetForm(); $('fTitle').value='Brand New'; applyDefaultsFromRows(R);""")
    dd = lambda sel, inp: d.execute_script(f"return {{opts:[...$('{sel}').options].map(o=>o.textContent), sel:$('{sel}').value, val:$('{inp}').value, other:!$('{inp}').hidden}}")
    m, s = dd('fMediumSel', 'fMedium'), dd('fSizeSel', 'fSize')
    check('form: medium options merged by case, most common first', m['opts'] == ['—', 'acrylic', 'watercolor', 'Other…'], m['opts'])
    check('form: size options merged (11x14 = 11x14 in)', s['opts'] == ['—', '16x20 in', '11x14 in', '5x7 in', 'Other…'], s['opts'])
    check('form: new painting gets the most common medium and size', m['val'] == 'acrylic' and s['val'] == '16x20 in', (m, s))
    d.execute_script("$('fSizeSel').value='__other'; $('fSizeSel').dispatchEvent(new Event('change'));")
    d.find_element('id', 'fSize').send_keys('12x16 in')
    s = dd('fSizeSel', 'fSize')
    check('form: size Other… shows a box; the aspect check uses it',
          s['other'] and s['val'] == '12x16 in' and '12×16' in d.execute_script("return $('aspectInfo').textContent"))
    d.execute_script("resetForm(); fillFromRow(['Old','Watercolor','Big City:1,Seasons','11x14','150']);")
    m, s = dd('fMediumSel', 'fMedium'), dd('fSizeSel', 'fSize')
    check("form: a row's 'Watercolor' / '11x14' select the sheet's spelling", m['sel'] == 'watercolor' and s['sel'] == '11x14 in', (m['sel'], s['sel']))
    check('form: a row ticks its collections', [b[0] for b in boxes() if b[1]] == ['Big City', 'Seasons'], boxes())
    d.execute_script("[...document.querySelectorAll('#fCollBoxes input')].find(b=>b.value==='Forest').click()")
    check("form: ticking keeps the ':1' order suffix", d.execute_script("return $('fCollection').value") == 'Big City:1,Forest,Seasons')
    d.execute_script("resetForm(); $('fTitle').value='T'; state.saveInFlight=false; saveAll();")
    time.sleep(0.3)
    st = d.execute_script("return [$('saveStatus').textContent, $('fCollBoxes').classList.contains('missing'), state.saveInFlight]")
    check('form: Save refuses without a collection', st[0] == 'Choose at least one collection.' and st[1] and st[2] is False, st)
    d.execute_script("[...document.querySelectorAll('#fCollBoxes input')].find(b=>b.value==='Greeting cards').click()")
    check('form: Greeting cards selects 5x7 in', d.execute_script("return $('fSizeSel').value") == '5x7 in')
    d.execute_script("$('fSizeSel').scrollIntoView({block:'center'})")
    shot(d, ctx, 'form')


def suite_email(d, ctx):
    for label, base, name, addr in [('sveta', f"http://localhost:{ctx['p_root']}/", 'Sveta Colours', ctx['emails']['sveta']),
                                    ('olga', f"http://localhost:{ctx['p_olga']}/", 'Olga Zai', ctx['emails']['olga'])]:
        src = urllib.request.urlopen(base).read().decode()
        check(f'email: {label} source has no address', addr not in src)
        d.get(base)
        wait_for(d, "!!document.querySelector('.contact-meta')", 15)
        check(f'email: {label} rendered page has no address', addr not in d.execute_script('return document.documentElement.outerHTML'))
        meta = d.execute_script("return document.querySelector('.contact-meta').innerText")
        check(f"email: {label} contact reads 'Email {name}'", f'Email {name}' in meta, meta.replace('\n', ' | '))
        check(f'email: {label} mailto built on click', d.execute_script("return mailUrl('Hi there')") == f'mailto:{addr}?subject=Hi%20there')
        d.execute_script('openContact()')
        time.sleep(0.4)
        row = d.execute_script("const r=document.querySelector('#contactBody .inquiry-email'); return r ? r.innerText : ''")
        check(f'email: {label} form row shows the name + Copy email', f'Email {name}' in row and 'copy email' in row.lower() and addr not in row, row)
        d.execute_script("window._c=null; navigator.clipboard.writeText = x => { window._c=x; return Promise.resolve(); }; copyStudioEmail(document.querySelector('#contactBody .inquiry-copy'));")
        time.sleep(0.3)
        check(f'email: {label} Copy email copies the address and says Copied',
              d.execute_script('return window._c') == addr and d.execute_script("return document.querySelector('#contactBody .inquiry-copy').textContent") == 'Copied')
        d.execute_script('closeContact()')
        foot = d.execute_script("return [...document.querySelectorAll('a')].filter(a=>a.textContent.trim()==='Contact').map(a=>a.getAttribute('href'))")
        check(f'email: {label} footer Contact has no mailto', bool(foot) and all(h == '#contact' for h in foot), foot)
        check(f'email: {label} no console errors', not errors(d), errors(d))
        shot(d, ctx, f'email_{label}')
    d.get(f"http://localhost:{ctx['p_root']}/?lang=ru")
    wait_for(d, "!!document.querySelector('.contact-meta')", 15)
    check('email: Russian label', 'Написать Sveta Colours' in d.execute_script("return document.querySelector('.contact-meta').innerText"))


def run_live(d, ctx):
    hosts = [('https://www.sveta-colours.com/admin/', 'Sveta Colours', '170BuYN'),
             ('https://sveta-colours.com/admin/', 'Sveta Colours', '170BuYN'),
             ('https://olgazai.com/admin/', 'Olga Zai', '1zrkk'),
             ('https://www.olgazai.com/admin/', 'Olga Zai', '1zrkk')]
    for url, name, sheet in hosts:
        d.get(url + '?v=%d' % time.time())
        wait_for(d, '!!window.SITE_CFG', 15)
        time.sleep(1)
        i = d.execute_script("""const g=document.getElementById('signInGate');
          return {title:document.title, gate: g ? g.innerText : '', sheet: SHEET_ID, folder: DRIVE_FOLDER, csv: !!PUBLISHED_CSV,
                  form: !!document.getElementById('fSizeSel') && !!document.getElementById('fMediumSel') && !!document.getElementById('fCollBoxes')}""")
        check(f'live: {url} loads {name} config', i['title'].startswith(name) and i['sheet'].startswith(sheet) and bool(i['folder']) and i['csv'], i['title'])
        check(f'live: {url} shows the sign-in gate', name in i['gate'])
        check(f'live: {url} has the current form', i['form'])
        check(f'live: {url} no console errors', not errors(d), errors(d))
        shot(d, ctx, 'live_' + url.split('//')[1].split('/')[0])
    for url in ('https://www.sveta-colours.com/admin/', 'https://olgazai.com/admin/'):
        d.get(url + '?v=%d' % time.time())
        wait_for(d, '!!document.getElementById("gateBtn")', 15)
        main = d.current_window_handle
        d.find_element('id', 'gateBtn').click()
        time.sleep(6)
        pops = [h for h in d.window_handles if h != main]
        if not pops:
            check(f'live: {url} sign-in window opens', False, 'no popup')
            continue
        d.switch_to.window(pops[0])
        time.sleep(2)
        body = d.find_element('tag name', 'body').text
        check(f'live: {url} Google sign-in without origin_mismatch',
              'origin_mismatch' not in body and 'Access blocked' not in body, body[:120].replace('\n', ' | '))
        d.close()
        d.switch_to.window(main)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='comma-separated suites: ' + ','.join(SUITES))
    ap.add_argument('--live', action='store_true', help='check the deployed admin pages instead')
    ap.add_argument('--show', action='store_true', help='run Chrome with a window')
    ap.add_argument('--shots', metavar='DIR', help='save screenshots here')
    a = ap.parse_args()
    if a.shots:
        os.makedirs(a.shots, exist_ok=True)
    ctx = {'shots': a.shots}

    if a.live:
        d = chrome(a.show)
        try:
            run_live(d, ctx)
        finally:
            d.quit()
    else:
        only = a.only.split(',') if a.only else SUITES
        bad = [s for s in only if s not in SUITES]
        if bad:
            ap.error(f'unknown suite(s): {bad}')
        tmp = tempfile.mkdtemp(prefix='verify_admin_')
        dist = os.path.join(tmp, 'olga_dist')
        subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'build_site.py'), 'olga', '--out', dist],
                       check=True, capture_output=True)
        ctx['photo'] = os.path.join(tmp, 'Skewed Test.jpeg')
        ctx['truth'] = make_skewed.make(ctx['photo'])
        ctx['emails'] = {s: json.load(open(os.path.join(REPO, 'sites', s, 'site.json')))['site']['contactEmail']
                         for s in ('sveta', 'olga')}
        ctx['p_root'], ctx['p_olga'] = serve(REPO), serve(dist)
        maps = f"MAP www.sveta-colours.com 127.0.0.1:{ctx['p_root']}, MAP olgazai.com 127.0.0.1:{ctx['p_olga']}"
        d = chrome(a.show, maps)
        try:
            for s in only:
                print(f'── {s}')
                try:
                    globals()['suite_' + s](d, ctx)
                except Exception as e:  # a crashed suite is a failure, the rest still run
                    check(f'{s}: suite ran without crashing', False, f'{type(e).__name__}: {str(e)[:200]}')
        finally:
            d.quit()
    print(f'\n{len(FAILS)} failure(s)' + (': ' + '; '.join(FAILS) if FAILS else ''))
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
