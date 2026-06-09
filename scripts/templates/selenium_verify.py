#!/usr/bin/env python3
"""Selenium UI verify template for the portfolio site.

Copy this file to /tmp/<intent>_verify.py and edit the ASSERTIONS section.
Bakes in the gotchas documented in docs/portfolio-playbook.md so you don't
re-discover them.

Run from inside the venv:
    cd tools/image_pipeline && source .venv/bin/activate
    python /tmp/<intent>_verify.py

The template verifies one batch of UI changes on the deployed site. Each
verify script should:

  1. Open the URL.
  2. Navigate to the target page via showPage() (not hash routing).
  3. Wait for a page-specific element.
  4. Dump geometry / state via execute_script.
  5. Hover-test or modal-test where applicable.
  6. Save a screenshot to /tmp/hover-test/.
  7. Print a clean PASS/FAIL summary.
"""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---- Config -----------------------------------------------------------------

SITE_URL = "https://www.sveta-colours.com/"
SCREENSHOT_DIR = "/tmp/hover-test"
SCREENSHOT_NAME = "verify.png"
WINDOW_SIZE = "1440,1800"

# ---- Setup ------------------------------------------------------------------

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
opts = Options()
opts.add_argument(f"--window-size={WINDOW_SIZE}")
driver = webdriver.Chrome(options=opts)


def cursor_visible() -> bool:
    """Custom cursor visibility — the cursor uses a `.visible` class toggle."""
    cursor = driver.find_element(By.CSS_SELECTOR, ".custom-cursor")
    return "visible" in (cursor.get_attribute("class") or "")


def goto_page(name: str, wait_selector: str) -> None:
    """Navigate SPA-style. Hash routing doesn't deep-link on first load —
    always use showPage() and wait for a page-specific element."""
    driver.execute_script(f"showPage({name!r});")
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
    )
    # Let masonry compute + aspect-ratio refinement settle.
    time.sleep(2.0)


def hover(element, reset_first: bool = True) -> None:
    """Move the mouse to `element`. If reset_first, move to h1 first so
    mouseenter re-fires (Selenium doesn't refire on same-target move)."""
    if reset_first:
        ActionChains(driver).move_to_element(
            driver.find_element(By.TAG_NAME, "h1")
        ).perform()
        time.sleep(0.25)
    ActionChains(driver).move_to_element(element).perform()
    time.sleep(0.4)


def dump_card_geometry() -> list[dict]:
    """Return rendered geometry for every card in the first gallery grid.
    Use execute_script so it's one round-trip — find_element per card is slow."""
    return driver.execute_script(
        """
        const grid = document.querySelector('.gallery-grid');
        if (!grid) return [];
        return Array.from(grid.querySelectorAll('.painting-card')).map(c => {
          const r = c.getBoundingClientRect();
          const img = c.querySelector('.card-image-inner');
          const imgR = img ? img.getBoundingClientRect() : null;
          return {
            title: (c.querySelector('.card-title') || {}).textContent || '',
            w: Math.round(r.width),
            h: Math.round(r.height),
            img_w: imgR ? Math.round(imgR.width) : null,
            img_h: imgR ? Math.round(imgR.height) : null,
            orient: c.getAttribute('data-orientation') || 'portrait',
            grid_row: c.style.gridRow || '',
            sold: c.classList.contains('is-sold'),
          };
        });
        """
    )


# ---- Verify -----------------------------------------------------------------

try:
    driver.get(SITE_URL)
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".painting-card"))
    )

    # Navigate to the works page. Replace 'paintings' / '.hang-chapter' with
    # whatever you're verifying.
    goto_page("paintings", ".hang-chapter")

    # ---- ASSERTIONS (edit per verify) --------------------------------------

    cards = dump_card_geometry()
    print(f"Found {len(cards)} cards in the first chapter.\n")

    print(f"  {'title':<32} {'w':<5} {'h':<5} orient    sold  row")
    for c in cards:
        print(
            f"  {c['title'][:30]:<32} {c['w']:<5} {c['h']:<5} "
            f"{c['orient']:<10} {str(c['sold']):<5} {c['grid_row']}"
        )

    # Example: assert at least one landscape spans 2 columns
    landscapes = [c for c in cards if c["orient"] == "landscape"]
    portraits = [c for c in cards if c["orient"] == "portrait"]
    if landscapes and portraits:
        avg_portrait_w = sum(c["w"] for c in portraits) / len(portraits)
        ls = landscapes[0]
        ratio = ls["w"] / avg_portrait_w
        print(
            f"\nLandscape '{ls['title']}' is {ratio:.1f}× the average "
            f"portrait width (expect ~2.0)."
        )

    # Example: custom cursor scope — visible over image, hidden over title
    if cards:
        first_card = driver.find_element(By.CSS_SELECTOR, ".painting-card")
        driver.execute_script(
            "arguments[0].scrollIntoView({block:'center'});", first_card
        )
        time.sleep(0.4)

        img = first_card.find_element(By.CSS_SELECTOR, ".card-image-inner")
        title = first_card.find_element(By.CSS_SELECTOR, ".card-title")

        hover(img)
        over_image = cursor_visible()
        hover(title)
        over_title = cursor_visible()

        print(f"\nCursor over image: visible={over_image}  (expect True)")
        print(f"Cursor over title: visible={over_title}  (expect False)")

    # ---- Screenshot --------------------------------------------------------

    out = os.path.join(SCREENSHOT_DIR, SCREENSHOT_NAME)
    driver.save_screenshot(out)
    print(f"\nScreenshot: {out}")
finally:
    driver.quit()
