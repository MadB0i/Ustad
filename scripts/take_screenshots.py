"""Take screenshots of the Ustad UI using Playwright (headless Chromium).

Run from project root:  python scripts/take_screenshots.py
Requires: pip install playwright && python -m playwright install chromium
"""

import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("pip install playwright && python -m playwright install chromium")

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:8177"


def take_screenshots():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        # 1. Main UI — full page
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_timeout(2000)  # let SSE events render
        page.screenshot(path=str(SCREENSHOTS_DIR / "main-ui.png"), full_page=False)
        print("  ✓ main-ui.png")

        # 2. Training config — scroll student panel down to grid-4
        page.evaluate("""() => {
            const s = document.querySelector('.panel-body.split > .scroll');
            if (s) s.scrollTop = 400;
        }""")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "training-config.png"), full_page=False)
        print("  ✓ training-config.png")

        # 3. Hardware advisor — scroll to show advisor panel
        page.evaluate("""() => {
            const s = document.querySelector('.panel-body.split > .scroll');
            if (s) s.scrollTop = 200;
        }""")
        page.wait_for_timeout(500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "hardware-advisor.png"), full_page=False)
        print("  ✓ hardware-advisor.png")

        browser.close()
    print(f"\nSaved to {SCREENSHOTS_DIR}")


if __name__ == "__main__":
    take_screenshots()
