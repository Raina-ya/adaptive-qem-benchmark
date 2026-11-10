"""Render slides/deck.html to slides/qem_project_slides.pdf (needs Playwright with Chromium)."""

from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 900})
    page.goto((HERE / "deck.html").as_uri())
    page.wait_for_load_state("networkidle")
    page.pdf(
        path=str(HERE / "qem_project_slides.pdf"),
        width="1600px",
        height="900px",
        print_background=True,
        prefer_css_page_size=True,
    )
    browser.close()
