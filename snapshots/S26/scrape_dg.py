#!/usr/bin/env python3
"""
scrape_dg.py — Spring 2026 (S26) snapshot of Dolce & Gabbana sunglasses.

Derived from Webscrapping/D&G.py with four changes:

  1. Removed `import pyautogui`. The original used a fixed-coordinate
     pyautogui click at (860, 550) to press "Carica altro", which broke
     on any screen-size change. Replaced with a Selenium-native click on
     the button identified by its stable class substring
     `category-pagination__load-more` + its inner text "Carica altro".

  2. Replaced every hashed CSS-modules selector with an unhashed
     substring match using [class*="..."]. D&G regenerates the hash
     suffixes on each frontend rebuild — the static-HTML probe showed
     that `--Acusx` (price), `--mBvPF` (price item) and `--Mnk4L`
     (search hit) had all already regenerated since F25. Substring
     selectors survive that.

  3. Prices are JS-rendered post-hydration (not in raw HTML), so the
     selector runs inside Selenium after each load-more click.

  4. Output redirected to ./snapshots/S26/raw/dg/ so the F25 baseline
     in /images_D&G/ stays untouched.

Run from the Tesi-BADS2025 repo root:
    cd /Users/benja/Tesi-BADS2025
    python snapshots/S26/scrape_dg.py
"""

from pathlib import Path
import re
import time
import csv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, ElementClickInterceptedException,
    StaleElementReferenceException,
)
import requests


REPO_ROOT  = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "snapshots" / "S26" / "raw" / "dg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH   = OUTPUT_DIR / "dolcegabbana_products.csv"

# Wipe previous D&G_*.jpg files (not the whole dir — only our own pattern).
# Avoids stale leftover files from earlier runs that had different indexing.
stale = list(OUTPUT_DIR.glob("D&G_*.jpg"))
for p in stale:
    p.unlink()
if stale:
    print(f"[clean] removed {len(stale)} stale D&G_*.jpg from previous run")

URL = "https://www.dolcegabbana.com/it-it/moda/uomo/occhiali-da-sole/"

# Stable substring selectors (survive D&G's hash regeneration).
# IMPORTANT: the trailing `--` in CARD_SELECTOR is the BEM-style "modifier"
# separator that D&G's CSS Modules build uses right before the hash suffix.
# Without it, the substring also matches sub-elements like
# `SearchHitsItem__search-hit__price-item--<hash>` (the price span) and
# `SearchHitsItem__search-hit__image--<hash>` (the image wrapper). That
# overcounts cards ~5× and breaks the inner image/price find_element calls.
CARD_SELECTOR        = "div[class*='SearchHitsItem__search-hit--']"
IMAGE_WRAPPER_SEL    = "a[class*='ProductMedia__product-media__image-wrapper']"
LOAD_MORE_BTN_SEL    = "button[class*='category-pagination__load-more']"
# Price candidates — D&G stacks several classes; we'll try in order
PRICE_SELECTORS = [
    "span[class*='product-price__discount']",   # discounted price (current price)
    "span[class*='product-price__regular']",    # regular price when no discount
    "span[class*='price-item'][class*='money']",
    "span[class*='product-price']",
    "span.money",
]

MAX_LOAD_MORE_CLICKS = 30   # safety cap
HYDRATION_STEP_PX    = 600  # viewport step for the post-pagination hydration pass
HYDRATION_PAUSE_S    = 0.6  # pause per step to let IntersectionObserver fire


# ─── Driver ───────────────────────────────────────────────────────────────────
print(f"[init] starting Chrome via Selenium Manager …")
driver = webdriver.Chrome()
driver.get(URL)
driver.maximize_window()
time.sleep(3)


# ─── Cookie banner ────────────────────────────────────────────────────────────
try:
    cookie_btn = WebDriverWait(driver, 6).until(
        EC.element_to_be_clickable((
            By.XPATH,
            "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti i cookie' or "
            "normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz')) = 'accept all cookies' or "
            "normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz')) = 'aceptar todas las cookies']",
        ))
    )
    cookie_btn.click()
    print("[cookies] accepted")
except (TimeoutException, NoSuchElementException):
    print("[cookies] banner not found — continuing")

time.sleep(2)


# ─── Load all products via Selenium-native click on "Carica altro" ────────────
# Pattern (matches the manual-scroll workflow): scroll to bottom → look for the
# "Carica altro" button → if found, click it; if not, pagination is done.
# Repeat until the button stops appearing entirely.
print("[scroll] loading all products …")
clicks = 0
prev_count = 0
stall = 0

while clicks < MAX_LOAD_MORE_CLICKS:
    # 1. Scroll to bottom to bring the button into the viewport
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    # 2. Look for the button — if it's gone, we're done paginating
    try:
        btn = driver.find_element(By.CSS_SELECTOR, LOAD_MORE_BTN_SEL)
        if "altro" not in btn.text.lower() and "more" not in btn.text.lower():
            print(f"[scroll] selector matched but text='{btn.text}' isn't Carica altro — stopping")
            break
    except NoSuchElementException:
        print(f"[scroll] no more 'Carica altro' button after {clicks} clicks — pagination complete")
        break

    # 3. Click it (via JS to bypass any overlay intercept)
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
        time.sleep(0.8)
        driver.execute_script("arguments[0].click();", btn)
        clicks += 1
        time.sleep(4)  # wait for new products to render
    except (ElementClickInterceptedException, StaleElementReferenceException) as e:
        print(f"[scroll] click intercepted ({type(e).__name__}); retrying")
        time.sleep(3)
        continue

    # 4. Sanity check: did the count actually grow?
    current_count = len(driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR))
    if current_count == prev_count:
        stall += 1
        print(f"[scroll] click {clicks}: count steady at {current_count} (stall {stall}/2)")
        if stall >= 2:
            print(f"[scroll] stalled after {clicks} clicks — stopping")
            break
    else:
        print(f"[scroll] click {clicks}: {prev_count} → {current_count}")
        prev_count = current_count
        stall = 0

print(f"[scroll] pagination done after {clicks} clicks. visible cards: {prev_count}")


# ─── Image hydration pass ─────────────────────────────────────────────────────
# D&G's images are lazy-loaded by an IntersectionObserver — they only swap
# from the SVG placeholder to the real CDN URL once the card enters the
# viewport. The "Carica altro" workflow only ever has the BOTTOM of the page
# in view, so all the freshly-rendered cards above stay un-hydrated.
# Fix: scroll from top to bottom slowly so every card crosses the viewport.
print("[hydrate] scrolling top→bottom to trigger lazy image load …")
driver.execute_script("window.scrollTo(0, 0);")
time.sleep(1.5)
page_h = driver.execute_script("return document.body.scrollHeight")
y = 0
while y < page_h:
    driver.execute_script(f"window.scrollTo(0, {y});")
    time.sleep(HYDRATION_PAUSE_S)
    y += HYDRATION_STEP_PX
# One final pause at the bottom so the last batch finishes hydrating
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(4)
print("[hydrate] done")


# ─── Image + price extraction ─────────────────────────────────────────────────
# Each visible product on D&G's grid renders TWO matching divs in the DOM
# (one for the main image, one for the hover-swap image), so the card count
# is exactly 2× the human-visible product count. We dedup by product_code
# (parsed from the CDN URL) after extraction.
print("[extract] collecting product data …")
products = []
seen_codes: set[str] = set()
cards = driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR)
print(f"[extract] {len(cards)} cards (expect 2× human product count)")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def is_placeholder(url: str) -> bool:
    if not url:
        return True
    return url.startswith("data:") or url.startswith("%3Csvg") or "<svg" in url[:40]


def read_img_url(img_el) -> str:
    """Pull the highest-resolution real URL out of an <img>, ignoring SVG placeholders."""
    srcset = img_el.get_attribute("srcset") or ""
    if srcset:
        urls = [s.strip().split()[0] for s in srcset.split(",") if s.strip()]
        for u in reversed(urls):  # last entry is typically widest
            if not is_placeholder(u):
                return u
    src = img_el.get_attribute("src") or ""
    return src if not is_placeholder(src) else ""


def code_from_href(href: str) -> str:
    """Extract product code from the detail-page URL.

    href looks like: '/.../occhiali-da-sole-{description}-VG4545VP1809V000.html'
    The trailing alphanumeric block before .html is the product code.
    """
    if not href:
        return ""
    m = re.search(r"-([A-Z0-9]+)\.html", href)
    return m.group(1) if m else ""


def fetch_detail_page(detail_url: str) -> tuple[str, str]:
    """Fallback: pull (image_url, price) from a product's detail page.

    The detail page is SSR-rendered, so its HTML carries the real CDN
    image URLs (no JS hydration needed) and a JSON-embedded price.
    We prefer the `_0.jpg` zoom URL (canonical front view).
    """
    try:
        if detail_url.startswith("/"):
            detail_url = "https://www.dolcegabbana.com" + detail_url
        import urllib.request
        req = urllib.request.Request(detail_url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [detail-fetch error] {e}")
        return "", ""

    # Image — prefer the `_0` (front view) URL, fall back to first zoom URL
    img_url = ""
    m = re.search(r'(https://www\.dolcegabbana\.com/dw/image/[^"\s]+/images/zoom/[^"\s?]+_0\.jpg)', html)
    if m:
        img_url = m.group(1)
    else:
        m = re.search(r'(https://www\.dolcegabbana\.com/dw/image/[^"\s]+/images/zoom/[^"\s?]+\.jpg)', html)
        if m:
            img_url = m.group(1)

    # Price — embedded in JSON like  "price": "285.00"
    price = ""
    m = re.search(r'"price"\s*:\s*"?([0-9.,]+)"?', html)
    if m:
        price = f"€{m.group(1).rstrip('.0').rstrip('.') if '.' in m.group(1) else m.group(1)}"
        # Normalize "€285" not "€285.00"
        try:
            v = float(m.group(1).replace(",", "."))
            price = f"€{int(round(v))}"
        except ValueError:
            price = f"€{m.group(1)}"

    return img_url, price


product_idx = 1
detail_fetches = 0
for card in cards:
    try:
        # --- Wrapper link (always present, hydration-independent) ---
        try:
            wrapper = card.find_element(By.CSS_SELECTOR, IMAGE_WRAPPER_SEL)
        except NoSuchElementException:
            continue

        href = wrapper.get_attribute("href") or ""
        product_code = code_from_href(href)
        if not product_code:
            continue

        # Dedup BEFORE attempting hydration — each unique product has 2 cards
        # (main + hover swap); we only want to process it once.
        if product_code in seen_codes:
            continue
        seen_codes.add(product_code)

        # --- IMAGE (try listing card first, fall back to detail page) ---
        img_url = ""
        try:
            img = wrapper.find_element(By.TAG_NAME, "img")
            img_url = read_img_url(img)
        except NoSuchElementException:
            pass

        # --- PRICE from listing card ---
        price_text = "N/A"
        for sel in PRICE_SELECTORS:
            try:
                t = card.find_element(By.CSS_SELECTOR, sel).text.strip()
                if t:
                    price_text = t
                    break
            except NoSuchElementException:
                continue

        # --- Detail-page fallback (covers ~80% of cards where image stayed lazy) ---
        if not img_url or price_text == "N/A":
            detail_img, detail_price = fetch_detail_page(href)
            detail_fetches += 1
            if not img_url and detail_img:
                img_url = detail_img
            if price_text == "N/A" and detail_price:
                price_text = detail_price

        if not img_url:
            print(f"[skip] {product_code}: no image even after detail-page fetch")
            continue

        products.append((f"D&G_{product_idx}", product_code, price_text, img_url))
        product_idx += 1

        if product_idx % 25 == 0:
            print(f"[progress] {product_idx-1} products, {detail_fetches} detail fetches so far")

    except Exception as e:
        print(f"[err] card {product_idx}: {e}")
        continue

print(f"[extract] {len(products)} products. detail-page fetches: {detail_fetches}")
print(f"[extract] price hit rate: {sum(1 for _,_,p,_ in products if p != 'N/A')}/{len(products)}")


# ─── Download images ──────────────────────────────────────────────────────────
print(f"[download] → {OUTPUT_DIR}")
count = 0
for name, code, price, img_url in products:
    img_path = OUTPUT_DIR / f"{name}.jpg"
    try:
        r = requests.get(img_url, timeout=15)
        r.raise_for_status()
        img_path.write_bytes(r.content)
        print(f"  ✓ {name}.jpg  {price}  (code={code})")
        count += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")


# ─── CSV ──────────────────────────────────────────────────────────────────────
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Product Name", "product_code", "Price"])
    w.writerows([(n, c, p) for n, c, p, _ in products])

print(f"[csv] wrote {CSV_PATH}")
print(f"[done] {count} images + {len(products)} prices saved to {OUTPUT_DIR}")

driver.quit()
