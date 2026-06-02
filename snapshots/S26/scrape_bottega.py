#!/usr/bin/env python3
"""
scrape_bottega.py — Spring 2026 (S26) snapshot of Bottega Veneta sunglasses.

Derived from Webscrapping/Bottega.py with three changes:

  1. Removed the leftover `import pyautogui` (the original never called it,
     just imported it). pyautogui isn't installed in the conda base env
     and Bottega's scraper is pure-Selenium anyway.

  2. Switched from a hardcoded chromedriver Service path (which had drifted
     to v139 against Chrome 148) to Selenium Manager (no Service arg);
     Selenium 4.6+ auto-downloads a matching driver.

  3. Output redirected to ./snapshots/S26/raw/bottega/  so the F25 baseline
     in /images_bottega/ stays untouched.

Run from the Tesi-BADS2025 repo root:
    cd /Users/benja/Tesi-BADS2025
    python snapshots/S26/scrape_bottega.py
"""

from pathlib import Path
import time
import csv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests


REPO_ROOT     = Path(__file__).resolve().parents[2]
OUTPUT_DIR    = REPO_ROOT / "snapshots" / "S26" / "raw" / "bottega"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH      = OUTPUT_DIR / "bottega_products.csv"

URL = "https://www.bottegaveneta.com/it-it/search?q=occhiali"


# ─── Driver (Selenium Manager auto-resolves the matching chromedriver) ────────
print(f"[init] starting Chrome via Selenium Manager …")
driver = webdriver.Chrome()
driver.get(URL)
driver.maximize_window()
time.sleep(3)


# ─── Cookie banner ────────────────────────────────────────────────────────────
try:
    cookie_btn = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((
            By.XPATH,
            "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
            "'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti i cookie']",
        ))
    )
    cookie_btn.click()
    print("[cookies] accepted")
except (TimeoutException, NoSuchElementException):
    print("[cookies] banner not found or wording changed — continuing")

time.sleep(3)


# ─── Scroll to load all products ──────────────────────────────────────────────
driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.4);")
print("[scroll] kickstart at 40%")
time.sleep(5)

last_height = driver.execute_script("return document.body.scrollHeight * 0.6")
products_count = 0
stall_passes = 0

while True:
    for _ in range(3):
        driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
        time.sleep(3)

    current_products = driver.find_elements(By.CSS_SELECTOR, "article.c-product")
    new_count = len(current_products)

    if new_count == products_count:
        stall_passes += 1
        print(f"[scroll] stall — count steady at {new_count} ({stall_passes}/2)")
        if stall_passes >= 2:
            break
    else:
        print(f"[scroll] loaded {new_count} cards")
        products_count = new_count
        stall_passes = 0

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("[scroll] page height stable")
        break
    last_height = new_height

print(f"[scroll] finished — {products_count} cards visible")
time.sleep(8)


# ─── Image + price extraction ─────────────────────────────────────────────────
print("[extract] collecting product data …")
products = []
cards = driver.find_elements(By.CSS_SELECTOR, "article.c-product[data-pid]")
print(f"[extract] {len(cards)} cards with data-pid")

product_idx = 1
for card in cards:
    wait = WebDriverWait(card, 5)
    img_url = ""
    price = "N/A"

    try:
        active_sel = (
            "ul.c-product__carousel "
            "li.c-product__carousel--slide.swiper-slide-active "
            "img.c-product__image"
        )
        fallback_sel = (
            "ul.c-product__carousel li.c-product__carousel--slide img.c-product__image"
        )

        try:
            img = card.find_element(By.CSS_SELECTOR, active_sel)
        except NoSuchElementException:
            try:
                img = card.find_element(By.CSS_SELECTOR, fallback_sel)
            except NoSuchElementException:
                continue

        try:
            wait.until(lambda d: img.get_attribute("srcset") or img.get_attribute("src"))
        except TimeoutException:
            print(f"[warn] card {product_idx}: srcset/src not loaded in 5s")

        srcset = img.get_attribute("srcset") or ""
        if srcset:
            parts = [p.strip() for p in srcset.split(",")]
            def width_of(p):
                toks = p.split()
                if len(toks) >= 2 and toks[1].endswith("w"):
                    try:    return int(toks[1][:-1])
                    except: return 0
                return 0
            best = max(parts, key=width_of)
            img_url = best.split()[0]
        else:
            img_url = img.get_attribute("src") or ""

        try:
            price_el = card.find_element(By.CSS_SELECTOR, "p.c-price__value--current")
            price = price_el.text.strip()
        except NoSuchElementException:
            price = "N/A"

        # data-pid is the brand-side product id — preserve it for downstream matching
        pid = card.get_attribute("data-pid") or ""

        if img_url:
            product_name = f"Bottega_{product_idx}"
            products.append((product_name, pid, price, img_url))
            product_idx += 1
        else:
            print(f"[skip] card {product_idx}: no image URL")

    except Exception as e:
        print(f"[err] card {product_idx}: {e}")
        continue

print(f"[extract] {len(products)} products with image+price")


# ─── Download images ──────────────────────────────────────────────────────────
print(f"[download] saving to {OUTPUT_DIR} …")
count = 0
for name, pid, price, img_url in products:
    img_path = OUTPUT_DIR / f"{name}.jpg"
    try:
        r = requests.get(img_url, timeout=10)
        r.raise_for_status()
        img_path.write_bytes(r.content)
        print(f"  ✓ {name}.jpg ({price})")
        count += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")


# ─── CSV (new column: data_pid, lets us match by SKU first, ViT as fallback) ──
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Product Name", "data_pid", "Price"])
    w.writerows([(name, pid, price) for name, pid, price, _ in products])

print(f"[csv] wrote {CSV_PATH}")
print(f"[done] {count} images + {len(products)} prices saved to {OUTPUT_DIR}")

driver.quit()
