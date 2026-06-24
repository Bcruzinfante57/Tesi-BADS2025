"""
Prada_2026.py — eyewear scraper for the 2026 season.

Adapted from the original thesis Prada.py:
  • Switches target URL from `/it/it/mens/accessories/c/10156EU` to the
    women's sunglasses listing
    `/it/it/womens/accessories/sunglasses/c/10086EU`. That's the same
    page playwright captures for the daily Observatory, so the dataset
    here mirrors that one but with the original thesis pipeline's
    selenium + chromedriver stack — the way every other 2025 brand
    catalogue was built.
  • Writes images + prices into `images_Prada_2026/` (keeps the thesis
    `images_Prada/` folder untouched as the 2025 baseline for cross-
    season comparisons).
  • Wraps the pyautogui mouse clicks in try/except so the script keeps
    running if the accessibility permission isn't granted to python on
    this machine (the original would crash out).

Run from the thesis repo root:
    /opt/anaconda3/bin/python Webscrapping/Prada_2026.py
"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import requests
import csv

# pyautogui is no longer needed — the load-more pagination switched to a
# pure-Selenium click in find_load_more_button(). Kept the original
# selenium-only flow in this revision for simplicity + portability.


# Local chromedriver in /Users/benja/tools/chromedriver is version 139
# but installed Chrome is 149 — major-version mismatch refuses to attach.
# Letting Selenium 4.32's built-in Selenium Manager fetch the matching
# driver automatically by omitting an explicit Service.
driver = webdriver.Chrome()
_ = Service  # silence unused-import for the previous-style Service ref

# 2026 target — every Prada eyewear category found in the navigation,
# with the maximum page number that needs to be walked explicitly.
# The Prada storefront paginates inside the listing URL: scrolling
# past page/1 lazy-loads page/2, page/3 etc. as the user scrolls
# (anchor "Mostra di più" only kicks off the first additional page).
# A purely-scroll approach gave intermittent results — sometimes only
# 84/110 of the 126 mens entries showed up before the IntersectionObserver
# timing fell out of sync. Visiting each /page/N URL directly side-
# steps the timing problem entirely.
URLS = [
    # slug,         base url,                                                                          max_page
    ("womens",      "https://www.prada.com/it/it/womens/accessories/sunglasses/c/10086EU",             5),
    ("mens",        "https://www.prada.com/it/it/mens/accessories/sunglasses/c/10163EU",               6),
    ("mens-rossa",  "https://www.prada.com/it/it/mens/prada-linea-rossa/sunglasses/c/10192EU",         3),
]


def accept_cookies():
    try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutto']",
                )
            )
        )
        cookie_btn.click()
        print("Cookies accepted")
    except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")


PRODUCT_SELECTOR = "li.w-full.h-auto.lg\\:h-full"

# ── Pagination: keep clicking "CARICA ALTRO" via Selenium ──────────
# The original thesis script used pyautogui to click the load-more
# button at hard-coded screen coordinates. That approach (a) needs
# pyautogui + accessibility permission and (b) breaks if the page
# layout shifts. Switching to a Selenium-based click that finds the
# button by visible text, so the scraper keeps going until no more
# pages exist.
def find_load_more_button(debug: bool = False):
    """Find the "Mostra di più" pagination element. On Prada it's an
    <a role="link" aria-label="Mostra di più"> — NOT a <button> — so
    we look up by aria-label first (most reliable) and fall back to
    a text search across both <button> and <a>."""
    # 1. Primary: aria-label match (covers the actual anchor element).
    for el in driver.find_elements(By.CSS_SELECTOR, "[aria-label]"):
        try:
            if not el.is_displayed():
                continue
            label = (el.get_attribute("aria-label") or "").strip().lower()
            if "mostra di pi" in label or "show more" in label:
                return el
        except Exception:
            continue

    # 2. Secondary: visible text on <button> or <a>.
    candidates = driver.find_elements(By.XPATH, "//button | //a")
    debug_seen: list[tuple[str, str]] = []
    for el in candidates:
        try:
            if not el.is_displayed():
                continue
            text = (el.text or "").strip()
            tag = el.tag_name
            if not text or len(text) > 80:
                continue
            low = text.lower()
            if debug:
                debug_seen.append((tag, text))
            if "mostra di pi" in low:
                return el
            if "show more" in low or "load more" in low or "carica altro" in low:
                return el
        except Exception:
            continue

    if debug:
        print("  [debug] visible clickable texts:")
        for tag, t in debug_seen[:30]:
            print(f"    · <{tag}> {t!r}")
    return None


IMAGE_CONTAINER_SELECTOR = "picture.std-small\\:block.std-small\\:h-full.std-small\\:w-full.hidden.std-large\\:block"


def scrape_one(slug: str, url: str, records: list) -> int:
    """Visit one category page, load every product on it, append
    (image_url, price, category) records to the shared list. Final
    Prada_N naming + dedup happen after all three categories are
    scraped. Returns the count of records appended."""
    print(f"\n══════════════════════════════════════════════════════════")
    print(f"  category: {slug}")
    print(f"  url:      {url}")
    print(f"══════════════════════════════════════════════════════════")

    driver.get(url)
    time.sleep(3)
    accept_cookies()
    time.sleep(3)

    # ── Load-more loop: click "Mostra di più" (up to 3×) and between
    #    each click do a slow scroll so newly-injected cards render
    #    and the next "Mostra di più" anchor (if any) shows up.
    #    Prada mens has 126 sunglasses total — typically one click
    #    plus a long hydration scroll, but the anchor sometimes
    #    reappears for a second page.
    initial_count = len(driver.find_elements(By.CSS_SELECTOR, PRODUCT_SELECTOR))
    print(f"  initial visible: {initial_count}")

    last_count_overall = initial_count
    MAX_CLICK_ROUNDS = 3
    for click_round in range(MAX_CLICK_ROUNDS):
        # Bring the anchor into viewport before searching.
        for frac in (0.5, 0.85, 1.0):
            driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight*{frac});")
            time.sleep(0.6)
        time.sleep(1.5)

        button = find_load_more_button(debug=(click_round == 0))
        if button is None:
            print(f"  no 'Mostra di più' (round {click_round + 1}) — assume done with clicks.")
            break

        btn_text = (button.text or "").strip() or (button.get_attribute("aria-label") or "")
        print(f"  → click {click_round + 1}: '{btn_text}'")
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)
            time.sleep(0.8)
            try:
                button.click()
            except Exception:
                driver.execute_script("arguments[0].click();", button)
        except Exception as e:
            print(f"  click failed ({type(e).__name__}); breaking.")
            break
        time.sleep(5)  # generous wait for the AJAX response to inject cards

        # Slow scroll from top to bottom — many small scrollBy()s so
        # IntersectionObservers fire for every row. Plateau threshold
        # 5 (≈ 9s of "no growth at bottom") so slow-rendering rows have
        # time to land before we bail.
        print(f"    slow scroll-down hydration (round {click_round + 1})…")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(0.8)
        last_seen = 0
        plateau = 0
        for step in range(120):
            driver.execute_script("window.scrollBy(0, 600);")
            time.sleep(0.4)
            if step % 4 == 3:
                count = len(driver.find_elements(By.CSS_SELECTOR, PRODUCT_SELECTOR))
                at_bottom = driver.execute_script(
                    "return (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 80);"
                )
                if count == last_seen and at_bottom:
                    plateau += 1
                    if plateau >= 5:
                        break
                else:
                    plateau = 0
                last_seen = count

        print(f"    after slow scroll: {last_seen} products visible")
        if last_seen <= last_count_overall:
            # Click didn't actually add products — stop trying.
            print(f"    no growth from this click; exiting click loop.")
            break
        last_count_overall = last_seen
        time.sleep(1.5)

    # ── Walk EACH product card and pull its image url + price as a unit.
    #    Iterating the cards (instead of cards-then-images) keeps every
    #    record self-consistent: the price you read belongs to the same
    #    <li> as the image you saved. Prevents the index-mismatch we
    #    had before when image_containers > product_cards. ─────────
    cards = driver.find_elements(By.CSS_SELECTOR, PRODUCT_SELECTOR)
    print(f"  product cards found: {len(cards)}")

    added = 0
    for item in cards:
        # Image URL
        img_url = ""
        try:
            source = item.find_element(
                By.CSS_SELECTOR, 'source[media="(min-width: 1440px)"]'
            )
            srcset = source.get_attribute("data-srcset") or source.get_attribute("srcset")
            if srcset:
                urls = [u.split(" ")[0] for u in srcset.split(", ")]
                img_url = urls[2] if len(urls) >= 3 else urls[-1]
        except NoSuchElementException:
            img_url = ""
        if not img_url:
            continue

        # Price
        try:
            price_element = item.find_element(
                By.CSS_SELECTOR, "p.product-card__price--new"
            )
            price = price_element.text.strip()
        except NoSuchElementException:
            price = "N/A"

        records.append({"img_url": img_url, "price": price, "category": slug})
        added += 1

    print(f"  → added {added} records")
    return added


# ── Run all 3 categories × their pages ────────────────────────────
records: list[dict] = []
for slug, base_url, max_page in URLS:
    print(f"\n┏━━ {slug.upper()} ━━ {max_page} pages ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    # Page 1: hit the bare base URL (Prada's listing renders the same
    # whether you ask for /page/1 explicitly or the base path).
    scrape_one(slug, base_url, records)
    # Pages 2..max_page: hit /page/N directly. The URL pagination
    # serves each batch independently — no scroll/click needed past
    # the cookie banner, which only fires the first time anyway.
    for page_num in range(2, max_page + 1):
        page_url = f"{base_url}/page/{page_num}"
        scrape_one(slug, page_url, records)

print(f"\nTotal raw records: {len(records)}")


# ── Dedupe by image URL (model code) ───────────────────────────────
# Prada often lists the same physical sunglasses model in BOTH the
# men's catalogue and Linea Rossa (and rarely in womens too). The
# image URL embeds the SKU/model code, so equal URLs = same product.
# Strip query params first (e.g. ?v=2026) so otherwise-identical URLs
# don't survive as separate keys.
def normalise(url: str) -> str:
    return url.split("?", 1)[0]

seen: set[str] = set()
unique: list[dict] = []
duplicates = 0
for r in records:
    key = normalise(r["img_url"])
    if key in seen:
        duplicates += 1
        continue
    seen.add(key)
    unique.append(r)

print(f"After dedup: {len(unique)} unique products ({duplicates} duplicates removed)")


# ── Assign final Prada_N names + download ──────────────────────────
image_folder = "images_Prada_2026"
os.makedirs(image_folder, exist_ok=True)

count = 0
final_rows: list[tuple[str, str, str]] = []
for i, r in enumerate(unique, start=1):
    name = f"Prada_{i}"
    img_path = os.path.join(image_folder, f"{name}.jpg")
    try:
        response = requests.get(r["img_url"], timeout=15)
        response.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(response.content)
        count += 1
    except Exception as e:
        print(f"Error downloading {r['img_url']}: {e}")
    final_rows.append((name, r["price"], r["category"]))

# ── Export CSV ─────────────────────────────────────────────────────
csv_path = os.path.join(image_folder, "prada_products.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Price", "Category"])
    writer.writerows(final_rows)

print(f"\nCSV saved at {csv_path}")
print(f"Image scraping completed: {count} unique Prada eyewear images saved (2026 season).")

driver.quit()
