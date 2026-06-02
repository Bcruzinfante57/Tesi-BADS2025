#!/usr/bin/env python3
"""Diagnostic: dump the outerHTML of one hydrated and one placeholder card."""
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get("https://www.dolcegabbana.com/it-it/moda/uomo/occhiali-da-sole/")
driver.maximize_window()
time.sleep(3)

# accept cookies
try:
    btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((
        By.XPATH,
        "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
        "'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti i cookie']")))
    btn.click()
except Exception:
    pass
time.sleep(2)

# scroll a bit to load first batch
driver.execute_script("window.scrollTo(0, 1500);")
time.sleep(3)

# click Carica altro twice to get more cards loaded
for _ in range(2):
    try:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        btn = driver.find_element(By.CSS_SELECTOR, "button[class*='category-pagination__load-more']")
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(3)
    except Exception as e:
        print(f"click failed: {e}")
        break

# scroll back to top and look at cards 1-5 and 80-85 (likely placeholder)
driver.execute_script("window.scrollTo(0, 0);")
time.sleep(2)
cards = driver.find_elements(By.CSS_SELECTOR, "div[class*='SearchHitsItem__search-hit--']")
print(f"total cards: {len(cards)}")
print()

def dump_card(idx, card):
    print(f"========== CARD {idx} ==========")
    # Get the inner img tag's HTML
    try:
        wrapper = card.find_element(By.CSS_SELECTOR, "a[class*='ProductMedia__product-media__image-wrapper']")
        print("WRAPPER outerHTML (first 600 chars):")
        wrapper_html = wrapper.get_attribute("outerHTML")
        print(wrapper_html[:600])
        print()
        img = wrapper.find_element(By.TAG_NAME, "img")
        print("IMG attributes:")
        for attr in ["src", "srcset", "data-src", "data-srcset", "data-original", "loading", "class"]:
            v = img.get_attribute(attr)
            if v:
                # truncate long
                vv = v[:150] + ("..." if len(v) > 150 else "")
                print(f"  {attr}: {vv}")
    except Exception as e:
        print(f"could not dump: {e}")
    print()

# Dump first 3 cards (likely hydrated)
for i, idx in enumerate([0, 1, 2]):
    if idx < len(cards):
        dump_card(idx + 1, cards[idx])

# Dump cards from middle / end (likely placeholder)
for idx in [80, 100, 150, 200]:
    if idx < len(cards):
        dump_card(idx + 1, cards[idx])

driver.quit()
