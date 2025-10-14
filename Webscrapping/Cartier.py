from selenium import webdriver
from selenium.webdriver.chrome.service import Service # Kept for reference, but not used below
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import requests
import re   
import pyautogui
import csv

# --- SETUP ---

# Use Selenium Manager (recommended method) to automatically find the correct chromedriver.
# NOTE: The line 'from selenium.webdriver.chrome.service import Service' is no longer required 
# if using this simplified initiation, which is robust against driver version mismatches.
driver = webdriver.Chrome() 

url = "https://www.cartier.com/it-it/bags-and-accessories/sunglasses?page=0&srule=recommended"

driver.get(url)
driver.maximize_window()

time.sleep(3)

# --- 1.1 ACCEPT COOKIES ---
try:
        # XPATH translated for the expected "Accept All Cookies" button text (case-insensitive)
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

# --- 1.2 HANDLE COUNTRY POP-UP: CLICK 'CONTINUE IN ITALY' ---
try:
    # XPATH looks for a button or element containing the exact text 'CONTINUE IN ITALY' 
    # and waits for it to be clickable.
    italy_btn = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'continue in italy']"))
    )
    italy_btn.click()
    print("Country pop-up handled: Clicked 'CONTINUE IN ITALY'.")
except (TimeoutException, NoSuchElementException):
    print("Country pop-up not found, different, or already handled.")

time.sleep(3)

# --- 2. INFINITE SCROLL AND "LOAD MORE" CLICKS (FIXED MOUSE POSITION LOGIC) ---

print("Establishing initial scroll position and product count...")
products_selector = "div.product__tile"  # Product container selector
FIXED_CLICK_X, FIXED_CLICK_Y = 960, 300 # Fixed coordinates for all clicks (960, 300)
FIXED_SCROLL_PERCENTAGE = 0.70 # Fixed scroll percentage (70%)

# --- PHASE 1: INITIAL ACTION (COUNT, SCROLL 70%, MOVE MOUSE, CLICK) ---

# 1) Count products (Base count before any action)
products_count = len(driver.find_elements(By.CSS_SELECTOR, products_selector)) 
print(f"1) Initial Products Count: {products_count}")

# 2) Get initial height and 70% position
last_height = driver.execute_script("return document.body.scrollHeight")
target_scroll_position_70 = int(last_height * FIXED_SCROLL_PERCENTAGE)

# 3) First Scroll (70%) - This is the ONLY scroll operation.
print(f"2) Performing ONLY initial scroll to {int(FIXED_SCROLL_PERCENTAGE * 100)}% (position: {target_scroll_position_70}px).")
driver.execute_script(f"window.scrollTo(0, {target_scroll_position_70});")
time.sleep(3) # Wait for the 'LOAD MORE' button to appear in view

# 4) Move Mouse to Fixed Position (The mouse stays here for the rest of the script)
print(f"3) Moving mouse to fixed 'LOAD MORE' button position (Coord: {FIXED_CLICK_X}, {FIXED_CLICK_Y}).")
pyautogui.moveTo(1100, 300, duration=0.5) 
pyautogui.moveTo(FIXED_CLICK_X, FIXED_CLICK_Y, duration=1.5) 
time.sleep(1)

# 5) First Click 
print("4) Performing initial 'LOAD MORE' click.")
pyautogui.click(duration=0.5)
time.sleep(5) # Wait for content to load

# 6) Count and verify before entering the loop
products_count_after_first_click = len(driver.find_elements(By.CSS_SELECTOR, products_selector))

if products_count_after_first_click > products_count:
    products_count = products_count_after_first_click
    print(f"5) Products loaded after initial click: {products_count}. Starting continuous click loop.")
else:
    print("Initial click did not load new products. Terminating script.")
    time.sleep(8)
    exit() 


# ----------------------------------------------------------------------
# --- CONTINUOUS LOOP (NO SCROLL -> CLICK -> COUNT) ---

print("Starting continuous click loop (Mouse stays fixed at button coordinates).")

while True:
        
    # *** SCROLLING STEP (1 & 2) REMOVED ***
    
    # 1) Click (Mouse is already in the FIXED position)
    print("Performing recurring 'LOAD MORE' click (No mouse movement).")
    pyautogui.click(duration=0.5)
    time.sleep(5) # Wait

    # 2) Count products and compare with the last count
    current_products = driver.find_elements(By.CSS_SELECTOR, products_selector)
    new_products_count = len(current_products)

    if new_products_count > products_count:
        # Count increased: Update and continue loop
        products_count = new_products_count
        print(f"Products loaded: {products_count} (increase detected).")
    else:
        # Count did not increase. Exit loop.
        print(f"Product count has not changed ({new_products_count}). Exiting scroll loop.")
        break

print("Scrolling finished. All products should be visible.")
time.sleep(8) # Final wait


# --- DATA COLLECTION CONFIGURATION ---
print("Collecting image URLs and prices for Fendi...")
products_data = []  # List to store (file_name, price, img_url)
product_idx = 1     # Counter for the sequential file name (FENDI_1, FENDI_2, ...)
MAX_WAIT_TIME = 10  # Maximum seconds for dynamic wait

try:
    # 1. Find all product containers
    grid_items = driver.find_elements(By.CSS_SELECTOR, "div.product")
    print(f"Number of product containers found: {len(grid_items)}")
    
    # 2. Iterate over each container
    for item in grid_items:
        img_url = ""
        price = "N/A"
        
        # *** CRITICAL STEP: Force product into view to trigger high-res lazy load ***
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item)
        time.sleep(0.5) # Short pause to allow event handlers to fire
        
        try:
            # --- IMAGE EXTRACTION FOR CARTIER ---
            img = item.find_element(By.CSS_SELECTOR, "img.product__image")
            
            # Scroll to ensure lazy loading triggers
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", img)
            time.sleep(0.5)

            img_url = img.get_attribute("src")

            # Convert relative path to absolute URL if needed
            if img_url and img_url.startswith("/"):
                img_url = "https://www.cartier.com" + img_url

            # Validation
            if not img_url or "placeholder" in img_url.lower():
                print(f"⚠️ Item {product_idx}: invalid or placeholder image URL, skipping.")
                continue

            # --- PRICE EXTRACTION ---
            # Selector for the price element
            price_selector = "p.product__price" 
            try:
                # Wait for the price element to be visible and have text
                price_element = WebDriverWait(item, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, price_selector))
                )
                price = price_element.text.strip()
            except (TimeoutException, NoSuchElementException):
                 price = "N/A" # Price not found, defaults to 'N/A'
            
            # --- SAVE DATA AND NAME ---
            if img_url:
                product_name = f"Cartier_{product_idx}"
                products_data.append((product_name, price, img_url))
                product_idx += 1 # Increase index only if the product is valid
                
        except NoSuchElementException:
            # If a container doesn't have the basic structure (image or price), skip it
            print(f"  Skipping item {product_idx}: Basic element not found.")
            product_idx += 1 
            continue 

except Exception as e:
    print(f"General error during data collection: {e}")

print(f"Total products scraped and ready for download: {len(products_data)}")


# --- IMAGE DOWNLOAD AND CSV PREPARATION ---
image_folder = "images_Cartier"
os.makedirs(image_folder, exist_ok=True)

count = 0
for product_name, price, img_url in products_data:
    img_path = os.path.join(image_folder, f"{product_name}.jpg")
    
    try:
        # Add a timeout to prevent the download from hanging
        img_data = requests.get(img_url, timeout=10).content
        with open(img_path, "wb") as f:
            f.write(img_data)
        print(f"Image saved: {product_name}.jpg ({price})")
        count += 1
    except Exception as e:
        print(f"Error downloading image from {img_url}: {e}")

# --- EXPORT CSV ---
csv_path = os.path.join(image_folder, "Cartier_products.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Price"])
    # Write only the name and price
    writer.writerows([(name, p) for name, p, url in products_data]) 

print(f"✅ CSV saved at {csv_path}")
print(f"✅ Scraping completed: {count} images saved + {len(products_data)} prices scraped.")

driver.quit()
