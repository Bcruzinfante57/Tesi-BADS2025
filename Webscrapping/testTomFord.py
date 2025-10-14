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

url = "https://www.tomfordfashion.it/it-it/occhiali/uomini/occhiali-da-sole/"

driver.get(url)
driver.maximize_window()

time.sleep(3)

# --- 1. ACCEPT COOKIES ---
try:
        # XPATH translated for the expected "Accept All Cookies" button text (case-insensitive)
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti i cookie']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

# --- 2. INFINITE SCROLL AND "LOAD MORE" CLICKS (ADJUSTED LOGIC) ---

print("Establishing initial scroll position and product count...")
products_selector = "div.product"  # Product container selector
FIXED_CLICK_X, FIXED_CLICK_Y = 860, 650 # Fixed coordinates for all clicks (860, 650)
INITIAL_SCROLL_PERCENTAGE = 0.85 # Initial scroll percentage (85%)
LOOP_SCROLL_PERCENTAGE = 1.00 # Loop scroll percentage (100%)

# --- PHASE 1: INITIAL ACTION (COUNT, SCROLL 85%, CLICK) ---

# 1) Count products (Base count before any action)
products_count = len(driver.find_elements(By.CSS_SELECTOR, products_selector)) 
print(f"1) Initial Products Count: {products_count}")

# 2) Get initial height and 85% position
last_height = driver.execute_script("return document.body.scrollHeight")
target_scroll_position_85 = int(last_height * INITIAL_SCROLL_PERCENTAGE)

# 3) First Scroll (85%)
print(f"2) First scroll to {int(INITIAL_SCROLL_PERCENTAGE * 100)}% (position: {target_scroll_position_85}px).")
driver.execute_script(f"window.scrollTo(0, {target_scroll_position_85});")
time.sleep(3) # Wait

# 4) First Click (Fixed Coordinates)
print(f"3) Performing initial 'LOAD MORE' click (Coord: {FIXED_CLICK_X}, {FIXED_CLICK_Y}).")
pyautogui.moveTo(FIXED_CLICK_X, FIXED_CLICK_Y, duration=1.5) 
time.sleep(1)
pyautogui.click(duration=0.5)
time.sleep(5) # Wait for content to load

# 5) Count and verify before entering the loop
products_count_after_first_click = len(driver.find_elements(By.CSS_SELECTOR, products_selector))

if products_count_after_first_click > products_count:
    products_count = products_count_after_first_click
    print(f"4) Products loaded after initial click: {products_count}. Starting continuous loop.")
else:
    print("Initial click did not load new products. Terminating script.")
    time.sleep(8)
    exit() 


# ----------------------------------------------------------------------
# --- CONTINUOUS LOOP (SCROLL 100% -> CLICK 860, 650 -> COUNT) ---

print(f"Starting continuous loop (Fixed {int(LOOP_SCROLL_PERCENTAGE * 100)}% scroll, Fixed click {FIXED_CLICK_X}, {FIXED_CLICK_Y}).")

while True:
        
    # 1) Recalculate height and the new 100% position
    new_height = driver.execute_script("return document.body.scrollHeight")
    target_scroll_position_100 = int(new_height * LOOP_SCROLL_PERCENTAGE)
    
    # 2) Scroll to 100% of the NEW height
    print(f"Scrolling to {int(LOOP_SCROLL_PERCENTAGE * 100)}% of the new height (position: {target_scroll_position_100}px).")
    # Note: Scrolling to document.body.scrollHeight (100%) moves the viewport to the very bottom
    driver.execute_script(f"window.scrollTo(0, {target_scroll_position_100});")
    time.sleep(3) # Wait

    # 3) Click (Fixed Coordinates)
    print(f"Performing recurring 'LOAD MORE' click (Coord: {FIXED_CLICK_X}, {FIXED_CLICK_Y}).")
    pyautogui.moveTo(FIXED_CLICK_X, FIXED_CLICK_Y, duration=1.5) 
    time.sleep(1)
    pyautogui.click(duration=0.5)
    time.sleep(5) # Wait

    # 4) Count products and compare with the last count
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

# --- 3. IMAGE URL COLLECTION (TOM FORD - INDEXED FORMAT) ---

# NOTE: The get_highest_res_url function remains removed.

print("Collecting PRIMARY image URLs (View 1) and Product IDs for Tom Ford...")
products_to_download = {} # {product_code: image_url}
product_idx = 1 # Initial index
MAX_WAIT_TIME = 10 

# --- PRODUCT IMAGE EXTRACTION LOGIC (MODIFIED FOR INDEXED ID) ---

try:
    # 1. Find all product containers
    grid_items = driver.find_elements(By.CSS_SELECTOR, "div.product") 
    print(f"Number of product containers found: {len(grid_items)}")
    
    # 2. Iterate over each container
    for item in grid_items:
        img_url = ""
        
        # *** 2a. Product ID is now based solely on the index (as requested) ***
        product_id_base = str(product_idx) 

        # *** 2b. CRITICAL STEP: Force product into view to trigger high-res lazy load ***
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item)
        time.sleep(0.5) 

        # --- IMAGE SELECTION LOGIC (Targeting the exact image class) ---
        try:
            # Selector: Targets the specific image tag with the required classes.
            img_selector = 'img.tile-image.tile-image-primary.loaded.loaded-alt' 
            
            # Find the primary image element directly inside the product container
            img = item.find_element(By.CSS_SELECTOR, img_selector)
            
            # Dynamically wait until the 'srcset' attribute has meaningful high-quality content
            try:
                # Assuming WebDriverWait and TimeoutException are imported
                wait = WebDriverWait(driver, MAX_WAIT_TIME)
                wait.until(
                    lambda d: img.get_attribute("srcset") and len(img.get_attribute("srcset")) > 10
                )
            except TimeoutException:
                pass # If timeout, proceed with whatever is loaded

            # --- ROBUST EXTRACTION LOGIC (Extracting Last URL in srcset or fallback to src) ---
            img_url = None
            srcset = img.get_attribute("srcset")
            
            # PRIORITY 1: Get the last URL from the srcset attribute
            if srcset:
                # Split by comma, strip whitespace, get the last segment, and take the URL part (before any ' w' or ' x')
                sources = [s.strip() for s in srcset.split(',') if s.strip()]
                if sources:
                    # Get the last source entry, and split it by space to get just the URL (index 0)
                    img_url = sources[-1].split()[0]
            
            # PRIORITY 2: Fallback to 'src'
            if not img_url or "placeholder" in str(img_url).lower():
                img_url = img.get_attribute("src")

            # Final Validation
            if not img_url or "placeholder" in str(img_url).lower() or not str(img_url).startswith("http"):
                print(f"  Skipping item {product_id_base}: No valid image URL found for Primary View (URL or Placeholder).")
                # product_idx is NOT incremented here.
                continue 
            
            # --- LOGIC TO EXTRACT THE PRODUCT CODE (FINAL INDEXED FORMAT) ---
            product_code = f"TOMFORD_{product_id_base}_VIEW1" 

            # If the extraction was successful, add the URL to the dictionary
            products_to_download[product_code] = img_url
            
            print(f"  ✅ Product {product_id_base} - Primary Image (View 1) URL obtained: {img_url[:60]}...")
            
            # *** product_idx only increments on successful extraction/logging ***
            product_idx += 1 


        except NoSuchElementException as e:
            # Handles cases where the image element is not found within the product container.
            print(f"  Product {product_id_base}: Image element not found with selector '{img_selector}'. Skipping. Error: {e}")
            # product_idx is NOT incremented here.
            continue
        except Exception as e:
            print(f"  Unexpected error on item {product_id_base}: {e}")
            # product_idx is NOT incremented here.
            continue


except Exception as e:
    print(f"General error locating product containers: {e}")

print(f"Collection finished. Total products with primary image URLs: {len(products_to_download)}")
# --- 4. PRICE COLLECTION ---
print("Collecting product prices...")
products_prices = []

try:
    # Selector que apunta al <span> que contiene el valor numérico (e.g., <span class="value" content="370.00">)
    # Se utiliza 'span.value' ya que contiene el atributo 'content' que tiene el valor limpio.
    price_value_elements = driver.find_elements(By.CSS_SELECTOR, "span.sales > span.value")
    print(f"Number of price value elements found: {len(price_value_elements)}")

    for element in price_value_elements:
        # Extraemos el valor directamente del atributo 'content'
        price_content = element.get_attribute("content")
        
        if price_content:
            products_prices.append(price_content)
        else:
            # Fallback por si el atributo 'content' está vacío o no existe
            products_prices.append(element.text.strip())


except Exception as e:
    print(f"Error locating prices: {e}")
    
print(f"Total prices collected: {len(products_prices)}")


# --- 5. IMAGE DOWNLOAD & CSV CREATION ---
image_folder = "images_TomFord"
os.makedirs(image_folder, exist_ok=True)

csv_path = os.path.join(image_folder, "TomFord_products.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product_ID", "Price"])

    # Loop through the collected image URLs and their associated prices
    # We use enumerate over the dictionary keys to maintain consistency and generate an index.
    for idx, (product_code, img_url) in enumerate(products_to_download.items(), start=1):
        
        # Use a sequential ID for simplicity, but the product_code (key) could also be used directly
        # product_id = product_code  # Alternative: use the extracted code directly
        product_id = f"TomFord_{idx}"

        # Get price (using index)
        price = products_prices[idx-1] if idx-1 < len(products_prices) else "N/A"

        # Download image
        img_path = os.path.join(image_folder, f"{product_id}.jpg")
        try:
            img_data = requests.get(img_url).content
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            print(f"Image saved: {product_id}.jpg")
        except Exception as e:
            print(f"Error downloading image from {img_url}: {e}")

        # Write to CSV
        writer.writerow([product_id, price])

driver.quit()
print(f"✅ Scraping completed: {len(products_to_download)} products saved with prices.")
print(f"✅ CSV saved at {csv_path}")