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
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'consenti tutto']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

# --- 2. INFINITE SCROLL AND "LOAD MORE" CLICKS (SIMPLIFIED FIXED LOGIC) ---

print("Establishing initial scroll position and product count...")
products_selector = "div.product"  # Product container selector
FIXED_CLICK_X, FIXED_CLICK_Y = 860, 600 # Fixed coordinates for all clicks (860, 600)
FIXED_SCROLL_PERCENTAGE = 0.70 # Fixed scroll percentage (70%)

# --- FASE 1: INITIAL ACTION (COUNT, SCROLL 70%, CLICK) ---

# 1) Count products (Base count before any action)
products_count = len(driver.find_elements(By.CSS_SELECTOR, products_selector)) 
print(f"1) Initial Products Count: {products_count}")

# 2) Get initial height and 70% position
last_height = driver.execute_script("return document.body.scrollHeight")
target_scroll_position_70 = int(last_height * FIXED_SCROLL_PERCENTAGE)

# 3) First Scroll (70%)
print(f"2) Scroll to 70% (position: {target_scroll_position_70}px).")
driver.execute_script(f"window.scrollTo(0, {target_scroll_position_70});")
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
# --- CONTINUOUS LOOP (SCROLL 70% -> CLICK 860, 600 -> COUNT) ---

print(f"Starting continuous loop (Fixed 70% scroll, Fixed click {FIXED_CLICK_X}, {FIXED_CLICK_Y}).")

while True:
        
    # 1) Recalculate height and the new 70% position
    new_height = driver.execute_script("return document.body.scrollHeight")
    target_scroll_position_70 = int(new_height * FIXED_SCROLL_PERCENTAGE)
    
    # 2) Scroll to 70% of the NEW height
    print(f"Scrolling to 70% of the new height (position: {target_scroll_position_70}px).")
    driver.execute_script(f"window.scrollTo(0, {target_scroll_position_70});") 
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

# --- 3. IMAGE URL COLLECTION (CHLOÉ) ---
def get_highest_res_url(srcset):
    """
    Parses a srcset string (e.g., 'url1 1x, url2 2x, url3 800w') and returns 
    the URL associated with the highest resolution/width.
    """
    if not srcset:
        return None

    # Split the srcset string into individual source entries
    sources = [s.strip() for s in srcset.split(',') if s.strip()]
    
    best_url = None
    max_value = -1

    for source in sources:
        # Regex to capture the URL and the descriptor (e.g., '1x', '800w')
        match = re.match(r"(\S+)\s+(\d+)([wx])", source)
        
        if match:
            url, value_str, descriptor = match.groups()
            value = int(value_str)
            
            # Prioritize based on the largest 'w' (width) or 'x' (pixel density) value
            if value > max_value:
                max_value = value
                best_url = url
                
    # If no descriptor was found (e.g., the string was just a single URL), 
    # fall back to the first URL found.
    if not best_url and sources:
        # Tries to find the URL without a descriptor
        return sources[0].split()[0]
        
    return best_url

print("Collecting SECOND image URLs (Slide Index 1) and Product IDs for Chloé...")
products_to_download = {} # {product_code: image_url}
product_idx = 1
MAX_WAIT_TIME = 10 
# Target Slide Index 1 corresponds to the SECOND image (0 is the first)
TARGET_SLIDE_INDEX = 1 

# Assuming TARGET_SLIDE_INDEX is still intended to represent the SECOND image (index 1),
# but the logic below relies on the specific class for the second image/view, which you provided.

# --- PRODUCT IMAGE EXTRACTION LOGIC (REVISED FOR SPECIFIC IMG CLASS) ---

try:
    # 1. Find all product containers (assuming this is the correct grid item wrapper)
    # Ensure 'div.product' is correct for the main item wrapper.
    grid_items = driver.find_elements(By.CSS_SELECTOR, "div.product") 
    print(f"Number of product containers found: {len(grid_items)}")
    
    # 2. Iterate over each container
    for item in grid_items:
        img_url = ""
        
        # *** CRITICAL STEP: Force product into view to trigger high-res lazy load ***
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item)
        time.sleep(0.5) 

        # --- IMAGE SELECTION LOGIC (Targeting the specific class) ---
        try:
            # Selector: Directly targets the specific image tag with the class you provided.
            # This selector is much simpler and focuses only on the image element.
            img_selector = 'img.tile-image.tile-image-2'
            
            # Find the image element for the second view
            img = item.find_element(By.CSS_SELECTOR, img_selector)
            
            # Dynamically wait until the 'srcset' attribute has meaningful high-quality content
            # (Keeping the robust wait logic)
            try:
                wait = WebDriverWait(driver, MAX_WAIT_TIME)
                wait.until(
                    lambda d: img.get_attribute("srcset") and len(img.get_attribute("srcset")) > 10
                )
            except TimeoutException:
                pass # If timeout, proceed with whatever is loaded

            # --- ROBUST EXTRACTION LOGIC ---
            img_url = None
            srcset = img.get_attribute("srcset")
            
            # PRIORITY 1: Use the helper function for the highest resolution URL from srcset
            if srcset:
                # IMPORTANT: Ensure 'get_highest_res_url' is defined elsewhere in your script.
                img_url = get_highest_res_url(srcset) 

            # PRIORITY 2: Fallback to 'src'
            if not img_url or "placeholder" in str(img_url).lower():
                img_url = img.get_attribute("src")

            # Final Validation
            if not img_url or "placeholder" in str(img_url).lower() or not str(img_url).startswith("http"):
                print(f"  Skipping item {product_idx}: No valid image URL found for Slide 2.")
                product_idx += 1 
                continue 
            
            # --- LOGIC TO EXTRACT THE PRODUCT CODE (Adjusted for Chloé) ---
            product_code = f"CHLOE_{product_idx}_VIEW2" # Use index as robust ID

            # If the extraction was successful, add the URL to the dictionary
            products_to_download[product_code] = img_url
            
            print(f"  ✅ Product {product_idx} - Second Image URL obtained: {img_url[:60]}...")
            
            product_idx += 1 


        except NoSuchElementException:
            # This handles cases where the product container exists, but the 'tile-image-2' element doesn't.
            print(f"  Product {product_idx}: Image element with class 'tile-image-2' not found in container. Skipping.")
            product_idx += 1 
            continue
        except Exception as e:
            print(f"  Unexpected error on item {product_idx}: {e}")
            product_idx += 1 
            continue


except Exception as e:
    print(f"General error locating product containers: {e}")

print(f"Collection finished. Total products with second image URLs: {len(products_to_download)}")


# --- 4. PRICE COLLECTION ---
print("Collecting product prices...")
products_prices = []

try:
    # Selector for prices (assuming price span text contains the price)
    price_items = driver.find_elements(By.CSS_SELECTOR, "span.sales")
    print(f"Number of prices found: {len(price_items)}")

    for item in price_items:
        price = item.text.strip()
        products_prices.append(price)

except Exception as e:
    print(f"Error locating prices: {e}")


# --- 5. IMAGE DOWNLOAD & CSV CREATION ---
image_folder = "images_Chloe"
os.makedirs(image_folder, exist_ok=True)

csv_path = os.path.join(image_folder, "chloe_products.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product_ID", "Price"])

    # Loop through the collected image URLs and their associated prices
    # We use enumerate over the dictionary keys to maintain consistency and generate an index.
    for idx, (product_code, img_url) in enumerate(products_to_download.items(), start=1):
        
        # Use a sequential ID for simplicity, but the product_code (key) could also be used directly
        # product_id = product_code  # Alternative: use the extracted code directly
        product_id = f"Chloe_{idx}"

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