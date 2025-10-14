from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.chrome.options import Options
from typing import Optional
import time
import os
import requests
import re   
import pyautogui
import csv

# --- SPECIAL CASE CONFIGURATION ---
# List of product indices (1-based) that require the THIRD image (slide index 2).
THIRD_IMAGE_INDICES = {23, 27, 32, 41, 135, 139, 140}


## --- HELPER FUNCTION FOR ROBUST SRCSET ANALYSIS ---
def get_highest_res_url(srcset_value: Optional[str]) -> Optional[str]:
    """
    Analyzes an srcset string (e.g., 'url1 300w, url2 1200w') and returns the URL 
    associated with the largest width (w) or density (x) descriptor.
    This ensures the highest possible resolution is selected.
    """
    if not srcset_value:
        return None

    best_url = None
    max_descriptor_value = 0.0

    entries = srcset_value.split(',')
    
    for entry in entries:
        parts = entry.strip().split()
        if len(parts) < 1:
            continue
        
        url = parts[0].strip()
        descriptor_value = 1.0 # Default value if no descriptor is found

        if len(parts) > 1:
            descriptor = parts[1].strip()
            try:
                # Prioritize 'w' (width)
                if 'w' in descriptor:
                    descriptor_value = float(descriptor.replace('w', ''))
                # Fallback to 'x' (density)
                elif 'x' in descriptor:
                    descriptor_value = float(descriptor.replace('x', ''))
            except ValueError:
                continue

        # If a higher descriptor value is found, update the best URL
        if descriptor_value > max_descriptor_value:
            max_descriptor_value = descriptor_value
            best_url = url
            
    return best_url

# --- DRIVER CONFIGURATION ---
# This one depends on the chromedriver path in your PC
# Replace this path with the location of your chromedriver
# NOTE: Using Service is the old way. Consider using `driver = webdriver.Chrome(options=options)` 
# without Service for automatic driver management.
service = Service("/Users/benja/tools/chromedriver") 
url = "https://www.ysl.com/it-it/search?q=occhiali%20da%20sole"


options = Options()
# Arguments to prevent Selenium bot detection (key factor)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
# Argument to simulate a real browser (Standard User-Agent)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
options.add_argument("--start-maximized") # Starting maximized helps proper rendering

driver = webdriver.Chrome(service=service, options=options)
driver.get(url)

time.sleep(3)

## Accept Cookies
try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti i cookie']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

# 4. Initial scroll to reveal the 'Vedi tutto' (See all) button
print("Scrolling down to reveal the 'Vedi tutto' button...")
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(5) # Wait for the button to load


# 5. Find the button and simulate a mouse click (Original logic conserved)
print("Searching for and simulating mouse click on the 'Vedi tutto' button...")
total_products_count = None
try:
        # Try to get the product count first
        vedi_tutto_button_element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//*[contains(normalize-space(), 'Vedi tutto')]"))
        )
        total_products_text = vedi_tutto_button_element.text
        match = re.search(r'\((\d+)\)', total_products_text)
        if match:
            total_products_count = int(match.group(1))
        
        # Move the cursor to fixed coordinates (using PyAutoGUI as in the original)
        # WARNING: PyAutoGUI is unreliable in automated environments!
        pyautogui.moveTo(860, 350, duration=1.5) # Use the tested X and Y coordinates
        
        time.sleep(1)
        pyautogui.click(duration=0.3)
        pyautogui.click(duration=0.3)
        print("'Vedi tutto' button clicked with PyAutoGUI. Loading all products.")
        time.sleep(5)
        
except (TimeoutException, NoSuchElementException, ElementNotInteractableException) as e:
        print(f"Error trying to click 'Vedi tutto': {e}. Assuming all products are already loaded.")


# 6. Final scroll to load all products (Original logic conserved)
if total_products_count:
        print(f"Waiting until all {total_products_count} products are loaded.")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            # Ensure the element selector is still 'li[id^='grid-item-']'
            grid_items = driver.find_elements(By.CSS_SELECTOR, "li[id^='grid-item-']") 
            if len(grid_items) >= total_products_count:
                print(f"Products loaded count: {len(grid_items)}. Success!")
                break
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("No more products are loading. Exiting loop.")
                break
            last_height = new_height
else:
        print("Could not determine total product count. Scrolling to end to ensure loading.")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("Scrolling finished. All products visible.")
                break
            last_height = new_height

time.sleep(10)

# --- DATA COLLECTION CONFIGURATION (MODIFIED EXTRACTION LOGIC) ---
print("Collecting image URLs and prices for YSL...")
products_data = []  # List to store (file_name, price, img_url)
product_idx = 1     # Counter for file name (YSL_1, YSL_2, ...)
MAX_WAIT_TIME = 10 

try:
    # 1. Find all product containers (individual cards)
    grid_items = driver.find_elements(By.CSS_SELECTOR, "li[id^='grid-item-']")
    print(f"Number of product containers found: {len(grid_items)}")
    
    # 2. Iterate over each container
    for item in grid_items:
        img_url = ""
        price = "N/A"
        
        # *** CRITICAL STEP: Force product into view to trigger high-res lazy load ***
        # Ensure the product is visible so its content loads
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item)
        time.sleep(0.5) 

        # --- SLIDE SELECTION LOGIC (NEW) ---
        slide_index = 1 # Default: use the second slide (index 1)
        if product_idx in THIRD_IMAGE_INDICES:
            slide_index = 2 # Special case: use the third slide (index 2)
            print(f"  --> Product {product_idx} identified as a special case. Using Slide Index 2.")
        else:
            # print(f"  --> Product {product_idx}: Using Slide Index 1 (default).")
            pass

        try:
            # Dynamic Selector: searches for the image within the DIV representing the slide_index
            img_selector = f'div[data-swiper-slide-index="{slide_index}"] img'
            img = item.find_element(By.CSS_SELECTOR, img_selector)
            
            # Dynamically wait until the 'srcset' attribute has significant content
            try:
                wait = WebDriverWait(driver, MAX_WAIT_TIME)
                wait.until(
                    lambda d: img.get_attribute("srcset") and len(img.get_attribute("srcset")) > 10
                )
            except TimeoutException:
                pass # If timeout, just use whatever loaded

            # --- ROBUST EXTRACTION LOGIC (USING HELPER FUNCTION) ---
            img_url = None
            srcset = img.get_attribute("srcset")
            
            # PRIORITY 1: Use helper function for max resolution from srcset
            if srcset:
                img_url = get_highest_res_url(srcset) 

            # PRIORITY 2: Fallback to 'src'
            if not img_url or "placeholder" in str(img_url).lower():
                img_url = img.get_attribute("src")

            # Final Validation
            if not img_url or "placeholder" in str(img_url).lower() or not str(img_url).startswith("http"):
                print(f"  Skipping item {product_idx}: No valid image URL found for slide {slide_index + 1}.")
                # Important: We increment the index here to keep the counter accurate
                product_idx += 1 
                continue 


            # --- PRICE EXTRACTION (USING data-qa with Regex) ---
            price_selector = "[data-qa^='plp-product-price-']"
            try:
                price_element = item.find_element(By.CSS_SELECTOR, price_selector)
                price = price_element.text.strip()
            except NoSuchElementException:
                price = "N/A" # Price not found

            
            # --- SAVE DATA AND NAME ---
            if img_url:
                product_name = f"YSL_{product_idx}"
                products_data.append((product_name, price, img_url))
                product_idx += 1 # Increment index only if product is valid
            else:
                print(f"Skipping item: No valid image URL found.")
                product_idx += 1 
                
        except NoSuchElementException:
            # If a container doesn't have the basic structure (image in the selected slide), skip
            product_idx += 1 
            continue 

except Exception as e:
    print(f"General error during data collection: {e}")

print(f"Total products scraped and ready for download: {len(products_data)}")

# --- IMAGE DOWNLOAD AND CSV PREPARATION ---
image_folder = "images_ysl"
os.makedirs(image_folder, exist_ok=True)

count = 0
# Iterate over products_data, which already contains only valid products
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
csv_path = os.path.join(image_folder, "ysl_products.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Price", "Image URL"])
    # Write name, price, and URL
    # Note: Added img_url to the writerow list for completeness
    writer.writerows([(name, p, u) for name, p, u in products_data]) 

print(f"✅ CSV saved at {csv_path}")
print(f"✅ Scraping completed: {count} images saved + {len(products_data)} prices scraped.")

driver.quit()