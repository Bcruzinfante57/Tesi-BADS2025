from selenium import webdriver
from selenium.webdriver.chrome.service import Service
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
from selenium.webdriver.chrome.options import Options
from typing import Optional

## --- HELPER FUNCTION FOR ROBUST SRCSET ANALYSIS ---
def get_highest_res_url(srcset_value: Optional[str]) -> Optional[str]:
    """
    Analyzes an srcset string (e.g., 'url1 300w, url2 1200w') and returns the URL 
    associated with the largest width (w) or density (x) descriptor.
    This ensures the highest possible resolution is selected, regardless of list order.
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
                # Prioritize 'w' (width) analysis over 'x' (density)
                if 'w' in descriptor:
                    descriptor_value = float(descriptor.replace('w', ''))
                elif 'x' in descriptor:
                    descriptor_value = float(descriptor.replace('x', ''))
            except ValueError:
                continue

        # If a higher descriptor value is found, update the best URL
        if descriptor_value > max_descriptor_value:
            max_descriptor_value = descriptor_value
            best_url = url
            
    return best_url

##This one depends on the chromedriver path in your PC
service = Service("/Users/benja/tools/chromedriver")
url = "https://www.fendi.com/it-it/search?q=occhiali&lang=it_IT"

options = Options()
# Arguments to prevent Selenium bot detection (key factor)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
# Argument to simulate a real browser (standard User-Agent)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
options.add_argument("--start-maximized") # Starting maximized helps ensure correct rendering

driver = webdriver.Chrome(options=options)
driver.get(url)

time.sleep(3)

## Accept Cookies
try:
        # Wait until the 'Accept All Cookies' button is clickable using its normalized text
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetto tutti i cookie']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

   #. Initial scroll to reveal the 'Load More' button **
print("Scrolling down to reveal the 'CARICA ALTRO' button...")
last_height = driver.execute_script("return document.body.scrollHeight")
products_count = 0
MAX_SCROLL_LOOPS = 20 # Safety break for infinite scroll
loop_counter = 0

while True:
    loop_counter += 1
    if loop_counter > MAX_SCROLL_LOOPS:
        print(f"Reached max scroll loops ({MAX_SCROLL_LOOPS}). Exiting scroll.")
        break
        
    # Scroll to the bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3) # Wait for content loading

    # Click CARICA ALTRO (Using PyAutoGUI is fragile, consider using Selenium click if possible)
    print("Attempting to click 'CARICA ALTRO'...")
    try:
        # Better approach: Find and click the button via Selenium
        load_more_btn = driver.find_element(By.CSS_SELECTOR, "button.load-more-btn") 
        driver.execute_script("arguments[0].click();", load_more_btn) # Use JS click for robustness
        print("Clicked 'CARICA ALTRO' using Selenium/JS.")
        time.sleep(5) # Wait after click for new products to load
    except NoSuchElementException:
        # Fallback to PyAutoGUI if Selenium click fails or button not found (original logic)
        print("Button not found via CSS. Falling back to PyAutoGUI coordinates.")
        pyautogui.moveTo(860, 250, duration=1.5)
        time.sleep(1)
        pyautogui.click(duration=0.5)
        time.sleep(5) # Wait after click

    # Get the current number of products on the page
    current_products = driver.find_elements(By.CSS_SELECTOR, "div.product")
    new_products_count = len(current_products)

    if new_products_count == products_count:
        print(f"Product count has not changed ({new_products_count}). Exiting scroll loop.")
        break
    else:
        print(f"Products loaded: {new_products_count}")
        products_count = new_products_count

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("Page height no longer changes. All products visible.")
        break
    
    last_height = new_height

print("Scrolling finished. All products should be visible.")
time.sleep(8) # Final wait for all elements to settle

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
            # --- IMAGE EXTRACTION (DYNAMIC WAIT FOR SRCSET) ---
            # Selector for the <img> within the visible product area
            img = item.find_element(By.CSS_SELECTOR, ".image-container img")
            
            # 🌟 DYNAMIC SRCSET WAIT: Wait until the 'srcset' attribute has content (indicating multiple resolutions are available)
            print(f"  Waiting dynamically up to {MAX_WAIT_TIME}s for srcset population (Item {product_idx})...")
            
            try:
                # The wait checks if the element's 'srcset' attribute is not empty and not None
                wait = WebDriverWait(driver, MAX_WAIT_TIME)
                wait.until(
                    lambda d: img.get_attribute("srcset") and len(img.get_attribute("srcset")) > 10
                ) # Check for a reasonable length to ensure it's not a minimal placeholder
                print("  High-res srcset data detected.")
            except TimeoutException:
                # If srcset fails to load, we still attempt extraction using the best available data
                print(f"  Timeout ({MAX_WAIT_TIME}s) reached. Srcset may not have loaded. Proceeding with best available URL.")

            # --- ROBUST EXTRACTION LOGIC (ALWAYS PRIORITIZE SRCSET) ---
            img_url = None

            # PRIORITY 1: Check 'srcset' and 'data-srcset' for the highest resolution
            srcset = img.get_attribute("srcset")
            data_srcset = img.get_attribute("data-srcset")
            
            # Use the robust helper function on the live srcset first
            if srcset:
                img_url = get_highest_res_url(srcset) 
            
            # Fallback to data-srcset if live srcset failed to provide a URL
            if (not img_url or "placeholder" in str(img_url).lower()) and data_srcset:
                img_url = get_highest_res_url(data_srcset) 

            # PRIORITY 2: Fallback to 'src' or 'data-src' if srcset logic failed entirely
            if not img_url or "placeholder" in str(img_url).lower():
                img_url = img.get_attribute("src")
            
            if not img_url or "placeholder" in str(img_url).lower():
                img_url = img.get_attribute("data-src")
            
            # Final Validation: Check for useful URL
            if not img_url or "placeholder" in str(img_url).lower() or not str(img_url).startswith("http"):
                print(f"  Skipping item {product_idx}: No valid image URL found.")
                continue 

            # --- PRICE EXTRACTION ---
            # Selector for the price element
            price_selector = ".price" 
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
                product_name = f"FENDI_{product_idx}"
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
image_folder = "images_Fendi"
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
csv_path = os.path.join(image_folder, "Fendi_products.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Price"])
    # Write only the name and price
    writer.writerows([(name, p) for name, p, url in products_data]) 

print(f"✅ CSV saved at {csv_path}")
print(f"✅ Scraping completed: {count} images saved + {len(products_data)} prices scraped.")

driver.quit()
