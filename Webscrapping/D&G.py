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

url = "https://www.dolcegabbana.com/it-it/moda/uomo/occhiali-da-sole/"

driver.get(url)
driver.maximize_window()

time.sleep(3)

# --- 1. ACCEPT COOKIES ---
try:
        # XPATH translated for the expected "Accept All Cookies" button text (case-insensitive)
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'aceptar todas las cookies' or normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accept all cookies']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

# --- 2. INFINITE SCROLL AND "LOAD MORE" CLICKS ---
print("Scrolling down to reveal the 'CARICA ALTRO' (Load More) button...")
last_height = driver.execute_script("return document.body.scrollHeight")
products_count = 0
while True:
    # Scroll to the bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3) # Wait for content to load

    # Click CARICA ALTRO using PyAutoGUI (based on fixed screen coordinates)
    # WARNING: PyAutoGUI clicks are reliant on screen resolution and window position.
    pyautogui.moveTo(860, 550, duration=1.5) # Use the tested X and Y coordinates
    time.sleep(1)
    pyautogui.click(duration=0.5)
    time.sleep(5) # Wait after the click for new products to load

    # Get the current count of products
    current_products = driver.find_elements(By.CSS_SELECTOR, "div.SearchHitsItem__search-hit--Mnk4L")
    new_products_count = len(current_products)

    if new_products_count == products_count:
        print(f"Product count has not changed ({new_products_count}). Exiting loop.")
        break
    else:
        print(f"Products loaded: {new_products_count}")
        products_count = new_products_count

    # Check page height change
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("Page height is no longer changing. All products visible.")
        break
    
    last_height = new_height

print("Scrolling finished. All visible products loaded.")
time.sleep(8)


# --- 3. IMAGE URL AND PRODUCT ID COLLECTION ---
print("Collecting image URLs and Product IDs...")
products_to_download = {} # {product_code: image_url}

try:
    # 1. Find all product media wrappers (containers)
    grid_items = driver.find_elements(By.CSS_SELECTOR, "a.ProductMedia__product-media__image-wrapper--HoWNY.product-media__image-wrapper")
    print(f"Number of product containers found: {len(grid_items)}")

    # 2. Iterate over each container to find the image
    for item in grid_items:
        try:
            # Find the <img> element inside the container
            img = item.find_element(By.TAG_NAME, "img")

            # Get the high-resolution URL from srcset or src
            img_url = ""
            srcset = img.get_attribute("srcset")
            if srcset:
                # Typically, the last URL in srcset is the largest
                urls = srcset.split(', ')
                img_url = urls[-1].split(' ')[0]
            else:
                img_url = img.get_attribute("src")

            if not img_url:
                continue

            # --- LOGIC TO EXTRACT THE PRODUCT CODE (ROBUST D&G PATTERN) ---
            product_code = None
            
            # Pattern specific to Dolce & Gabbana image URLs (extracting the identifier after /zoom/)
            dolce_match = re.search(r'images/zoom/(\S+)\?', img_url)
            if dolce_match:
                product_code = dolce_match.group(1).replace('.jpg', '')

            # If a code is found, add to the dictionary
            if product_code:
                products_to_download[product_code] = img_url

        except NoSuchElementException:
            continue

except Exception as e:
    print(f"Error locating products: {e}")

print(f"Total product image URLs found: {len(products_to_download)}")

# --- 4. PRICE COLLECTION ---
print("Collecting product prices...")
products_prices = []

try:
    # Selector for prices (assuming price span text contains the price)
    price_items = driver.find_elements(By.CSS_SELECTOR, "span.money.ProductPriceDiscount__product-price__discount--Acusx.product-price__discount.SearchHitsItem__search-hit__price-item--mBvPF")
    print(f"Number of prices found: {len(price_items)}")

    for item in price_items:
        price = item.text.strip()
        products_prices.append(price)

except Exception as e:
    print(f"Error locating prices: {e}")


# --- 5. IMAGE DOWNLOAD & CSV CREATION ---
image_folder = "images_D&G"
os.makedirs(image_folder, exist_ok=True)

csv_path = os.path.join(image_folder, "dolcegabbana_products.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product_ID", "Price"])

    # Loop through the collected image URLs and their associated prices
    # We use enumerate over the dictionary keys to maintain consistency and generate an index.
    for idx, (product_code, img_url) in enumerate(products_to_download.items(), start=1):
        
        # Use a sequential ID for simplicity, but the product_code (key) could also be used directly
        # product_id = product_code  # Alternative: use the extracted code directly
        product_id = f"D&G_{idx}"

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