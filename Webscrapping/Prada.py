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

##This one depends on the chromedriver path in your PC

service = Service("/Users/benja/tools/chromedriver")
driver = webdriver.Chrome(service=service)

url = "https://www.prada.com/it/it/mens/accessories/c/10156EU"

driver.get(url)
driver.maximize_window()

time.sleep(3)

## Accept Cookies
try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutto']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

## Sunglasses section

# Click CARICA ALTRO
pyautogui.moveTo(1200, 350, duration=1.5) 
time.sleep(1)
pyautogui.click(duration=0.5)
time.sleep(3) # Attesa dopo il clic


time.sleep(5)

   #. Scorrimento iniziale per rivelare il pulsante 'Carica Altro' **
print("Scrolling down to reveal the 'CLICK ALTER' button...")
driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.3);")

last_height = driver.execute_script("return document.body.scrollHeight")
products_count = 0

# To the bottom
time.sleep(2) # Caricamento
driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.7);")

# Click CARICA ALTRO
pyautogui.moveTo(860, 650, duration=1.5) 
time.sleep(1)
pyautogui.click(duration=0.5)
time.sleep(3) # Attesa dopo il clic

# To the top
driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.3);")
time.sleep(5) # Caricamento

PRODUCT_SELECTOR = "li.w-full.h-auto.lg\\:h-full"

while True:
 # To the bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.8);")
    time.sleep(5) # Caricamento

# Ottieni il numero attuale di prodotti sulla pagina utilizzando i selettori combinati
    current_products = driver.find_elements(By.CSS_SELECTOR, PRODUCT_SELECTOR)
    new_products_count = len(current_products)

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight*0.85);")
    time.sleep(5) # Caricamento

    # Ottieni il numero attuale di prodotti sulla pagina utilizzando i selettori combinati
    current_products = driver.find_elements(By.CSS_SELECTOR, PRODUCT_SELECTOR)
    new_products_count = len(current_products)

    if new_products_count == products_count:
        print(f"Number of products has not changed ({new_products_count}). Exiting the loop.")
        break
    else:
        print(f"Products loaded: {new_products_count}")
        products_count = new_products_count

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("The page height no longer changes. All products are visible.")
        break
    
    last_height = new_height

print("Scrolling finished. All products are visible.")
time.sleep(8)


# --- IMAGE COLLECTION ---
print("Collecting image URLs...")
products_to_download = {}

IMAGE_CONTAINER_SELECTOR = "picture.std-small\\:block.std-small\\:h-full.std-small\\:w-full.hidden.std-large\\:block"

try:
    # 1. Find all product containers
    grid_items = driver.find_elements(By.CSS_SELECTOR, IMAGE_CONTAINER_SELECTOR)
    print(f"Number of product containers found: {len(grid_items)}")

    products_to_download = {}

    # 2. Iterate over each container to find la imagen
    for idx, item in enumerate(grid_items, start=1):
        try:
            img_url = ""

            try:
                source = item.find_element(By.CSS_SELECTOR, 'source[media="(min-width: 1440px)"]')
                srcset = source.get_attribute("data-srcset")
                if not srcset:
                    srcset = source.get_attribute("srcset")
                
                if srcset:
                    urls_with_density = srcset.split(', ')
                    urls = [u.split(" ")[0] for u in urls_with_density]
                    # Assuming the third URL is the highest resolution
                    if len(urls) >= 3:
                        img_url = urls[2]
                        
                    else:
                        img_url = urls[-1]
                        
            except NoSuchElementException:
                continue
            
            if not img_url:
                continue

            # Save with name Prada_1, Prada_2...
            product_name = f"Prada_{idx}"
            products_to_download[product_name] = img_url
        except NoSuchElementException:
            continue

except Exception as e:
    print(f"Error locating products: {e}")

print(f"Total product image URLs found: {len(products_to_download)}")


# Price Extraction

print("Collecting product prices...")
products_data = []

try:
    # 1. Find all product containers
    grid_items = driver.find_elements(By.CSS_SELECTOR, PRODUCT_SELECTOR)
    print(f"Number of product containers found: {len(grid_items)}")

    products_data = []

    # 2. Iterate over each container to find price
    for idx, item in enumerate(grid_items, start=1):
        try:
            img_url = ""

            try:
                price_element = item.find_element(By.CSS_SELECTOR, "p.product-card__price--new")
                price = price_element.text.strip()
            except NoSuchElementException:
                price = "N/A"

            product_name = f"Prada_{idx}"

            products_data.append((product_name, price))

        except NoSuchElementException:
            continue

except Exception as e:
    print(f"Error locating products: {e}")

print(f"Total product prices found: {len(products_data)}")


#List
products_data.append((product_name, price))


# --- DESCARGA ---
image_folder = "images_Prada"
os.makedirs(image_folder, exist_ok=True)

count = 0
for name, img_url in products_to_download.items():
    img_path = os.path.join(image_folder, f"{name}.jpg")
    
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(response.content)
        print(f"Image saved: {name}.jpg")
        count += 1
    except Exception as e:
        print(f"Error downloading image from {img_url}: {e}")

# --- EXPORT CSV ---
csv_path = os.path.join(image_folder, "prada_products.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Price"])
    writer.writerows(products_data)

print(f"CSV saved at {csv_path}")

driver.quit()
print(f"Image scraping completed: {count} Prada images saved.")

