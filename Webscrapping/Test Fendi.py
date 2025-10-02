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

##This one depends on the chromedriver path in your PC

service = Service("/Users/benja/tools/chromedriver")
url = "https://www.fendi.com/it-it/search?q=occhiali&lang=it_IT"


options = Options()
# Argumentos para evitar la detección de bot de Selenium (el factor clave)
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
# Argumento para simular un navegador real (User-Agent estándar)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
options.add_argument("--start-maximized") # Iniciar maximizado ayuda a renderizar correctamente

driver = webdriver.Chrome(options=options)
driver.get(url)

time.sleep(3)

## Accept Cookies
try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetto tutti i cookie']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

   #. Scorrimento iniziale per rivelare il pulsante 'Carica Altro' **
print("Desplazándose hacia abajo para revelar el botón 'CARICA ALTRO'...")
last_height = driver.execute_script("return document.body.scrollHeight")
products_count = 0
while True:
    # To the bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3) # Caricamento

    # Click CARICA ALTRO
    pyautogui.moveTo(860, 250, duration=1.5)
    time.sleep(1)
    pyautogui.click(duration=0.5)
    time.sleep(5) # Attesa dopo il clic

    # Ottieni il numero attuale di prodotti sulla pagina utilizzando i selettori combinati
    current_products = driver.find_elements(By.CSS_SELECTOR, "div.product")
    new_products_count = len(current_products)

    if new_products_count == products_count:
        print(f"Número de productos no ha cambiado ({new_products_count}). Saliendo del bucle.")
        break
    else:
        print(f"Productos cargados: {new_products_count}")
        products_count = new_products_count

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("La altura de la página ya no cambia. Todos los productos visibles.")
        break
    
    last_height = new_height

print("Desplazamiento finalizado. Todos los productos visibles.")
time.sleep(8)

# --- DATA COLLECTION CONFIGURATION ---
print("Collecting image URLs and prices for Fendi...")
products_data = []  # List to store (file_name, price, img_url)
product_idx = 1     # Counter for the sequential file name (FENDI_1, FENDI_2, ...)

try:
    # 1. Find all product containers (using the same selector as the scroll loop)
    # *CORRECTION: We use div.product instead of the YSL selector*
    grid_items = driver.find_elements(By.CSS_SELECTOR, "div.product")
    print(f"Number of product containers found: {len(grid_items)}")
    
    # 2. Iterate over each container
    for item in grid_items:
        img_url = ""
        price = "N/A"
        
        try:
            # --- IMAGE EXTRACTION (CORRECTED AND ADAPTED FOR LAZY LOADING) ---
            # *CORRECTION: We search srcset/src first, then data-srcset/data-src*
            img = item.find_element(By.CSS_SELECTOR, "img")
            
            # Attempt 1: Get the high-resolution URL from 'srcset' or 'src' (if already loaded)
            srcset = img.get_attribute("srcset")
            if srcset:
                urls = srcset.split(', ')
                # Try to find the highest resolution URL (usually the last one)
                img_url = urls[-1].split(' ')[0]
            else:
                img_url = img.get_attribute("src")

            # Attempt 2: If Attempt 1 failed (lazy loading), search in 'data-' attributes
            if not img_url or "placeholder" in img_url.lower(): # Check if it's empty or a placeholder
                data_srcset = img.get_attribute("data-srcset")
                data_src = img.get_attribute("data-src")
                
                if data_srcset:
                    urls = data_srcset.split(', ')
                    img_url = urls[-1].split(' ')[0] # Use the highest resolution version from data-srcset
                elif data_src:
                    img_url = data_src
            
            # Validate that we have a useful URL
            if not img_url or "placeholder" in img_url.lower() or not img_url.startswith("http"):
                print(f"Skipping item: No valid image URL found after checking src/srcset/data-src/data-srcset.")
                continue 

            # --- PRICE EXTRACTION (CORRECTED) ---
            # *CORRECTION: We use a general selector for Fendi (you must verify the exact class)*
            # We assume the price is in an element with the class 'price'
            price_selector = ".price" # Or: span.product-price, .price-box__price, etc.
            try:
                price_element = item.find_element(By.CSS_SELECTOR, price_selector)
                price = price_element.text.strip()
            except NoSuchElementException:
                price = "N/A" # Price not found, defaults to 'N/A'

            
            # --- SAVE DATA AND NAME ---
            if img_url:
                product_name = f"FENDI_{product_idx}" # *NAME CHANGE: YSL -> FENDI*
                products_data.append((product_name, price, img_url))
                product_idx += 1 # Increase index only if the product is valid
            else:
                # This else should no longer execute if the upper filter works, but it remains as a safety net.
                print(f"Skipping item: No valid image URL found (Final Check).")
                
        except NoSuchElementException:
            # If a container doesn't have the basic structure (image or price), skip it
            continue 

except Exception as e:
    print(f"General error during data collection: {e}")

print(f"Total products scraped and ready for download: {len(products_data)}")

# --- IMAGE DOWNLOAD AND CSV PREPARATION ---
image_folder = "images_Fendi" # *FOLDER NAME CHANGE*
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
csv_path = os.path.join(image_folder, "Fendi_products.csv") # *CSV NAME CHANGE*
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Price"])
    # Write only the name and price
    writer.writerows([(name, p) for name, p, url in products_data]) 

print(f"✅ CSV saved at {csv_path}")
print(f"✅ Scraping completed: {count} images saved + {len(products_data)} prices scraped.")

driver.quit()