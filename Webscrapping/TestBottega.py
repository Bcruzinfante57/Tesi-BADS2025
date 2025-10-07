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

url = "https://www.bottegaveneta.com/it-it/search?q=occhiali"

driver.get(url)
driver.maximize_window()

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

# Scrolling to simulate a human
driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.4);")
print("Desplazando 60%...")
time.sleep(5)

print("Iniciando desplazamiento para asegurar que todos los productos se carguen.")
time.sleep(2) # Espera inicial

# Main loop for scrolling
last_height = driver.execute_script("return document.body.scrollHeight* 0.6")
products_count = 0

while True:
    # Short Scroll
    driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
    time.sleep(3)

    # Short Scroll
    driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
    time.sleep(3)

    # Short Scroll
    driver.execute_script("window.scrollBy(0, window.innerHeight * 0.8);")
    time.sleep(3)

    # Get the current number of products on the page using combo selectors
    current_products = driver.find_elements(By.CSS_SELECTOR, "article.c-product")
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
time.sleep(10)


# --- IMAGE & PRICE COLLECTION ---
print("Collecting product data...")
# Cambiamos 'products' para que almacene el nombre, el precio Y la URL de la imagen
products = [] 

cards = driver.find_elements(By.CSS_SELECTOR, "article.c-product[data-pid]")
print(f"Número de cards de productos: {len(cards)}")

# *1. INICIALIZAMOS EL CONTADOR MANUALMENTE EN 1
product_idx = 1

# *2. ITERAMOS SOBRE LAS TARJETAS SIN USAR ENUMERATE
for card in cards:
    # *3. La espera es local a la tarjeta, no depende del índice
    wait = WebDriverWait(card, 5) 
    img_url = ""
    price = "N/A"
    
    try:
        # --- IMAGE ---
        active_img_selector = (
            "ul.c-product__carousel "
            "li.c-product__carousel--slide.swiper-slide-active "
            "img.c-product__image"
        )
        fallback_img_selector = (
            "ul.c-product__carousel li.c-product__carousel--slide img.c-product__image"
        )

        img = None
        try:
            # Intentar encontrar la imagen activa (si es un carrusel)
            img = card.find_element(By.CSS_SELECTOR, active_img_selector)
        except NoSuchElementException:
            try:
                # Intentar encontrar cualquier imagen dentro del carrusel
                img = card.find_element(By.CSS_SELECTOR, fallback_img_selector)
            except NoSuchElementException:
                # Si no hay imagen, saltar el producto y NO aumentar el índice
                continue

        # *2. CRITICAL FIX: Wait for the 'srcset' attribute to be non-empty
        try:
            # Esperar hasta que el atributo 'srcset' tenga algún valor (indicando que cargó)
            wait.until(lambda d: img.get_attribute("srcset") or img.get_attribute("src"))
        except TimeoutException:
            # Si falla la espera, img_url será "", lo que se captura más abajo.
            # NO aumentar el índice aquí.
            print(f"Warning: Image for card index {product_idx} failed to load 'srcset' or 'src' attribute within 5 seconds.")
            pass

        # Take best quality from srcset
        srcset = img.get_attribute("srcset") or ""
        if srcset:
            parts = [p.strip() for p in srcset.split(",")]
            def width_of(p):
                toks = p.split()
                if len(toks) >= 2 and toks[1].endswith("w"):
                    try:
                        return int(toks[1][:-1])
                    except:
                        return 0
                return 0
            best = max(parts, key=width_of)
            img_url = best.split()[0]
        else:
            # Usar 'src' como fallback
            img_url = img.get_attribute("src")

        # --- PRICE ---
        try:
            price_element = card.find_element(By.CSS_SELECTOR, "p.c-price__value--current")
            price = price_element.text.strip()
        except NoSuchElementException:
            price = "N/A"

        # --- IDENTIFIER ---
        
        # *4. Guardar y aumentar el índice SOLO si la URL es válida
        if img_url:
            product_name = f"Bottega_{product_idx}" # Creamos el nombre con el índice actual
            products.append((product_name, price, img_url))
            product_idx += 1 # Aumentamos el índice SOLO si el producto se añade
        else:
            print(f"Skipping card: No valid image URL found (Index would have been {product_idx}).")
            
    except Exception as e:
        print(f"Error processing card (Index would have been {product_idx}): {e}")
        continue

print(f"Total products scraped: {len(products)}")

# --- DOWNLOAD IMAGES ---
image_folder = "images_bottega"
os.makedirs(image_folder, exist_ok=True)

count = 0
for product_name, price, img_url in products:
    img_path = os.path.join(image_folder, f"{product_name}.jpg")

    try:
        # Esta llamada ahora usa la img_url única para cada producto
        img_data = requests.get(img_url, timeout=10).content
        with open(img_path, "wb") as f:
            f.write(img_data)
        print(f"Image saved: {product_name}.jpg ({price})")
        count += 1
    except Exception as e:
        print(f"Error downloading image from {img_url}: {e}")

# --- EXPORT CSV ---
csv_path = os.path.join(image_folder, "bottega_products.csv")
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product Name", "Price"])
    # El CSV solo necesita el nombre y el precio, lo cual está bien
    writer.writerows([(name, p) for name, p, url in products]) 

print(f"✅ CSV saved at {csv_path}")
print(f"✅ Scraping completed: {count} images saved + {len(products)} prices scraped.")

driver.quit()