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

url = "https://www.dolcegabbana.com/it-it/moda/uomo/occhiali-da-sole/"

driver.get(url)
driver.maximize_window()

time.sleep(3)

## Accept Cookies
try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'aceptar todas las cookies']"))
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
    pyautogui.moveTo(860, 550, duration=1.5) # Usa las coordenadas X e Y que has probado
    time.sleep(1)
    pyautogui.click(duration=0.5)
    time.sleep(5) # Attesa dopo il clic

    # Ottieni il numero attuale di prodotti sulla pagina utilizzando i selettori combinati
    current_products = driver.find_elements(By.CSS_SELECTOR, "div.SearchHitsItem__search-hit--Mnk4L")
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


# --- RECOLECCIÓN DE IMÁGENES ---
print("Collecting image URLs...")
products_to_download = {}

try:
    # 1. Encontrar todos los contenedores de productos
    grid_items = driver.find_elements(By.CSS_SELECTOR, "a.ProductMedia__product-media__image-wrapper--HoWNY.product-media__image-wrapper")
    print(f"Número de contenedores de productos encontrados: {len(grid_items)}")

    # 2. Iterar sobre cada contenedor para encontrar la imagen
    for item in grid_items:
        try:
            # Encontrar el elemento <img> dentro del contenedor
            img = item.find_element(By.TAG_NAME, "img")

            # Obtener la URL de alta resolución del srcset o src
            img_url = ""
            srcset = img.get_attribute("srcset")
            if srcset:
                urls = srcset.split(', ')
                img_url = urls[-1].split(' ')[0]
            else:
                img_url = img.get_attribute("src")

            if not img_url:
                continue

            # --- LÓGICA PARA EXTRAER EL CÓDIGO DEL PRODUCTO (MÁS ROBUSTA) ---
            product_code = None
            
            # Patrón para Dolce & Gabbana
            dolce_match = re.search(r'images/zoom/(\S+)\?', img_url)
            if dolce_match:
                product_code = dolce_match.group(1).replace('.jpg', '')

            if product_code:
                products_to_download[product_code] = img_url

        except NoSuchElementException:
            continue

except Exception as e:
    print(f"Error localizando productos: {e}")

print(f"Total product image URLs found: {len(products_to_download)}")

# --- RECOLECCIÓN DE PRECIOS ---
print("Collecting product prices...")
products_prices = []

try:
    price_items = driver.find_elements(By.CSS_SELECTOR, "span.money.ProductPriceDiscount__product-price__discount--Acusx.product-price__discount.SearchHitsItem__search-hit__price-item--mBvPF")
    print(f"Número de precios encontrados: {len(price_items)}")

    for item in price_items:
        price = item.text.strip()
        products_prices.append(price)

except Exception as e:
    print(f"Error localizando precios: {e}")


# --- DESCARGA DE IMÁGENES + CREACIÓN CSV ---
image_folder = "images_D&G"
os.makedirs(image_folder, exist_ok=True)

csv_path = os.path.join(image_folder, "dolcegabbana_products.csv")

with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Product_ID", "Price"])

    for idx, img_url in enumerate(products_to_download, start=1):
        product_id = f"D&G_{idx}"
        price = products_prices[idx-1] if idx-1 < len(products_prices) else "N/A"

        # Descargar imagen
        img_path = os.path.join(image_folder, f"{product_id}.jpg")
        try:
            img_data = requests.get(img_url).content
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            print(f"Image saved: {product_id}.jpg")
        except Exception as e:
            print(f"Error downloading image from {img_url}: {e}")

        # Escribir en CSV
        writer.writerow([product_id, price])

driver.quit()
print(f"✅ Scraping completed: {len(products_to_download)} products saved with prices.")
print(f"✅ CSV saved at {csv_path}")