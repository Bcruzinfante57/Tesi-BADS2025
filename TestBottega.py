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

# Desplazamiento final para cargar todos los productos
print("Iniciando desplazamiento para asegurar que todos los productos se carguen.")
products_count = 0
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Desplazarse hasta el final de la página
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3) # Esperar a que se carguen los nuevos productos

    # Verificar si el número de productos ya no aumenta
    current_products = driver.find_elements(By.CSS_SELECTOR, "article.c-product")
    new_products_count = len(current_products)

    if new_products_count == products_count:
        print(f"Número de productos no ha cambiado ({new_products_count}). Saliendo del bucle.")
        break
    else:
        print(f"Productos cargados: {new_products_count}")
        products_count = new_products_count

    # Verificar si la altura de la página ha cambiado
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("La altura de la página ya no cambia. Todos los productos visibles.")
        break
    
    last_height = new_height

print("Desplazamiento finalizado. Todos los productos visibles.")

time.sleep(10)

# --- RECOLECCIÓN DE IMÁGENES ---
print("Collecting image URLs...")
products_to_download = {}

try:
    # Encontrar todos los contenedores de productos usando la clase c-product
    grid_items = driver.find_elements(By.CSS_SELECTOR, "article.c-product")
    print(f"Número de contenedores de productos encontrados: {len(grid_items)}")
    
    for item in grid_items:
        try:
            img = item.find_element(By.TAG_NAME, "img")
            
            srcset = img.get_attribute("srcset")
            if srcset:
                urls = srcset.split(', ')
                img_url = urls[-1].split(' ')[0]
            else:
                img_url = img.get_attribute("src")

            code_match = re.search(r'dam.kering.com/(\S+)\?', img_url)
            if img_url and "kering.com" in img_url and code_match:
                product_code = code_match.group(1).split('/')[-1].replace('.A.jpg', '')
                products_to_download[product_code] = img_url
        except NoSuchElementException:
            continue

except Exception as e:
    print(f"Error localizando productos: {e}")

print(f"Total product image URLs found: {len(products_to_download)}")

# --- DESCARGA ---
image_folder = "imagenes_bottega"
os.makedirs(image_folder, exist_ok=True)

count = 0
for code, img_url in products_to_download.items():
    img_path = os.path.join(image_folder, f"{code}.jpg")
    
    try:
        img_data = requests.get(img_url).content
        with open(img_path, "wb") as f:
            f.write(img_data)
        print(f"Image saved: {code}.jpg")
        count += 1
    except Exception as e:
        print(f"Error downloading image from {img_url}: {e}")

print(f"Image scraping completed for {count} products of Bottega Veneta.")

driver.quit()