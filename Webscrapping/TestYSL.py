
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
url = "https://www.ysl.com/it-it/search?q=occhiali%20da%20sole"


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
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti i cookie']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

time.sleep(3)

   # **4. Desplazamiento inicial para revelar el botón 'Vedi tutto' **
print("Desplazándose hacia abajo para revelar el botón 'Vedi tutto'...")
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(5) # Espera a que el botón se cargue


# 5. Encontrar el botón y simular el clic de un mouse
print("Buscando y simulando clic de mouse en el botón 'Vedi tutto'...")
total_products_count = None
try:
        
        # Mover el cursor a las coordenadas fijas
        pyautogui.moveTo(860, 350, duration=1.5) # Usa las coordenadas X e Y que has probado
        
        time.sleep(1)
        pyautogui.click(duration=0.3)
        pyautogui.click(duration=0.3)
        print("Botón 'Vedi tutto' clicado con PyAutoGUI. Cargando todos los productos.")
        time.sleep(5)
        

        vedi_tutto_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//*[contains(normalize-space(), 'Vedi tutto')]"))
        )
        total_products_text = vedi_tutto_button.text
        match = re.search(r'\((\d+)\)', total_products_text)
        if match:
            total_products_count = int(match.group(1))

except (TimeoutException, NoSuchElementException, ElementNotInteractableException):
        print("Botón 'Vedi tutto' no encontrado. Asumiendo que todos los productos ya están cargados.")


 # **6. Desplazamiento final para cargar todos los productos**
if total_products_count:
        print(f"Esperando hasta que se carguen los {total_products_count} productos.")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            grid_items = driver.find_elements(By.CSS_SELECTOR, "li[id^='grid-item-']")
            if len(grid_items) >= total_products_count:
                print(f"Número de productos cargados: {len(grid_items)}. ¡Éxito!")
                break
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("No se cargan más productos. Saliendo del bucle.")
                break
            last_height = new_height
else:
        print("No se pudo determinar el número total de productos. Desplazándose al final para asegurar la carga.")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("Desplazamiento finalizado. Todos los productos visibles.")
                break
            last_height = new_height

time.sleep(10)

# --- CONFIGURACIÓN DE DATOS ---
print("Collecting image URLs and prices for YSL...")
products_data = []  # Lista para almacenar (nombre_archivo, precio, img_url)
product_idx = 1     # Contador para el nombre del archivo (YSL_1, YSL_2, ...)

try:
    # 1. Encontrar todos los contenedores de productos (las tarjetas individuales)
    grid_items = driver.find_elements(By.CSS_SELECTOR, "li[id^='grid-item-']")
    print(f"Número de contenedores de productos encontrados: {len(grid_items)}")
    
    # 2. Iterar sobre cada contenedor
    for item in grid_items:
        img_url = ""
        price = "N/A"
        
        # Intentamos obtener la imagen y el precio dentro del mismo contenedor 'item'
        try:
            # --- IMAGE EXTRACTION ---
            img = item.find_element(By.TAG_NAME, "img")
            
            # Obtener la URL de alta resolución del srcset
            srcset = img.get_attribute("srcset")
            if srcset:
                urls = srcset.split(', ')
                # Intentamos encontrar la URL con la resolución más alta (generalmente la última)
                img_url = urls[-1].split(' ')[0]
            else:
                img_url = img.get_attribute("src")

            # Validar que tengamos una URL
            if not img_url:
                continue 

            # --- PRICE EXTRACTION (USANDO data-qa con Regex) ---
            # El selector busca cualquier elemento cuyo atributo 'data-qa' comience con "plp-product-price-"
            price_selector = "[data-qa^='plp-product-price-']"
            try:
                price_element = item.find_element(By.CSS_SELECTOR, price_selector)
                price = price_element.text.strip()
            except NoSuchElementException:
                price = "N/A" # No se encontró el precio, se mantiene 'N/A'

            
            # --- GUARDAR DATOS Y NOMBRAR ---
            if img_url:
                product_name = f"YSL_{product_idx}"
                products_data.append((product_name, price, img_url))
                product_idx += 1 # Aumentar el índice solo si el producto es válido
            else:
                print(f"Skipping item: No valid image URL found.")
                
        except NoSuchElementException:
            # Si un contenedor no tiene la estructura básica (imagen), se salta
            continue 

except Exception as e:
    print(f"Error general durante la recolección de datos: {e}")

print(f"Total products scraped and ready for download: {len(products_data)}")

# --- DESCARGA DE IMÁGENES Y PREPARACIÓN CSV ---
image_folder = "images_ysl"
os.makedirs(image_folder, exist_ok=True)

count = 0
for product_name, price, img_url in products_data:
    img_path = os.path.join(image_folder, f"{product_name}.jpg")
    
    try:
        # Añadir un timeout para evitar que la descarga se quede colgada
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
    writer.writerow(["Product Name", "Price"])
    # Escribimos solo el nombre y el precio
    writer.writerows([(name, p) for name, p, url in products_data]) 

print(f"✅ CSV saved at {csv_path}")
print(f"✅ Scraping completed: {count} images saved + {len(products_data)} prices scraped.")

driver.quit()

