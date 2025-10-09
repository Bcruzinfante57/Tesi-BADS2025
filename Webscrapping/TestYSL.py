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

# --- CONFIGURACIÓN DE CASOS ESPECIALES ---
# Lista de índices de productos (1-based) que requieren la TERCERA imagen (slide index 2).
# Los índices son: 23, 27, 32, 41, 135, 139, 140
THIRD_IMAGE_INDICES = {23, 27, 32, 41, 135, 139, 140}


## --- HELPER FUNCTION FOR ROBUST SRCSET ANALYSIS ---
def get_highest_res_url(srcset_value: Optional[str]) -> Optional[str]:
    """
    Analiza una cadena srcset (por ejemplo, 'url1 300w, url2 1200w') y devuelve la URL 
    asociada con el descriptor de ancho (w) o densidad (x) más grande.
    Esto asegura que se selecciona la máxima resolución posible.
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
        descriptor_value = 1.0 # Valor predeterminado si no se encuentra descriptor

        if len(parts) > 1:
            descriptor = parts[1].strip()
            try:
                # Prioriza 'w' (ancho)
                if 'w' in descriptor:
                    descriptor_value = float(descriptor.replace('w', ''))
                elif 'x' in descriptor:
                    descriptor_value = float(descriptor.replace('x', ''))
            except ValueError:
                continue

        # Si se encuentra un valor de descriptor más alto, actualiza la mejor URL
        if descriptor_value > max_descriptor_value:
            max_descriptor_value = descriptor_value
            best_url = url
            
    return best_url

# --- CONFIGURACIÓN DE DRIVER ---
##This one depends on the chromedriver path in your PC
# Reemplace esta ruta con la ubicación de su chromedriver
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


# 5. Encontrar el botón y simular el clic de un mouse (Lógica original conservada)
print("Buscando y simulando clic de mouse en el botón 'Vedi tutto'...")
total_products_count = None
try:
        # Intenta obtener el contador de productos primero
        vedi_tutto_button_element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//*[contains(normalize-space(), 'Vedi tutto')]"))
        )
        total_products_text = vedi_tutto_button_element.text
        match = re.search(r'\((\d+)\)', total_products_text)
        if match:
            total_products_count = int(match.group(1))
        
        # Mover el cursor a las coordenadas fijas (usando PyAutoGUI como en el original)
        pyautogui.moveTo(860, 350, duration=1.5) # Usa las coordenadas X e Y que has probado
        
        time.sleep(1)
        pyautogui.click(duration=0.3)
        pyautogui.click(duration=0.3)
        print("Botón 'Vedi tutto' clicado con PyAutoGUI. Cargando todos los productos.")
        time.sleep(5)
        
except (TimeoutException, NoSuchElementException, ElementNotInteractableException) as e:
        print(f"Error al intentar cliquear 'Vedi tutto': {e}. Asumiendo que todos los productos ya están cargados.")


 # **6. Desplazamiento final para cargar todos los productos (Lógica original conservada)**
if total_products_count:
        print(f"Esperando hasta que se carguen los {total_products_count} productos.")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            # Asegurar que el selector de elementos sigue siendo 'li[id^='grid-item-']'
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

# --- CONFIGURACIÓN DE DATOS (LÓGICA DE EXTRACCIÓN MODIFICADA) ---
print("Collecting image URLs and prices for YSL...")
products_data = []  # Lista para almacenar (nombre_archivo, precio, img_url)
product_idx = 1     # Contador para el nombre del archivo (YSL_1, YSL_2, ...)
MAX_WAIT_TIME = 10 

try:
    # 1. Encontrar todos los contenedores de productos (las tarjetas individuales)
    grid_items = driver.find_elements(By.CSS_SELECTOR, "li[id^='grid-item-']")
    print(f"Número de contenedores de productos encontrados: {len(grid_items)}")
    
    # 2. Iterar sobre cada contenedor
    for item in grid_items:
        img_url = ""
        price = "N/A"
        
        # *** CRITICAL STEP: Force product into view to trigger high-res lazy load ***
        # Aseguramos que el producto esté visible para que se cargue su contenido
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", item)
        time.sleep(0.5) 

        # --- LÓGICA DE SELECCIÓN DE SLIDE (NUEVA) ---
        slide_index = 1 # Por defecto, usamos el slide 2 (índice 1)
        if product_idx in THIRD_IMAGE_INDICES:
            slide_index = 2 # Si es un caso especial, usamos el slide 3 (índice 2)
            print(f"  --> Producto {product_idx} identificado como caso especial. Usando Slide Index 2.")
        else:
            # print(f"  --> Producto {product_idx}: Usando Slide Index 1 (por defecto).")
            pass

        try:
            # Selector dinámico: busca la imagen dentro del DIV que representa el slide_index
            img_selector = f'div[data-swiper-slide-index="{slide_index}"] img'
            img = item.find_element(By.CSS_SELECTOR, img_selector)
            
            # Esperar dinámicamente hasta que el atributo 'srcset' tenga contenido significativo
            try:
                wait = WebDriverWait(driver, MAX_WAIT_TIME)
                wait.until(
                    lambda d: img.get_attribute("srcset") and len(img.get_attribute("srcset")) > 10
                )
            except TimeoutException:
                pass # Si hay timeout, simplemente usamos lo que haya cargado

            # --- LÓGICA DE EXTRACCIÓN ROBUSTA (USANDO LA FUNCIÓN HELPER) ---
            img_url = None
            srcset = img.get_attribute("srcset")
            
            # PRIORITY 1: Usar la función helper para la máxima resolución de srcset
            if srcset:
                img_url = get_highest_res_url(srcset) 

            # PRIORITY 2: Fallback a 'src'
            if not img_url or "placeholder" in str(img_url).lower():
                img_url = img.get_attribute("src")

            # Final Validation
            if not img_url or "placeholder" in str(img_url).lower() or not str(img_url).startswith("http"):
                print(f"  Saltando ítem {product_idx}: No se encontró una URL de imagen válida para el slide {slide_index + 1}.")
                # Importante: Aumentar el índice solo si el producto es válido, pero aquí lo aumentamos
                # para que el siguiente producto tenga el product_idx correcto.
                product_idx += 1 
                continue 


            # --- PRICE EXTRACTION (USANDO data-qa con Regex) ---
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
                product_idx += 1 
                
        except NoSuchElementException:
            # Si un contenedor no tiene la estructura básica (imagen en el slide seleccionado), se salta
            product_idx += 1 
            continue 

except Exception as e:
    print(f"Error general durante la recolección de datos: {e}")

print(f"Total products scraped and ready for download: {len(products_data)}")

# --- DESCARGA DE IMÁGENES Y PREPARACIÓN CSV ---
image_folder = "images_ysl"
os.makedirs(image_folder, exist_ok=True)

count = 0
# Es importante iterar sobre products_data, que ya contiene solo los productos válidos
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
    writer.writerows([(name, p, url) for name, p, url in products_data]) 

print(f"✅ CSV saved at {csv_path}")
print(f"✅ Scraping completed: {count} images saved + {len(products_data)} prices scraped.")

driver.quit()
