import undetected_chromedriver as uc
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

##This one depends on the chromedriver path in your PC

driver = uc.Chrome()

url = "https://www.ysl.com/it-it/search?q=occhiali%20da%20sole"

driver.get(url)
driver.maximize_window()

time.sleep(3)

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

    # **5. Encontrar y hacer clic en el botón 'Vedi tutto'**
print("Buscando y haciendo clic en el botón 'Vedi tutto'...")

total_products_count = None
try:
        vedi_tutto_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//*[contains(normalize-space(), 'Vedi tutto')]"))
        )
        vedi_tutto_button.click()
        print("Botón 'Vedi tutto' clicado. Cargando todos los productos.")
        time.sleep(5) # Espera a que los productos se carguen
        total_products_text = vedi_tutto_button.text
        match = re.search(r'\((\d+)\)', total_products_text)
        if match:
            total_products_count = int(match.group(1))
        
        driver.execute_script("arguments[0].click();", vedi_tutto_button)
        print("Botón 'Vedi tutto' clicado. Cargando todos los productos.")
        time.sleep(5)

except (TimeoutException, NoSuchElementException, ElementNotInteractableException):
        print("Botón 'Vedi tutto' no encontrado o no es clicable. Asumiendo que todos los productos ya están cargados.")

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

# --- RECOLECCIÓN DE IMÁGENES ---
print("Collecting image URLs...")
products_to_download = {}

try:
    # 1. Encontrar todos los contenedores de productos
    grid_items = driver.find_elements(By.CSS_SELECTOR, "li[id^='grid-item-']")
    print(f"Número de contenedores de productos encontrados: {len(grid_items)}")
    
    # 2. Iterar sobre cada contenedor para encontrar la primera imagen
    for item in grid_items:
        try:
            img = item.find_element(By.TAG_NAME, "img")
            
            # Obtener la URL de alta resolución del srcset
            srcset = img.get_attribute("srcset")
            if srcset:
                urls = srcset.split(', ')
                img_url = urls[-1].split(' ')[0]
            else:
                img_url = img.get_attribute("src")

            # Validar y extraer el código de producto de la URL
            code_match = re.search(r'dam.kering.com/(\S+)\?', img_url)
            if img_url and "kering.com" in img_url and code_match:
                product_code = code_match.group(1).split('/')[-1].replace('.A.jpg', '')
                products_to_download[product_code] = img_url
        except NoSuchElementException:
            continue # Si un contenedor no tiene imagen, se salta

except Exception as e:
    print(f"Error localizando productos: {e}")

print(f"Total product image URLs found: {len(products_to_download)}")

# --- DESCARGA ---
image_folder = "imagenes_ysl"
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

driver.quit()
print(f"Image scraping completed for {count} products of YSL.")

