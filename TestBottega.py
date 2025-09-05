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

# Desplazamiento en pasos para simular un humano
driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.4);")
print("Desplazando 60%...")
time.sleep(5)

print("Iniciando desplazamiento para asegurar que todos los productos se carguen.")
time.sleep(2) # Espera inicial

# Bucle principal para el desplazamiento
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

    # Obtener el número actual de productos en la página usando selectores combinados
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

# --- RECOLECCIÓN DE IMÁGENES (solo la foto del anteojo en la slide ACTIVA) ---
print("Collecting image URLs...")
products_to_download = {}
seen = set()

# 1) Cards de producto
cards = driver.find_elements(By.CSS_SELECTOR, "article.c-product[data-pid]")
print(f"Número de cards de productos: {len(cards)}")

for card in cards:
    try:
        pid = card.get_attribute("data-pid")
        if not pid or pid in seen:
            continue

        # 2) Dentro de cada card: slide activa del carrusel
        active_img_selector = (
            "ul.c-product__carousel "
            "li.c-product__carousel--slide.swiper-slide-active "
            "img.c-product__image"
        )

        # Pequeño retry por si el carrusel tarda en pintar
        img = None
        for _ in range(6):
            try:
                img = card.find_element(By.CSS_SELECTOR, active_img_selector)
                break
            except NoSuchElementException:
                time.sleep(0.5)

        # Fallback: si no encuentra la activa, toma la primera slide
        if img is None:
            try:
                img = card.find_element(
                    By.CSS_SELECTOR,
                    "ul.c-product__carousel li.c-product__carousel--slide img.c-product__image"
                )
            except NoSuchElementException:
                continue

        # 3) Tomar la URL de mejor calidad del srcset
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
            best = max(parts, key=width_of)          # la de mayor ancho (ej. Large 1367w)
            url = best.split()[0]
        else:
            url = img.get_attribute("src")

        if url:
            products_to_download[pid] = url
            seen.add(pid)

    except Exception:
        continue

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