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

def scrape_brand_eyewear(brand_name, max_products=150):
    brand_urls = {
        "miu miu": "https://www.miumiu.com/it/it/accessories/eyewear/c/10267EU",
        "ysl": "https://www.ysl.com/it-it/search?q=occhiali%20da%20sole",
        "gucci": "https://www.gucci.com/it/it/st/newsearchpage?searchString=occhiali%20da%20sole&sortBy=relev&search-cat=header-search",
        "louis vuitton": "https://it.louisvuitton.com/ita-it/homepage?search=occhiali%20da%20sole"
    }

    brand_key = brand_name.lower()
    if brand_key not in brand_urls:
        print(f"Error: La marca '{brand_name}' no está en la lista de URLs disponibles.")
        return

    start_url = brand_urls[brand_key]
    image_folder = f"imagenes_{brand_key}"
    os.makedirs(image_folder, exist_ok=True)

    service = Service("/Users/benja/tools/chromedriver")
    driver = webdriver.Chrome(service=service)
    driver.get(start_url)
    time.sleep(3)

    try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[normalize-space(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'accetta tutti i cookie']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
    except (TimeoutException, NoSuchElementException):
        print("Cookie notice not found or different.")

    # Lógica de desplazamiento dinámico para cargar productos
    print("Scrolling down to load products until the page stops expanding...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_counter = 0
    max_scrolls = 10 
    
    while scroll_counter < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            print("No new content loaded after scrolling. Reached the end of the page.")
            break
        
        last_height = new_height
        scroll_counter += 1
    
    time.sleep(10) # Pausa final para asegurar que todas las imágenes se carguen

    print("Collecting image URLs...")
    
    images = driver.find_elements(By.CSS_SELECTOR, "img")
    products_to_download = {}
    
    for img in images:
        src = img.get_attribute("src")
        code_match = re.search(r'([A-Za-z0-9]{14})', src)
        if src and "kering.com" in src and code_match:
            product_code = code_match.group(1)
            if product_code not in products_to_download:
                 products_to_download[product_code] = src

    print(f"Total product image URLs found: {len(products_to_download)}")

    count = 0
    for code, img_url in products_to_download.items():
        if count >= max_products:
            print(f"Goal of {max_products} products reached. Stopping download.")
            break

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
    print(f"Image scraping completed for {count} products of {brand_name.upper()}.")

if __name__ == '__main__':
    # OPTION 1: Only Miu Miu
    #print("Scraping Miu Miu...")
    #scrape_brand_eyewear("MiuMiu")

    #OPTION 2: Only YSL 
    print("\nScraping YSL...")
    scrape_brand_eyewear("ysl")

    # OPTION 4: Only Gucci
    #print("\nScraping Gucci...")
    #scrape_brand_eyewear("Gucci")
     
    # OPTION 5: Only Louis Vuitton
    #print("\nScraping Louis Vuitton...")
    #scrape_brand_eyewear("Louis Vuitton")      