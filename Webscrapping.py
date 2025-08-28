from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests

def scrape_brand_eyewear(brand_name, max_products=50):
    # Dicc. para mapear marcas a URLs. Puedes agregar más marcas aquí.
    brand_urls = {
        "miu miu": "https://www.miumiu.com/it/it/accessories/eyewear/c/10267EU",
        "ysl": "https://www.ysl.com/it-it/search?q=occhiali%20da%20sole",
        "dior": "https://www.dior.com/it_it/fashion/products/search?query=occhiali%20da%20sole",
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

    # Aceptar cookies con selector genérico
    try:
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'Accett') or contains(text(), 'Accept')]"))
        )
        cookie_btn.click()
        print("Cookies accepted")
    except:
        print("Cookie notice not found or different.")

    # Selectores genéricos para paginación y enlaces de producto
    pagination_selectors = [
        "a.next-page",
        "a[rel='next']",
        ".pagination-next a"
    ]
    product_link_selectors = "a[href*='product'], a[class*='product'], a.item"

    is_pagination = False
    next_page_link = None
    for selector in pagination_selectors:
        try:
            next_page_link = driver.find_element(By.CSS_SELECTOR, selector)
            if next_page_link:
                is_pagination = True
                break
        except:
            continue

    total_product_links = set()

    if is_pagination:
        print("Modo de navegación: Paginación tradicional detectada.")
        while True:
            products = driver.find_elements(By.CSS_SELECTOR, product_link_selectors)
            current_page_links = {p.get_attribute("href") for p in products if p.get_attribute("href")}
            total_product_links.update(current_page_links)

            try:
                next_page_link = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                print(f"Moving to next page: {next_page_link.get_attribute('href')}")
                next_page_link.click()
                time.sleep(3)
            except:
                print("No more pages found. Exiting pagination loop.")
                break
    else:
        print("Modo de navegación: Scroll infinito asumido.")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print("No new content loaded after scrolling. Stopping.")
                break
            last_height = new_height

        products = driver.find_elements(By.CSS_SELECTOR, product_link_selectors)
        total_product_links = {p.get_attribute("href") for p in products if p.get_attribute("href")}

    print(f"Total product links found: {len(total_product_links)}")

    count = 0
    for url in total_product_links:
        if count >= max_products:
            print(f"Goal of {max_products} products reached. Stopping download.")
            break

        driver.get(url)
        time.sleep(2)

        try:
            # Selectores genéricos para código de producto e imagen
            code_element = driver.find_element(By.CSS_SELECTOR, ".product-code, .sku, .item-number, [itemprop='sku']")
            code = code_element.text.strip() if code_element.text else os.path.basename(url)

            img_element = driver.find_element(By.CSS_SELECTOR, "img[alt*='product'], img[class*='product'], img.main-image, img")
            img_url = img_element.get_attribute("src")

            img_path = os.path.join(image_folder, f"{code}.jpg")
            with open(img_path, "wb") as f:
                f.write(requests.get(img_url).content)

            print(f"Image saved: {code}")
            count += 1

        except Exception as e:
            print(f"Error downloading image from {url}: {e}")

    driver.quit()
    print(f"Image scraping completed for {count} products of {brand_name.upper()}.")

if __name__ == '__main__':
    # OPTION 1: Only Miu Miu
    #print("Scraping Miu Miu...")
    #scrape_brand_eyewear("MiuMiu")

    OPTION 2: Only YSL 
    print("\nScraping YSL...")
    scrape_brand_eyewear("YSL")

    # OPTION 3: Only Dior
    # print("\nScraping Dior...")
    # scrape_brand_eyewear("Dior")
    # 
    # OPTION 4: Only Gucci
    # print("\nScraping Gucci...")
    # scrape_brand_eyewear("Gucci")
     
    # OPTION 5: Only Louis Vuitton
    #print("\nScraping Louis Vuitton...")
    #scrape_brand_eyewear("Louis Vuitton")      