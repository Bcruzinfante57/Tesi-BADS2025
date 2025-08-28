# Step 1: Define the function to scrape a specific brand
def scrape_brand_eyewear(brand_name):
    # Dicc. para mapear marcas a URLs. Puedes agregar más marcas aquí.
    brand_urls = {
        "miu miu": "https://www.miumiu.com/it/it/accessories/eyewear/c/10267EU",
        "ysl": "https://www.ysl.com/it-it/search?q=occhiali%20da%20sole",
        "dior": "https://www.dior.com/it_it/fashion/products/search?query=occhiali%20da%20sole",
        "gucci": "https://www.gucci.com/it/it/ca/women/womens-accessories/womens-eyew",
        "Louis Vuitton": "https://it.louisvuitton.com/ita-it/homepage?search=occhiali%20da%20sole" 
        # Add more brands here...
    }

    # Verifica si la marca existe en el diccionario
    if brand_name.lower() not in brand_urls:
        print(f"Error: La marca '{brand_name}' no está en la lista de URLs disponibles.")
        return

    # Asigna la URL y la carpeta de imágenes dinámicamente
    start_url = brand_urls[brand_name.lower()]
    image_folder = f"imagenes_{brand_name.lower()}"
    os.makedirs(image_folder, exist_ok=True)
    
    # Rest of your original script
    # Step 2: Initialize WebDriver
    service = Service("/Users/benja/tools/chromedriver")  # Adjust path as needed
    driver = webdriver.Chrome(service=service)
    wait = WebDriverWait(driver, 20)

    # Step 3: Open the brand page
    driver.get(start_url)
    time.sleep(3)
    
    # Step 4: Accept cookie notice
    try:
        # El XPATH para cookies puede variar entre páginas
        cookie_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//*[text()='Accetta tutto']"))
        )
        cookie_btn.click()
        print("Cookies accepted")
    except:
        print("Cookie notice not found or different.")
    
    # Step 5: Scroll and collect product URLs
    # This might need to be adjusted for each brand
    products = driver.find_elements(By.CSS_SELECTOR, "a.product-grid__item-link")
    product_links = list({p.get_attribute("href") for p in products if p.get_attribute("href")})

    # Step 6: Download one image per product URL
    for url in product_links:
        driver.get(url)
        time.sleep(2)
    
        try:
            code = driver.find_element(By.CSS_SELECTOR, ".product-code span").text
            img_url = driver.find_element(By.CSS_SELECTOR, "img.gallery__image").get_attribute("src")
    
            img_path = os.path.join(image_folder, f"{code}.jpg")
            with open(img_path, "wb") as f:
                f.write(requests.get(img_url).content)
    
            print(f"Image saved: {code}")
    
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
    
    # Step 7: Finish process
    driver.quit()
    print(f"Image scraping completed for {brand_name.upper()}.")



if __name__ == '__main__':
    # OPTION 1: Only Miu Miu
    #print("Scraping Miu Miu...")
    #scrape_brand_eyewear("MiuMiu")

    # OPTION 2: Only YSL 
    # print("\nScraping YSL...")
    # scrape_brand_eyewear("YSL")

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