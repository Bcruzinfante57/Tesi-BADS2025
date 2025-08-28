from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests

# Step 1: Initialize WebDriver and create image folder
service = Service("/Users/benja/tools/chromedriver")  # Adjust path as needed
driver = webdriver.Chrome(service=service)
wait = WebDriverWait(driver, 20)
image_folder = "imagenes_miumiu"
os.makedirs(image_folder, exist_ok=True)

# Step 2: Open Miu Miu eyewear page
driver.get("https://www.miumiu.com/it/it/accessories/eyewear/c/10267EU")
time.sleep(3)

# Step 3: Accept cookie notice
try:
    cookie_btn = WebDriverWait(driver, 5).until(
        EC.element_to_be_clickable((By.XPATH, "//*[text()='Accetta tutto']"))
    )
    cookie_btn.click()
    print("Cookies accepted")
except:
    print("Cookie notice not found")

# Step 4: Scroll to load all product thumbnails
for _ in range(10):
    driver.execute_script("window.scrollBy(0, 500);")
    time.sleep(1)

# Step 5: Collect product URLs
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
print("Image scraping completed.")
