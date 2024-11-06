from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image, ImageDraw
import os
import pandas as pd

def make_screenshots(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    original_size = driver.get_window_size()
    required_width = driver.execute_script('return document.body.parentNode.scrollWidth')
    required_height = driver.execute_script('return document.body.parentNode.scrollHeight')
    driver.set_window_size(required_width, required_height)
    
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    path = os.path.join(os.path.dirname(__file__), 'screenshots')
    if not os.path.exists(path):
        os.makedirs(path)
        
    full_screenshot_path = f'{path}/full_screen.png'
    driver.save_screenshot(full_screenshot_path)
    
    element_coords = []

    # Liens
    a_tags = driver.find_elements(By.TAG_NAME, 'a')
    json_coords = {}

    for i, a_tag in enumerate(a_tags):
        location = a_tag.location
        size = a_tag.size
        element_coords.append((location['x'], location['y'], size['width'], size['height']))

        if a_tag.get_attribute("href"):
            #json_coords[f'{location["x"]}_{location["y"]}_{size["width"]}_{size["height"]}'] = a_tag.get_attribute("href")
            json_coords[a_tag.get_attribute("href")] = f'x: {location["x"]}, y: {location["y"]}, w: {size["width"]}, h: {size["height"]}'
        
        screenshot_path = f'{path}/a_tag_{i}.png'

        try:
            a_tag.screenshot(screenshot_path)
        except Exception as e:
            print(e)

    # Save
    df = pd.DataFrame(json_coords.items(), columns=['coords', 'link'])
    df.to_json(f'{path}/links.json', orient='records')
    
    driver.set_window_size(original_size['width'], original_size['height'])
    driver.quit()

    # Rectangles
    image = Image.open(full_screenshot_path)
    draw = ImageDraw.Draw(image)
    for coords in element_coords:
        x, y, width, height = coords
        end_x, end_y = x + width, y + height
        draw.rectangle([x, y, end_x, end_y], outline="red", width=2)        
    
    modified_screenshot_path = f'{path}/full_screen2.png'
    image.save(modified_screenshot_path)

if __name__ == '__main__':
    #url = 'https://all.accor.com/ssr/app/hotelf1/hotels/multihotels/index.fr.shtml?rids=2432,2536,B3B4,2476,2498,2387,2420,B3A3,2259,2461,B3B7,2487,B3A5,3176,5010,3488,3072,2287,2539,3506,2274,2278,2230,2339,2354,2403,2500&compositions=1&stayplus=false&snu=false&hideWDR=false&accessibleRooms=false&hideHotelDetails=false'
    url = 'https://stackoverflow.com/questions/41721734/take-screenshot-of-full-page-with-selenium-python-with-chromedriver'
    make_screenshots(url)
