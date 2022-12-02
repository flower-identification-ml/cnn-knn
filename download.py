import requests
import re
import os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


def save_image(name, url_list):
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400'
    }
    for i in range(len(url_list)):
        url = url_list[i]
        response = requests.get(url, headers=headers)
        print(response.status_code)  # Check the request status. If 200 is returned, the request status is normal

        file_path = './images/'+name+"/"
        file_name = name+'.'+str(i+1200)+'.jpg'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        else:
            path = file_path + file_name  # File storage path
            with open(path, 'wb') as f:  # Image data is written locally, wb means binary storage
                for chunk in response.iter_content(chunk_size=128):
                    f.write(chunk)


def get_urls(url, page_number):
    options = webdriver.ChromeOptions()
    options.add_argument(
        'user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400",--headless'
    )
    img_tags = []
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    for i in range(page_number):
        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
        # wait until find the element
        next_page = WebDriverWait(driver, 5, ignored_exceptions=ignored_exceptions)\
            .until(expected_conditions.presence_of_element_located((By.LINK_TEXT, '下一页')))
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        img_tag = (soup.find_all("img", {"class": 'lazy'}))
        for e in img_tag:
            img_tags.append(e['src'])
        next_page.click()

    url_list = []
    start = 'https://'  # Defining filter criteria
    end = '.jpg'
    for i in img_tags:
        reg = re.findall(r"//(.*).jpg", str(i), re.S)
        if len(reg) != 0:
            img_url = start + reg[0] + end
            print(img_url)
            url_list.append(img_url)
    print(len(url_list))
    return url_list


# Six
URL = 'https://sc.chinaz.com/tupian/?keyword=向日葵'
urls = get_urls(URL, 9)
save_image('Sunflower', urls)