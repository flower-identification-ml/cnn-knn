import re
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait


def get_urls(base_url):
    url_list = []
    html = get_full_html(base_url)
    soup = BeautifulSoup(html, 'html.parser')
    # items = soup.find_all('img')
    items = get_full_html(base_url, 10)
    s = 'http'
    e = '.jpg'
    for i in items:
        reg = re.findall(r"http(.*).jpg", str(i), re.S)
        if len(reg) != 0:
            img_url = s + reg[0] + e
            url_list.append(img_url)
    print(len(url_list))
    return url_list

def get_full_html(url):
    options = webdriver.ChromeOptions()
    options.add_argument(
        'user-agent="Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19",--headless'
    )
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    html = driver.page_source
    return html

#
# URL = 'http://www.iplant.cn/info/Nelumbo?t=p'
# print(get_full_html(URL, 10, '莲属'))

x = 'dataname://'
print()
print(x.split('//'))

