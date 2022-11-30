import time

import requests
import re
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
        print(response.status_code)  # 查看请求状态，返回200说明正常
        path = './images/'+name+"/"+name+'.'+str(i)+'.jpg'  # 文件储存地址
        with open(path, 'wb') as f:  # 把图片数据写入本地，wb表示二进制储存
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)


def get_urls(url, page_number, alt_name):
    options = webdriver.ChromeOptions()
    options.add_argument(
        'user-agent="Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19",--headless'
    )
    img_tags = []
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    for i in range(page_number):
        ignored_exceptions = (NoSuchElementException, StaleElementReferenceException,)
        # wait until find the element
        next_page = WebDriverWait(driver, 5, ignored_exceptions=ignored_exceptions)\
            .until(expected_conditions.presence_of_element_located((By.LINK_TEXT, '下一页')))
        next_page.click()
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        img_tags += (soup.find_all("img", {"alt": alt_name}))
    print(str(len(img_tags))+' images')
    url_list = []
    start = 'http'  # 定义筛选条件
    end = '.jpg'  # 定义图片后缀
    for i in img_tags:
        reg = re.findall(r"http(.*).jpg", str(i), re.S)
        if len(reg) != 0:
            img_url = start + reg[0] + end
            url_list.append(img_url)
    return url_list


# URL = 'http://www.iplant.cn/info/Nelumbo?t=p'
# urls = get_urls(URL, 20, "莲属")
# save_image('Nelumbo', urls)

# URL = 'http://www.iplant.cn/info/Prunus%20mume?t=p'
# urls = get_urls(URL, 100, "梅")
# save_image('PrunusMume', urls)

URL = 'http://www.iplant.cn/info/Rosa?t=p'
urls = get_urls(URL, 2, "蔷薇属")
save_image('Rosa', urls)

