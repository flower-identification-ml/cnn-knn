import time

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
        file_name = name+'.'+str(i)+'.jpg'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        else:
            path = file_path + file_name  # File storage path
            with open(path, 'wb') as f:  # Image data is written locally, wb means binary storage
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

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        img_tags += (soup.find_all("img", {"alt": alt_name}))
        time.sleep(5)
        next_page.click()
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


# # first class
# URL = "http://www.iplant.cn/info/Nelumbo%20nucifera?t=p"
# urls = get_urls(URL, 200, "莲")
# save_image('Lotus', urls)

# # second class
# URL = "http://www.iplant.cn/info/郁金香?t=p"
# urls = get_urls(URL, 200, "郁金香")
# save_image('Tulips', urls)
#
# third class
# URL = 'http://www.iplant.cn/info/向日葵?t=p'
# urls = get_urls(URL, 200, "向日葵")
# save_image('Sunflower', urls)
#
# Fourth
# URL = 'http://www.iplant.cn/info/Cymbidium%20goeringii?t=p'
# urls = get_urls(URL, 200, "春兰")
# save_image('Orchid', urls)
#
# # Five
# URL = 'http://www.iplant.cn/info/Prunus%20mume?t=p'
# urls = get_urls(URL, 200, "梅")
# save_image('PrunusMume', urls)
#
# # Six
# URL = 'http://www.iplant.cn/info/Rosa?t=p'
# urls = get_urls(URL, 2, "蔷薇属")
# save_image('Rosa', urls)

URL = 'http://www.iplant.cn/info/薰衣草?t=p'
urls = get_urls(URL, 34, "薰衣草")
save_image('Lavender', urls)


