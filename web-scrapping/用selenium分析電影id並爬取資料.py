import  pandas as pd
import numpy as np

# 操作 browser 的 驅動程式
from selenium import webdriver

# 負責開啟和關閉 Chrome 的套件
from selenium.webdriver.chrome.service import Service

# 自動下載 Chrome Driver 的套件
from webdriver_manager.chrome import ChromeDriverManager

# 例外處理的工具
from selenium.common.exceptions import TimeoutException

# 面對動態網頁，等待、了解某個元素的狀態，通常與 exptected_conditions 和 By 搭配
from selenium.webdriver.support.ui import WebDriverWait

# 搭配 WebDriverWait 使用，對元素狀態的一種期待條件，若條件發生，則等待結束，往下一行執行
from selenium.webdriver.support import expected_conditions as EC

# 期待元素出現要透過什麼方式指定，經常與 EC、WebDriverWait 一起使用
from selenium.webdriver.common.by import By

# 強制停止/強制等待 (程式執行期間休息一下)
from time import sleep

# 隨機取得 User-Agent
from fake_useragent import UserAgent
ua = UserAgent()

# 匯入os
import os

# 啟動瀏覽器工具的選項
my_options = webdriver.ChromeOptions()
my_options.add_argument("--start-maximized")         # 最大化視窗
my_options.add_argument("--incognito")               # 開啟無痕模式
my_options.add_argument("--disable-popup-blocking")  # 禁用彈出攔截
my_options.add_argument("--disable-notifications")   # 取消通知
my_options.add_argument(f'--user-agent={ua.random}') # (Optional)加入 User-Agent

# 建立下載路徑/資料夾，不存在就新增 (os.getcwd() 會取得當前的程式工作目錄)
folderPath = os.path.join(os.getcwd(), 'files')
if not os.path.exists(folderPath):
    os.makedirs(folderPath)
    
 # 1 匯入tmdb的movie list
df_combined3 = pd.read_json('combined_list.json')

# 2 輸入mojo 入口網站
url = "https://www.boxofficemojo.com/?ref_=bo_nb_se_mojologo"
# 使用Chrome的WebDriver
driver = webdriver.Chrome(
    options = my_options,
    service = Service(ChromeDriverManager().install())
)

# 3 造訪網頁
driver.get(url)

# 4 執行程式
# movie_link_list 表示直接搜尋得到的網址
# 流水碼分兩種, 當直接搜尋電影時, 該電影有一組流水號
# 點進去搜尋domestic票房時, 該電影又有一組流水號
# 故movie_id_list 表示如果搜尋的到domestic票房(排除國外電影), 就把點進domestic以後的網址的流水碼寫進movie_link_list
movie_link_list = []
movie_id_list=[]
count=0
for i in df_combined3['title'][:10]:
    inputElement = driver.find_element(By.CSS_SELECTOR, 'input#mojo-search-text-input')
    inputElement.clear()
    inputElement.send_keys(i)
    sleep(1)
    inputElement.submit()
    try:
        WebDriverWait(driver,5).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div.a-fixed-left-grid-col.a-col-right")
            )
        )
        # 如果有找到這部電影連結, 則點進去
        hrefElement = driver.find_element(By.CSS_SELECTOR, 'div.a-fixed-left-grid-col.a-col-right > a')
        movie_link_list.append(hrefElement.get_attribute('href'))
        hrefElement.click()
        
        domesticElement = driver.find_element(By.CSS_SELECTOR, 'td:nth-child(1) > a')
        if domesticElement.text != 'Domestic':
            movie_id_list.append("NA")
        else:
            movie_id_list.append(domesticElement.get_attribute('href').split(sep='/')[4])
    

    except TimeoutException:
        # 如果搜尋不到這部電影, 則加上NA
        movie_link_list.append("NA")
        movie_id_list.append("NA")
        continue
        


driver.quit()
