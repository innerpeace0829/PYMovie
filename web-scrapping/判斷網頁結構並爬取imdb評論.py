from bs4 import BeautifulSoup as bs
import requests as req
import json, os, pprint, time, re
import pandas as pd
import numpy as np
folderPath = 'IMDB'
if not os.path.exists(folderPath):
    os.makedirs(folderPath)
row = ["Title", "Author", "Date", "Up Vote", "Total Vote", "Rating", "Review"]

def get_IMDB_review(imdb_id):
    global df

    base_url = "https://www.imdb.com/"
    url=f"https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ql_3"
    key = ""
    res = req.get(url)
    res.encoding = 'utf-8'
    soup = bs(res.text, "lxml")
    
    title = [item.select_one(".title").text for item in soup.select(".lister-item-content")]
    author = [item.select_one(".display-name-link").text for item in soup.select(".lister-item-content")]
    date = [item.select_one(".review-date").text for item in soup.select(".lister-item-content")]
    upvote = [item.select_one('.actions.text-muted').text.split(sep=" ")[20] for item in soup.select(".lister-item-content")]
    totalvote = [item.select_one('.actions.text-muted').text.split(sep=" ")[23] for item in soup.select(".lister-item-content")]
    rating=[(item.select_one("span.rating-other-user-rating > span").text if len(item.select("span.rating-other-user-rating > span"))==2 else "") for item in soup.select(".lister-item-content") ]
    review = [item.select_one('.text').text for item in soup.select(".lister-item-content")]

    # 確認LOADMORE資料內容
    load_more = soup.select_one(".load-more-data")
    flag = True
    # 找到load_more tag, 才需要往下抓
    # 第二波的第一次抓loadmore
    if load_more.text != '\n':
        ajaxurl = load_more['data-ajaxurl']
        base_url = base_url + ajaxurl + "?ref_=undefined&paginationKey="
        key = load_more['data-key']
    else :
        while flag:
            url = base_url + key
            res = req.get(url)
            res.encoding = 'utf-8'
            soup = bs(res.text, "lxml")
            title2 = [item.select_one(".title").text for item in soup.select(".lister-item-content")]
            title += title2
            author2 = [item.select_one(".display-name-link").text for item in soup.select(".lister-item-content")]
            author += author2
            date2 = [item.select_one(".review-date").text for item in soup.select(".lister-item-content")]
            date += date2
            upvote2 = [item.select_one('.actions.text-muted').text.split(sep=" ")[20] for item in soup.select(".lister-item-content")]
            upvote += upvote2
            totalvote2 = [item.select_one('.actions.text-muted').text.split(sep=" ")[23] for item in soup.select(".lister-item-content")]
            totalvote += totalvote2
            rating2 =[(item.select_one("span.rating-other-user-rating > span").text if len(item.select("span.rating-other-user-rating > span"))==2 else "") for item in soup.select(".lister-item-content") ]
            rating += rating2
            review2 = [item.select_one('.text').text for item in soup.select(".lister-item-content")]
            review += review2

            load_more = soup.select_one(".load-more-data")
            if load_more:
                key = load_more['data-key']
            else:
                flag = False
    sumlist = np.array((title, author, date, upvote, totalvote, rating, review))
    length = sumlist.T.shape[0] 
    IMDBID = np.full((length,1),imdb_id)
    convertnp = np.concatenate((IMDBID,sumlist.T),axis = 1)
    df = pd.DataFrame(convertnp,columns = ['IMDBId','title', 'author', 'date', 'upvote', 'totalvote', 'rating', 'review'])
    df.to_csv(f"{folderPath}/{imdb_id}.csv",index = False)
    print(f"{imdb_id} success")
    
    
# 針對你手上擁有的movie id資料進行爬蟲即可完成
# [get_IMDB_review(i) for i in imdb_id]
