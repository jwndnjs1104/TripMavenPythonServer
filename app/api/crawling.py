from fastapi import APIRouter
from bs4 import BeautifulSoup
import requests

#APIRouter 객체를 사용한 라우팅, main에서 최종적으로 엔트포인트 설정하고 라우팅하면 된다.
router = APIRouter()

#https://news.naver.com/main/list.naver?oid=052&listType=title

@router.post("/")
def crawling():
    url = 'https://news.naver.com/main/list.naver?oid=052&listType=title'
    res = requests.get(url)
    source = res.text
    soup = BeautifulSoup(source, 'html.parser')
    selectors = soup.select(selector='#main_content > div.list_body.newsflash_body > ul:nth-child(2) > li > a')
    headlines = []
    for headline in selectors:
        headlines.append(headline.get_text())
    return headlines

