from fastapi import APIRouter, Form
from app.services.ocr_service import OCRService
import sys
import requests

router = APIRouter()
id = 'yq1nf7y6jv'
secret_key = '5A8yk14sOGVhEHo6hmg4Kkw6ihf67c5SdgkayXuc'

@router.post("/")
async def speech_to_text(filepath: str = Form(...)):
    client_id = id
    client_secret = secret_key
    lang = "Kor"  # 언어 코드 ( Kor, Jpn, Eng, Chn )
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang
    
    #음성 데이터 넣기
    data = open(filepath, 'rb')

    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(url, data=data, headers=headers)
    rescode = response.status_code

    if (rescode == 200):
        return response.text
    else:
        return "Error : " + response.text
