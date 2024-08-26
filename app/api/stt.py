from fastapi import APIRouter, Form, UploadFile
import sys
import requests

router = APIRouter()
id = 'yq1nf7y6jv'
secret_key = '5A8yk14sOGVhEHo6hmg4Kkw6ihf67c5SdgkayXuc'

@router.post("/")
async def speech_to_text(voice: UploadFile):
    # 음성 파일 읽기
    # 음성 데이터 포맷은 mp3, aac, ac3, ogg, flac, wav
    voice_data = await voice.read()

    client_id = id
    client_secret = secret_key
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/octet-stream"
    }
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=Kor"
    response = requests.post(url, data=voice_data, headers=headers)
    rescode = response.status_code

    if (rescode == 200):
        return response.text
    else:
        return "Error : " + response.text
