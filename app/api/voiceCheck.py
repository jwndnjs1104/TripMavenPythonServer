from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.voice_check_service import Sound_Check_Class
import io

router = APIRouter()

@router.post("/")
async def voice_analysis(file: UploadFile = File(...)):
    try:
        # 음성 파일 읽기
        file_bytes = await file.read()
        # Base64로 인코딩
        file_like_object = io.BytesIO(file_bytes)

        #메인 실행 함수
        sound_check = Sound_Check_Class(file_like_object)
        response = sound_check.voice_run('0')

    except:
        raise HTTPException(status_code=404, detail="Voice Evaluation failed")
    return response