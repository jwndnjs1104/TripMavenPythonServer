from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.pronEvaluation_service import PronEvaluationService
from app.services.verifyLicense_service import VerifyLicenseService
import base64

router = APIRouter()
pron_evaluation_service = PronEvaluationService()
@router.post("/")
async def pron_evaluation(voice: UploadFile, text: str = Form(...)):
    try:
        # 음성 파일 읽기
        voice_data = await voice.read()
        # Base64로 인코딩
        encoded_voice = base64.b64encode(voice_data).decode('utf-8')
        response = pron_evaluation_service.evaluate(encoded_voice, text)
    except:
        raise HTTPException(status_code=404, detail="Voice Evaluation failed")

    return response