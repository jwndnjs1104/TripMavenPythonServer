from fastapi import APIRouter, HTTPException, Form
from app.services.ocr_service import OCRService

#APIRouter 객체를 사용한 라우팅, main에서 최종적으로 엔트포인트 설정하고 라우팅하면 된다.
router = APIRouter()
ocr_service = OCRService()

#여기서 엔드포인트 설정하는게 아니라 그냥 기본으로 "/"설정해 주면 됨
@router.post("/")
async def detect(base64Encoded: str = Form(...), ocrValue: str = Form(...)):
    '''
    POST 요청으로 받은 데이터를 처리하는 FastAPI 엔드포인트
    base64Encoded: base64로 인코딩된 이미지 데이터
    ocrValue: 'ocr' 또는 다른 값으로 OCR 또는 객체 탐지 선택
    '''
    if ocrValue == 'ocr':
        detects = ocr_service.ocr_detect(base64Encoded)
    else:
        detects = ocr_service.object_detect(base64Encoded)

    if not detects:
        raise HTTPException(status_code=404, detail="Detection failed")

    return {"result": ''.join(detects)}

