from fastapi import APIRouter, HTTPException, Form
from app.services.object_service import ObjectDetectionService

#APIRouter 객체를 사용한 라우팅
router = APIRouter()
object_service = ObjectDetectionService()

@router.post("/")
async def detect(base64Encoded: str = Form(...)): #클라이언트로부터 Form 형식을 받아야 한다는 의미임(필수 필드)
    detects = object_service.object_detect(base64Encoded)
    if not detects:
        raise HTTPException(status_code=404, detail="Detection failed")
    return {"result": ''.join(detects)}
