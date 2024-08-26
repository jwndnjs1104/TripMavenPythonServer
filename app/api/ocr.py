from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.ocr_service import OCRService
from fastapi.responses import RedirectResponse
import base64

#APIRouter 객체를 사용한 라우팅, main에서 최종적으로 엔트포인트 설정하고 라우팅하면 된다.
router = APIRouter()
ocr_service = OCRService()
#image: UploadFile = File(...)
#base64Encoded: str = Form(...)

@router.post("/")
async def detect(image: UploadFile, ocrValue: str = Form(...)):
    '''
    POST 요청으로 받은 데이터를 처리하는 FastAPI 엔드포인트
    base64Encoded: base64로 인코딩된 이미지 데이터
    ocrValue: 'ocr' 또는 다른 값으로 OCR 또는 객체 탐지 선택
    '''
    # 이미지 파일 읽기
    image_data = await image.read()
    # Base64로 인코딩
    base64Encoded = base64.b64encode(image_data).decode('utf-8')

    if ocrValue == 'ocr':
        detects = ocr_service.ocr_detect(base64Encoded)
    else:
        detects = ocr_service.object_detect(base64Encoded)

    if not detects:
        raise HTTPException(status_code=404, detail="Detection failed")

    list_ = detects[0].split('\n')
    print(list_)
    name = 'default'
    number = 'default'
    subject = 'default'
    for index, element in enumerate(list_):
        if '문서확인번호' in element:
            number = element.replace('문서확인번호', '').replace(':', '').strip()
        elif element == '명':
            print(element, index)
            name = list_[index+1].strip()
        elif '자 격 명' in element:
            subject = element.replace('자 격 명','').strip()

    return {'name':name, 'number':number, 'subject':subject}

