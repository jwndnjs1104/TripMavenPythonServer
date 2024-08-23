from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.ocr_service import OCRService
#from app.services.verifyLicense_service import VerificationGuideLicense
import base64

router = APIRouter()
ocr_service = OCRService()
#verify_guide_license = VerificationGuideLicense()

@router.post("/license")
async def varify_license(image: UploadFile):
    '''
    파일로 받아서 베이스64인코딩해서 ocr로 넘기기
    '''
    try:
        # 이미지 파일 읽기
        image_data = await image.read()
        # Base64로 인코딩
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        print(encoded_image)
        #detects = ocr_service.ocr_detect(encoded_image)

        '''
        ocr 돌린것 중에서 이름이랑 자격증 관리번호 가져와서 밑에 verify에 넣어주기
        '''

        #isVerify = verify_guide_license.verify(name, regnum1, regnum2)

    except:
        raise HTTPException(status_code=404, detail="Detection failed")

    return {"result":"test"}

