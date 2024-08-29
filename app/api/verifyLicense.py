from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.ocr_service import OCRService
from app.services.verifyLicense_service import VerifyLicenseService
import base64

router = APIRouter()
ocr_service = OCRService()
verify_license_service = VerifyLicenseService()
#base64Encoded: str = Form(...)
#image: UploadFile

@router.post("/")
async def varify_license(subject: str = Form(...), name: str = Form(...), number: str = Form(...)):
    try:
        # 이미지 파일 읽기
        #image_data = await image.read()
        # Base64로 인코딩
        #encoded_image = base64.b64encode(image_data).decode('utf-8')

        #detects = ocr_service.ocr_detect(encoded_image)
        #print(detects)
        regnum1 = ''
        regnum2 = ''
        # if '-08-' in number:
        #     regnum1, regnum2 = number.split('-08-')
        if 'HRD' in number:
            regnum1, regnum2 = number.replace('HRD-','').strip().split('-')

        print(name, regnum1, regnum2)
        verifiedData = verify_license_service.verify2(name, regnum1, regnum2)
        #팝업창의 결과를 다시 한번 OCR돌려서 받은 값과 위에서 받은 값이 3개가 일치하면
        #인증 성공이다.
        print(verifiedData)
        if verifiedData.isSuccess == False:
            return False
        else:
            return True
    except:
        raise HTTPException(status_code=404, detail="License Verification failed")

