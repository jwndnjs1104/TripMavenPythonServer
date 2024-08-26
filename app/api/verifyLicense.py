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
async def varify_license(detects_result: str = Form(...)):
    try:
        # 이미지 파일 읽기
        #image_data = await image.read()
        # Base64로 인코딩
        #encoded_image = base64.b64encode(image_data).decode('utf-8')

        #detects = ocr_service.ocr_detect(encoded_image)
        #print(detects)

        list_ = detects_result[0].split('\n')
        print(list_)

        name = ''
        number = ''
        for element in list_:
            if '관리번호' in element:
                number = element.replace('관리번호','').replace(':','').strip()
            elif '성명' in element:
                name = element.replace('성명','').replace(':','').strip()

        regnum1 = ''
        regnum2 = ''
        if '-08-' in number:
            regnum1, regnum2 = number.split('-08-')

        print(name, regnum1, regnum2)

        isVerify = verify_license_service.verify(name, regnum1, regnum2)

    except:
        raise HTTPException(status_code=404, detail="License Verification failed")

    return {"result":isVerify}

