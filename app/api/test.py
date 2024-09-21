from fastapi import APIRouter, HTTPException, Form, UploadFile, File
import os, re
from app.services.whisperSTT_service import WhisperVoiceEvaluation
router = APIRouter()
whisperModel = WhisperVoiceEvaluation()

#임시 파일 저장 경로
SAVE_DIRECTORY = r'D:\JJW\Workspace\pythonServer\pythonServer\uploaded_files'
#디렉토리가 없으면 생성
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

@router.post("/")
async def test(voice: UploadFile = File(...), text: str = Form(...)):
    try:
        #발음 정확도, 간접적으로
        #말하기 속도
        print('테스트 들어옴')
        
        #파일 저장 위치(파일 형식은 webm도 가능해서 변환 안함)
        file_location = os.path.join(SAVE_DIRECTORY, voice.filename)

        #파일을 서버에 저장
        with open(file_location, "wb") as buffer:
            buffer.write(await voice.read())

        #클래스 생성 후 평가 함수 호출, 결과 받기
        result = whisperModel.evaluate(file_location, text)

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=404, detail=f"Test failed: {str(e)}")
    # finally:
        # 파일 삭제
        # if file_location and os.path.exists(file_location):
        #     os.remove(file_location)
    return result