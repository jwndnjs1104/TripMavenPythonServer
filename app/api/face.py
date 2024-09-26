# face.py
import os
import tempfile
import subprocess
from fastapi import APIRouter, File, UploadFile
from typing import Dict
from app.services.face_service import process_video

router = APIRouter()

# 비디오 파일을 mp4로 변환하는 함수
def convert_to_mp4(input_file_path: str, output_file_path: str) -> bool:
    try:
        # ffmpeg를 사용해 비디오 파일을 mp4로 변환
        command = ['ffmpeg', '-i', input_file_path, '-c:v', 'libx264', output_file_path]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {str(e)}")
        return False

@router.post("/")
async def upload_video(file: UploadFile = File(...)) -> Dict:
    print('file:: ', file)

    # 업로드된 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # mp4로 변환할 파일 경로 설정
    mp4_temp_file_path = temp_file_path.replace(os.path.splitext(temp_file_path)[1], '.mp4')

    # 파일이 mp4가 아닌 경우 변환
    if not file.filename.endswith('.mp4'):
        conversion_success = convert_to_mp4(temp_file_path, mp4_temp_file_path)
        if not conversion_success:
            # 파일 사용 후 삭제
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return {"error": "비디오 변환 실패"}

        # 원본 파일 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    else:
        mp4_temp_file_path = temp_file_path

    # 비디오 처리 함수 호출
    with open(mp4_temp_file_path, "rb") as video_file:  # 파일을 열고 처리
        result = process_video(video_file.read())  # 비디오 처리

    # 변환된 mp4 파일 삭제
    if os.path.exists(mp4_temp_file_path):
        os.remove(mp4_temp_file_path)

    return result
