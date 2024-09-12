from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.voice_check_service import Sound_Check_Class
import io, os
import numpy as np

router = APIRouter()

# 파일 저장 경로
SAVE_DIRECTORY = "uploaded_files/"

# 디렉토리가 없으면 생성
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

@router.post("/")
async def voice_analysis(voice: UploadFile, gender: int = Form(...)):
    try:
        # 파일 경로 지정
        file_location = os.path.join(SAVE_DIRECTORY, voice.filename)

        # 파일을 서버에 저장
        with open(file_location, "wb") as buffer:
            buffer.write(await voice.read())

        #메인 실행 함수
        response = voice_run(file_location, gender)

    except:
        raise HTTPException(status_code=404, detail="Voice Evaluation failed")
    return response

def voice_run(filepath, sex):
    sound = Sound_Check_Class(filepath)
    waveform = sound.load_wave()
    pitch_analysis = sound.extract_pitch(waveform)
    interpolated = sound.set_pitch_analysis(pitch_analysis)

    voice_check_point = sound.sound_model(interpolated, pitch_analysis, sex)

    mean = int(round(np.nanmean(interpolated)))
    std = int(round(np.nanstd(interpolated)))
    x = pitch_analysis.xs()
    x = np.round(x, 5)
    x = x.tolist()
    y = np.nan_to_num(interpolated)
    y = y.tolist()

    # 분석결과를 json 파일에 저장
    voice_json = {
        "voice": {
            "x": x,
            "y": y
        },
        "voice_mean": mean,

        "voice_std": std,
        "voice_check": voice_check_point
    }
    sound.del_file(filepath)

    return voice_json