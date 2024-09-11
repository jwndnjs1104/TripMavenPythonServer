from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.voice_check_service import Sound_Check_Class
import io
import numpy as np

router = APIRouter()
@router.post("/")
async def voice_analysis(voice: UploadFile):
    try:
        # 음성 파일 읽기
        file_bytes = await voice.read()
        file_like_object = io.BytesIO(file_bytes)

        #메인 실행 함수
        #'D:/JJW/Utility/test.wav'
        response = voice_run(file_like_object, 0)

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

    return voice_json