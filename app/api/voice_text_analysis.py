from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from app.services.voice_check_service import Sound_Check_Class
from app.services.nlp_check_service import text_analysis
import io, os, re
import numpy as np

router = APIRouter()

# 파일 저장 경로
SAVE_DIRECTORY = "uploaded_files/"

# 디렉토리가 없으면 생성
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

@router.post("/")
async def combined_analysis(voice: UploadFile = File(None), gender: int = Form(...), text: str = Form(...)):
    try:
        response = {}

        # 음성 파일 분석
        if voice:
            # 파일 경로 지정
            file_location = os.path.join(SAVE_DIRECTORY, voice.filename)

            # 파일을 서버에 저장
            with open(file_location, "wb") as buffer:
                buffer.write(await voice.read())

            # 메인 실행 함수
            response["voice_analysis"] = voice_run(file_location, gender)

        # 텍스트 파일 분석

        # if text:
        #     p = re.compile(r'[ .,]') #쉼표, 온점 제거, 띄어쓰기 제거
        #     text = re.sub(p, '', text)
        #
        #     response["text_analysis"] = text_analysis(text)

        if not voice and not text:
            raise HTTPException(status_code=400, detail="No file provided for analysis")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Analysis failed: {str(e)}")

    return response


#voice 분석 실행 함수
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