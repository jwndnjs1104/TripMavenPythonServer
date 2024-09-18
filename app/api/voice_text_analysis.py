from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from app.services.voice_check_service import Sound_Check_Class
from app.services.nlp_check_service import text_analysis
import os, re, base64, io
import numpy as np

router = APIRouter()

#파일 저장 경로
SAVE_DIRECTORY = "uploaded_files/"
#디렉토리가 없으면 생성
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

@router.post("/")
async def combined_analysis(voice: UploadFile = File(...), gender: int = Form(...), text: str = Form(...)):
    try:
        #응답데이터 저장 객체
        print('디버그1')
        response = {}
        p = re.compile(r'[.,]')  # 쉼표와 온점만 제거
        text = re.sub(p, '', text)

        #음성 분석(목소리톤)
        #파일 경로 지정
        print('디버그2')
        file_location = os.path.join(SAVE_DIRECTORY, voice.filename)
        print('디버그3')
        #파일을 서버에 저장
        with open(file_location, "wb") as buffer:
            buffer.write(await voice.read())
        print('디버그4')
        #목소리 톤 분석
        response["voice_analysis"] = voice_run(file_location, gender)
        print('디버그5')

        # 텍스트 파일 분석(stt된 내용에 대한 평가, .? 비율, 워드클라우드용, 불필요한 추임새 )
        response["text_analysis"] = text_analysis(text)
        print('디버그6')
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Analysis failed: {str(e)}")
    return response

#voice 분석 실행 함수
def voice_run(filepath, sex):
    sound = Sound_Check_Class(filepath)
    print(sound.filepath)
    print('디버그6')
    waveform = sound.load_wave()
    print('디버그7')
    pitch_analysis = sound.extract_pitch(waveform)
    print('디버그8')
    interpolated = sound.set_pitch_analysis(pitch_analysis)
    print('디버그9')

    voice_check_point = sound.sound_model(interpolated, pitch_analysis, sex)
    print('디버그10')
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