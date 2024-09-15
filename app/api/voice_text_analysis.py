from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from app.services.voice_check_service import Sound_Check_Class
from app.services.nlp_check_service import text_analysis
import os, re, base64, io
import numpy as np
#import ffmpeg

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
        response = {}

        p = re.compile(r'[.,]')  # 쉼표와 온점만 제거
        text = re.sub(p, '', text)

        # webm에서 오디오 추출
        #voice = await extract_audio_from_webm(voice)

        #음성 분석(목소리톤)
        if voice:
            #파일 경로 지정
            file_location = os.path.join(SAVE_DIRECTORY, voice.filename)
            #파일을 서버에 저장
            with open(file_location, "wb") as buffer:
                buffer.write(await voice.read())

            #목소리 톤 분석
            response["voice_analysis"] = voice_run(file_location, gender)

        # 텍스트 파일 분석(stt된 내용에 대한 평가, .? 비율, 워드클라우드용, 불필요한 추임새 )
        if text:
            response["text_analysis"] = text_analysis(text)


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


# async def extract_audio_from_webm(file: UploadFile):
#     # webm 파일을 바이트 스트림으로 읽음
#     input_stream = io.BytesIO(await file.read())
#
#     # 출력될 오디오 파일을 위한 바이트 스트림 준비
#     output_stream = io.BytesIO()
#
#     # ffmpeg 명령어 실행: webm에서 오디오만 추출하여 output_stream에 저장
#     try:
#         (
#             ffmpeg
#             .input('pipe:0', format='webm')  # 'pipe:0'은 input_stream을 뜻함
#             .output('pipe:1', format='wav')  # 'pipe:1'은 output_stream을 뜻함, WAV로 추출
#             .run(input=input_stream, output=output_stream)
#         )
#     except ffmpeg.Error as e:
#         print(f"FFmpeg error: {e.stderr}")
#         raise
#
#     # 바이트 스트림의 현재 위치를 처음으로 되돌림
#     output_stream.seek(0)
#
#     return output_stream.getvalue()  # 오디오 데이터를 바이트로 반환