from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from app.services.voice_check_service import Sound_Check_Class
from app.services.nlp_check_service import text_analysis
import os, re
import numpy as np
import wave
from pydub import AudioSegment
from app.services.whisperSTT_service import WhisperVoiceEvaluation

router = APIRouter()
whisperModel = WhisperVoiceEvaluation()

#파일 저장 경로
SAVE_DIRECTORY = r'D:\JJW\Workspace\pythonServer\pythonServer\uploaded_files'
#디렉토리가 없으면 생성
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

def check_wav_file(file_path):
    try:
        with wave.open(file_path, 'rb') as f:
            print(f.getparams())  # 파일 정보 출력
    except wave.Error as e:
        print(f"Invalid WAV file: {e}")

def convert_webm_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="webm")
    audio.export(output_file, format="wav")

@router.post("/")
async def combined_analysis(voice: UploadFile = File(...), gender: int = Form(...), text: str = Form(...), isVoiceTest: str = Form(...)):
    try:
        #응답데이터 저장 객체
        response = {}
        p = re.compile(r'[.,]')  # 쉼표와 온점만 제거
        text = re.sub(p, '', text)

        #음성 분석(목소리톤)
        #파일 경로 지정
        file_location = os.path.join(SAVE_DIRECTORY, voice.filename)
        print('디버그3:',file_location)
        #파일을 서버에 저장
        with open(file_location, "wb") as buffer:
            buffer.write(await voice.read())
        #webm 파일을 wav 파일로 변경
        converted_file_location = os.path.join(SAVE_DIRECTORY, 'converted_audio.wav')
        print('디버그4:',converted_file_location)
        convert_webm_to_wav(file_location,converted_file_location)

        check_wav_file(converted_file_location)
        #x, sr = librosa.load(file_location)
        #print(f'디버깅, x:{x}, sr:{sr}')

        # 텍스트 파일 분석(stt된 내용에 대한 평가, .? 비율, 워드클라우드용, 불필요한 추임새 )
        # 발음 테스트의 경우 아래 결과는 무의미하다 판단해서 제외했음
        response["text_analysis"] = ""
        if isVoiceTest == '0':
            response["text_analysis"] = text_analysis(text)
        print('디버그5')

        # 목소리 톤 분석(발음 테스트시 이것만 반환됨)
        response["voice_tone"] = voice_run(converted_file_location, gender)
        print('디버그6')

        # 말하기 속도 및 발음 정확도 측정
        result = whisperModel.evaluate(file_location, text)
        if result:  # None이 아닌지 확인
            response["speed_result"] = result.get('speed_result', {})
            response["pronunciation_precision"] = result.get('pronunciation_precision', {})
        else:
            # 에러 처리 또는 기본값 설정
            response["speed_result"] = {}
            response["pronunciation_precision"] = {}
            print("평가 결과를 얻지 못했습니다.")

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=404, detail=f"Analysis failed: {str(e)}")
    return response

#voice 분석 실행 함수
def voice_run(filepath, sex):
    sound = Sound_Check_Class(filepath)
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