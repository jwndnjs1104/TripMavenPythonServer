from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import FileResponse
from app.services.voice_check_service import Sound_Check_Class
from app.services.nlp_check_service import text_analysis
from app.services.whisperSTT_service import WhisperVoiceEvaluation
import os, re, wave
import numpy as np
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from app.services.face_service import plot_line_graph

router = APIRouter()
whisperModel = WhisperVoiceEvaluation()

#파일 저장 경로
SAVE_DIRECTORY = r'D:\.LGR\Proj\PythonServer\uploaded_files'
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

# moviepy를 사용하여 비디오에서 오디오 추출하는 함수
def extract_audio(video_file_path: str, output_audio_file_path: str):
    try:
        print('비디오 추출함수 진입')
        video = VideoFileClip(video_file_path)
        print('비디오 생성')
        video.audio.write_audiofile(output_audio_file_path)
        video.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 추출 실패: {e}")


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

        # 파일 경로 지정
        file_location = os.path.join(SAVE_DIRECTORY, voice.filename)
        print('디버그3:', file_location)
        # 파일을 서버에 저장
        with open(file_location, "wb") as buffer:
            buffer.write(await voice.read())

        # 영상파일 받았을때 추출할 음성파일 경로 만들기
        extracted_file_location = os.path.join(SAVE_DIRECTORY, 'extracted_audio.webm')
        print('디버그4:', extracted_file_location)

        # webm 파일을 wav 파일로 변경하기 위해서 wav 파일 경로 만들기
        converted_file_location = os.path.join(SAVE_DIRECTORY, 'converted_audio.wav')
        print('디버그5:', converted_file_location)
        convert_webm_to_wav(file_location, converted_file_location)  # webm 오디오에서 wav로 변환

        #영상 테스트시 추출한 오디오 파일(webm)
        # if isVoiceTest == '0':
        #     print('모의 테스트시')
        #     extract_audio(file_location, extracted_file_location) #영상에서 오디오 추출(webm 오디오 파일일 것같음)
        #     convert_webm_to_wav(extracted_file_location, converted_file_location) #webm 오디오에서 wav로 변환
        # else:
        #     print('발음 테스트시')
        #     if voice.filename.endswith('.webm'):
        #         print('webm파일인 경우')
        #         convert_webm_to_wav(file_location, converted_file_location)
        #     else:
        #         converted_file_location=file_location
        # print('최종 파일 경로:',converted_file_location)

        # 텍스트 파일 분석(stt된 내용에 대한 평가, .? 비율, 워드클라우드용, 불필요한 추임새 )
        # 발음 테스트의 경우 아래 결과는 무의미하다 판단해서 제외했음
        response["text_analysis"] = ""
        if isVoiceTest == '0':
            response["text_analysis"] = text_analysis(text)
        print('디버그6')

        # 목소리 톤 분석(발음 테스트시 이것만 반환됨)(wav)
        response["voice_tone"] = voice_run(converted_file_location, gender)
        print('디버그7')

        # 말하기 속도 및 발음 정확도 측정(webm)
        # if isVoiceTest == '0':
        #     result = whisperModel.evaluate(extracted_file_location, text)
        # else:
        #     result = whisperModel.evaluate(file_location, text)
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

    voice_check_point = sound.sound_model(interpolated, sex)
    print('디버그10')
    mean = int(round(np.nanmean(interpolated)))
    std = int(round(np.nanstd(interpolated)))
    x = pitch_analysis.xs()
    x = np.round(x, 5)
    x = x.tolist()
    y = np.nan_to_num(interpolated)
    y = y.tolist()
    base64Image = plot_line_graph(x,y,'Voice_Tone','Time','Hz')

    # 분석결과를 json 파일에 저장
    voice_json = {
        "voice": base64Image,
        "voice_mean": mean,

        "voice_std": std,
        "voice_check": voice_check_point
    }
    sound.del_file(filepath)

    return voice_json