from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.pronEvaluation_service import PronEvaluationService
import base64, io, os
from pydub import AudioSegment
from fastapi import Request
#import ffmpeg

router = APIRouter()
pron_evaluation_service = PronEvaluationService()

SAVE_DIRECTORY = 'uploaded_files'
#디렉토리가 없으면 생성
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

#webm파일을 pcm파일로 변환
def convert_webm_to_pcm(input_file, output_file):
    try:
        # WebM 파일을 읽어서 PCM 형식으로 변환
        audio = AudioSegment.from_file(input_file, format="webm")
        # PCM 형식으로 파일 저장
        audio.export(output_file, format="raw")
        print('pcm형식으로 저장 완료')
    except Exception as e:
        print('오류:',e)
#webm파일을 WAV파일로 변환하는 함수
def convert_webm_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file, format="webm")
    audio.export(output_file, format="wav")
    
# WAV 파일을 base64로 인코딩하는 함수
def encode_wav_to_base64(file_path):
    with open(file_path, 'rb') as wav_file:
        # 파일을 바이너리 모드로 읽어서 base64로 인코딩
        encoded_string = base64.b64encode(wav_file.read()).decode('utf-8')
    return encoded_string

@router.post("/")
async def pron_evaluation(request: Request, voice: UploadFile = File(...), text: str = Form(...)):
    try:
        print(f"Request Content-Type: {request.headers.get('content-type')}")
        print('파이썬 서버 들어온 파일:', voice)
        # 파일 경로 지정
        file_location = os.path.join(SAVE_DIRECTORY, voice.filename)
        print('file_location:',file_location)

        # 파일을 서버에 저장
        with open(file_location, "wb") as buffer:
            buffer.write(await voice.read())
        print('파일 이름',voice.filename)

        # webm 파일을 wav 파일로 변경
        if voice.filename.endswith('.webm'):
            print('파일이 webm 일때')
            converted_file_location = os.path.join(SAVE_DIRECTORY, 'converted_audio.pcm')
            convert_webm_to_pcm(file_location, converted_file_location)
            file_location = converted_file_location
        print('최종 file_location:', file_location)

        #base64 인코딩하기
        encoded_voice = encode_wav_to_base64(file_location)
        response = pron_evaluation_service.evaluate(encoded_voice, text)
    except:
        raise HTTPException(status_code=404, detail="Voice Evaluation failed")
    #finally:
        # 파일 삭제
        # if file_location and os.path.exists(file_location):
        #     os.remove(file_location)

    return response