from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from app.services.pronEvaluation_service import PronEvaluationService
import base64, io, ffmpeg
import asyncio

router = APIRouter()
pron_evaluation_service = PronEvaluationService()

@router.post("/")
async def pron_evaluation(voice: UploadFile, text: str = Form(...)):
    try:
        #webm에서 오디오 추출
        #audio_data = await extract_audio_from_webm(voice)
        #Base64로 인코딩
        #encoded_voice = base64.b64encode(audio_data).decode('utf-8')

        #음성 파일 읽기
        voice_data = await voice.read()
        #Base64로 인코딩
        encoded_voice = base64.b64encode(voice_data).decode('utf-8')
        response = pron_evaluation_service.evaluate(encoded_voice, text)
    except:
        raise HTTPException(status_code=404, detail="Voice Evaluation failed")

    return response


async def extract_audio_from_webm(file: UploadFile):
    # webm 파일을 바이트 스트림으로 읽음
    input_stream = io.BytesIO(await file.read())

    # 출력될 오디오 파일을 위한 바이트 스트림 준비
    output_stream = io.BytesIO()

    # ffmpeg 명령어 실행: webm에서 오디오만 추출하여 output_stream에 저장
    try:
        (
            ffmpeg
            .input('pipe:0', format='webm')  # 'pipe:0'은 input_stream을 뜻함
            .output('pipe:1', format='wav')  # 'pipe:1'은 output_stream을 뜻함, WAV로 추출
            .run(input=input_stream, output=output_stream)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr}")
        raise

    # 바이트 스트림의 현재 위치를 처음으로 되돌림
    output_stream.seek(0)

    return output_stream.getvalue()  # 오디오 데이터를 바이트로 반환