ai api를 위한 fastapi 서버

pip install fastapi 

pip install "uvicorn[standard]" 

pip install SQLAlchemy cx_Oracle

pip install selenium

pip install --upgrade paho-mqtt

pip install python-multipart

pip install requests

pip install --upgrade google-cloud-vision

pip install openai

pip install bs4

pip install Image

pip install praat-parselmouth

pip install opencv-python

pip install mediapipe

pip install keras

pip install konlpy

pip install ffmpeg-python

pip install kiwipiepy

pip install moviepy imageio-ffmpeg

Whisper 설치
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

ffmpeg 설치
사이트에 가서 압축파일 다운받고 c드라이브에 풀기
https://github.com/BtbN/FFmpeg-Builds/releases
환경변수(시스템 변수)에서 PATH에 bin폴더경로 추가하기(이럼 끗)
cmd에서 잘 설치됐는지 확인하기: ffmpeg -version

파이토치 설치 해야함 
pip3 install torch torchvision torchaudio

pip install tiktoken
pip install jamo


실행법
터미널에서 아래 코드 실행
uvicorn app.main:app --port 8282 --reload
