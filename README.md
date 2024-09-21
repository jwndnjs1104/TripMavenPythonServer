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

pip install praat-parselmouth\

pip install opencv-python

pip install mediapipe

pip install keras

pip install konlpyf

pip install ffmpeg-python



Whisper 설치
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

cmd에서 ffmpeg 설치  
sudo apt-get install ffmpeg  
 또는 
sudo apt install ffmpeg

파이토치 설치 해야함 
pip3 install torch torchvision torchaudio

pip install tiktoken
pip install jamo


실행법
터미널에서 아래 코드 실행
uvicorn app.main:app --port 8282 --reload
