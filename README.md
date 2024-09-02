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

실행법
터미널에서 아래 코드 실행
uvicorn app.main:app --port 8282 --reload
