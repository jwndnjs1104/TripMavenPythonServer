# -*- coding:utf-8 -*-
import urllib3
import json
import base64

#openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Pronunciation"  # 영어
openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor" # 한국어

accessKey = "79e5a1f4-d732-4baf-8d08-ed3fadbf88a3"
audioFilePath = "AUDIO_FILE_PATH"
languageCode = "LANGUAGE_CODE"
script = "PRONUNCIATION_SCRIPT"

file = open(audioFilePath, "rb")
audioContents = base64.b64encode(file.read()).decode("utf8")
file.close()

requestJson = {
    "argument": {
        "language_code": languageCode,
        "script": script,
        "audio": audioContents
    }
}

http = urllib3.PoolManager()
response = http.request(
    "POST",
    openApiURL,
    headers={"Content-Type": "application/json; charset=UTF-8", "Authorization": accessKey},
    body=json.dumps(requestJson)
)

print("[responseCode] " + str(response.status))
print("[responBody]")
print(str(response.data, "utf-8"))