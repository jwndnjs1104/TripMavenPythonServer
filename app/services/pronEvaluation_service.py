import urllib3
import json
import base64

#openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Pronunciation"  # 영어
openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor" # 한국어
accessKey = "79e5a1f4-d732-4baf-8d08-ed3fadbf88a3"

class PronEvaluationService:
    def evaluate(self, encoded_voice, text):
        try:
            print('발음 측정 함수 들어왓당')
            languageCode = "korean"
            script = text

            requestJson = {
                "argument": {
                    "language_code": languageCode,
                    "script": script,
                    "audio": encoded_voice
                }
            }

            http = urllib3.PoolManager()
            print('발음 측정 요청 전')
            response = http.request(
                "POST",
                url=openApiURL,
                headers={"Content-Type": "application/json; charset=UTF-8", "Authorization": accessKey},
                body=json.dumps(requestJson)
            )

            print("[responseCode] " + str(response.status))
            print("[responBody]\n"+str(response.data, "utf-8"))

            return response.data

        except :
            return False