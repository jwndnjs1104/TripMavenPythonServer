import urllib3
import json
import base64

#openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Pronunciation"  # 영어
openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor" # 한국어
#https://aiopen.etri.re.kr/serviceList 에 있는 발음평가 API를 쓰려 했으나 요청이 잘 들어가지 않아 쓰지 않음
accessKey = "위 사이트에서 발급 받은 access 키 넣어야 함"

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