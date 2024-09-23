import whisper
import re
import jiwer
from whisper.tokenizer import get_tokenizer
from jamo import h2j, j2hcj
from konlpy.tag import Okt

class WhisperVoiceEvaluation:
    def __init__(self):
        # Whisper는 로컬에서 작동, api로 호출 안함
        Whisper = whisper.load_model("base", device="cpu")
        self.model = Whisper
        self.okt = Okt()

    def normalize_korean(self, text):
        # 한글 자모 분리
        jamo = j2hcj(h2j(text))
        # 띄어쓰기 제거
        jamo = re.sub(r'\s+', '', jamo)
        return jamo

    def calculate_pronunciation_accuracy(self, avg_logprob_score, cer):
        # Log probability를 0-1 범위로 정규화
        # 일반적으로 Whisper의 log probability는 -1 ~ 0 범위
        if(avg_logprob_score<-1):
            avg_logprob_score=-1
        normalized_logprob = 1 + avg_logprob_score

        # CER을 0-1 범위로 변환 (0이 가장 좋고, 1이 가장 나쁨)
        error_rate = min(1, cer)  # CER이 1을 넘을 경우 1로 제한
        accuracy_from_cer = 1 - error_rate

        # Log probability와 CER-based accuracy의 가중 평균
        # 여기서는 동일한 가중치를 사용, 필요에 따라 조정 가능
        weighted_accuracy = (normalized_logprob + accuracy_from_cer) / 2

        # 결과를 0-100 범위의 점수로 변환
        final_score = weighted_accuracy * 100

        return final_score

    def get_speaking_speed(self, STTresult):
        # 전체 텍스트
        full_text = STTresult['text']

        # 텍스트를 띄어쓰기로 나누어 단어 리스트 생성
        word_list = full_text.split()
        word_count = len(word_list)

        # 세그먼트 확인하여 실제 말한 시간 계산
        segments = STTresult['segments']
        total_spoken_time = 0

        # no_speech_prob가 0.5 이하인 세그먼트만 계산 (임의 기준)
        for segment in segments:
            if segment['no_speech_prob'] < 0.5:
                spoken_time = segment['end'] - segment['start']
                total_spoken_time += spoken_time

        # 말하기 속도 계산 (단어 수 / 말한 시간)
        speaking_speed_per_second = word_count / total_spoken_time if total_spoken_time > 0 else 0
        speaking_speed_per_min = speaking_speed_per_second*60


        # 한국어 자음 및 모음 리스트
        filtered_text = re.sub(r'[^가-힣]', '', full_text)
        phoneme_count = len(filtered_text)

        # 초당 음소량 계산
        phonemes_per_second = phoneme_count / total_spoken_time if total_spoken_time > 0 else 0
        phonemes_per_min = phonemes_per_second*60

        print(f"말한 시간: {total_spoken_time:.2f} 초")
        print(f"전체 텍스트: {full_text}")
        print(f"단어 수: {word_count}")
        print(f"초당 말하기 속도: {speaking_speed_per_second:.2f} 단어/초")
        print(f"분당 말하기 속도(WPM): {speaking_speed_per_min:.2f} 단어/분")
        print(f"총 음소 수: {phoneme_count}")
        print(f"초당 음소량: {phonemes_per_second:.2f} 음소/초")
        print(f"분당 음소량: {phonemes_per_min:.2f} 음소/분")

        result = {
            'total_spoken_time':total_spoken_time,
            'full_text': full_text,
            'word_count': word_count,
            'speaking_speed_per_second': round(speaking_speed_per_second),
            'speaking_speed_per_min': round(speaking_speed_per_min),
            'phoneme_count': phoneme_count,
            'phonemes_per_second': round(phonemes_per_second),
            'phonemes_per_min': round(phonemes_per_min)
        }
        return result

    def get_pronunciation_precision(self, STTresult, reference_text, no_speech_prob_threshold=0.5):

        #reference_text는 정답 텍스트
        # 전체 텍스트와 세그먼트
        reference_text = re.sub(r'[^가-힣]', '', reference_text)
        transcribed_text = STTresult['text']
        transcribed_text = re.sub(r'[^가-힣]', '', transcribed_text)
        segments = STTresult['segments']

        # 형태소 분석
        reference_morphs = ' '.join(self.okt.morphs(reference_text))
        transcribed_morphs = ' '.join(self.okt.morphs(transcribed_text))

        # 정규화
        normalized_reference = self.normalize_korean(reference_morphs)
        normalized_transcribed = self.normalize_korean(transcribed_morphs)

        # 말한 시간 동안의 평균 log probability 계산
        total_avg_logprob = 0
        total_spoken_time = 0

        for segment in segments:
            if segment['no_speech_prob'] < no_speech_prob_threshold:
                segment_time = segment['end'] - segment['start']
                total_avg_logprob += segment['avg_logprob'] * segment_time
                total_spoken_time += segment_time

        # 전체 평균 log probability
        avg_logprob_score = total_avg_logprob / total_spoken_time if total_spoken_time > 0 else -1.0

        # CER 계산
        cer = jiwer.wer(normalized_reference, normalized_transcribed)

        pronunciation_accuracy = self.calculate_pronunciation_accuracy(avg_logprob_score, cer)

        # 발음 정확도 점수 계산 (log probability와 CER 기반)
        pronunciation_precision = {
            'pronunciation_accuracy':pronunciation_accuracy,
            'avg_logprob_score':avg_logprob_score,
            'cer':cer,
            'reference_text':reference_text,
            'transcribed_text': transcribed_text
        }

        return pronunciation_precision

    def evaluate(self, file_location, reference_text):
        try:
            print('음성 평가 함수 들어옴')
            STTresult =self.model.transcribe(file_location, fp16=False, language="ko")
            print(STTresult)

            #테스트
            # tokenizer = get_tokenizer(self.model.is_multilingual)
            # tokens = result['segments'][0]['tokens'][:3]
            # print('토큰:',tokens)
            # decoded_text = tokenizer.decode(tokens)
            # print('토큰 디코드:', decoded_text)

            # 말하기 속도
            speed_result = self.get_speaking_speed(STTresult)

            # 발음 정확도(간접적으로)
            pronunciation_precision = self.get_pronunciation_precision(STTresult, reference_text=reference_text, no_speech_prob_threshold=0.5)

            result = {
                'speed_result':speed_result,
                'pronunciation_precision':pronunciation_precision
            }
            return result

        except Exception as e:
            print('에러났당:',e)
            return False