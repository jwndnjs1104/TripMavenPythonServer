import parselmouth
import numpy as np
import os
import matplotlib.pyplot as plt  # 그래프 시각화를 위한 라이브러리

# 음성 분석 클래스 입니다.
class Sound_Check_Class:

    # 초기값으로 wav파일과 성별을 입력받습니다.
    def __init__(self, filepath):
        file_location = os.path.abspath(os.path.join(filepath))
        print('생성자 안:',file_location)
        self.filepath = file_location

    # wav 데이터를 받고 waveform(파형)데이터로 변환하는 함수입니다.
    def load_wave(self):
        waveform = parselmouth.Sound(self.filepath)
        print('웨이브폼1')
        waveform = parselmouth.praat.call(waveform, "Convert to mono")
        print('웨이브폼2')
        return waveform

    # pitch의 max값과 min 값을 설정하는 함수입니다.
    def extract_pitch(self, waveform):
        pitch_analysis = waveform.to_pitch(pitch_ceiling=400, pitch_floor=50)
        return pitch_analysis

    # extract_pitch함수의 pitch를 받고 interpolated 리턴하는 함수입니다.
    def set_pitch_analysis(self, pitch_analysis):
        # pitch 데이터 전처리
        interpolated = pitch_analysis.interpolate().selected_array['frequency']
        interpolated[interpolated == 0] = np.nan
        # 전처리된 데이터를 분석 모델에 적용
        return interpolated

    # np.array 타입의 interpolated 와 sex 받아 음성 분석을 하는 함수입니다.
    def sound_model(self, interpolated, sex):
        std = int(round(np.nanstd(interpolated)))

        # 평균(mean)과, 표준편차(std) 분석하는 코드입니다.
        if int(sex) == 0:  # 남자
            if std < 20:
                return '너무 단조롭습니다(Bad)'

            elif std >= 20 and std < 30:
                return '적정합니다(Good)'

            elif std >= 30 and std <= 55:
                return '알아듣기 쉬워 이해하기 좋습니다(Perfect)'

            elif std > 55:
                return '너무 산만하거나 긴장중이네요(Bad)'

        elif int(sex) == 1:  # 여자
            if std < 45:
                return '너무 단조롭습니다(Bad)'

            elif std >= 45 and std < 55:
                return '적정합니다(Good)'

            elif std >= 55 and std <= 70:
                return '알아듣기 쉬워 이해하기 좋습니다(Perfect)'

            elif std > 70:
                return '너무 산만하거나 긴장중이네요(Bad)'

    # 파일삭제 함수입니다.
    def del_file(self, path):
        try:
            os.remove(path)
        except:
            pass

    # 메인 실행 함수 입니다.
