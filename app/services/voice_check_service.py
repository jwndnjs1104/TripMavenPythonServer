import parselmouth
import numpy as np
import os


# 음성 분석 클래스 입니다.
class Sound_Check_Class:

    # 초기값으로 wav파일과 성별을 입력받습니다.
    def __init__(self, file_like_object):
        self.file_like_object = file_like_object

    # wav 데이터를 받고 waveform(파형)데이터로 변환하는 함수입니다.
    def load_wave(self):
        # 파일 객체를 직접 parselmouth.Sound에 전달
        waveform = parselmouth.Sound(self.file_like_object)
        waveform = parselmouth.praat.call(waveform, "Convert to mono")
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
    def sound_model(self, interpolated, pitch_analysis, sex):
        std = int(round(np.nanstd(interpolated)))

        # 평균(mean)과, 표준편차(std) 분석하는 코드입니다.
        if int(sex) == 0:
            if std < 20:
                return 1

            elif std >= 20 and std < 30:
                return 2

            elif std >= 30 and std <= 55:
                return 3

            elif std > 55:
                return 4

        elif int(sex) == 1:
            if std < 45:
                return 1

            elif std >= 45 and std < 55:
                return 2

            elif std >= 55 and std <= 70:
                return 3
            elif std > 70:
                return 4

    # 파일삭제 함수입니다.
    def del_file(self, path):
        try:
            os.remove(path)
        except:
            pass

    # 메인 실행 함수 입니다.
    def voice_run(self, sex):
        waveform = self.load_wave()
        pitch_analysis = self.extract_pitch(waveform)
        interpolated = self.set_pitch_analysis(pitch_analysis)

        voice_check_point = self.sound_model(interpolated, pitch_analysis, sex)

        mean = int(round(np.nanmean(interpolated)))
        std = int(round(np.nanstd(interpolated)))
        x = pitch_analysis.xs()
        x = np.round(x, 5)
        x = x.tolist()
        y = np.nan_to_num(interpolated)
        y = y.tolist()

        # 분석결과를 json 파일에 저장
        voice_json = {
            "voice": {
                "x": x,
                "y": y
            },
            "voice_mean": mean,

            "voice_std": std,
            "voice_check": voice_check_point
        }

        return voice_json