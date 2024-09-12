import cv2
import numpy as np
import tempfile
import os
from typing import Dict
import mediapipe as mp  # 얼굴 인식
from scipy.spatial import distance  # 두 점 사이의 거리 계산을 위한 모듈
import tensorflow as tf  # 머신러닝 모델 실행
import matplotlib.pyplot as plt  # 그래프 시각화를 위한 라이브러리
import base64  # 이미지 데이터를 인코딩하기 위한 라이브러리
from io import BytesIO  # 이미지 데이터를 메모리 버퍼에 저장하기 위한 라이브러리

# 선 그래프를 그린 후 base64 형식으로 인코딩하여 반환하는 함수
def plot_line_graph(x, y, title, x_label, y_label):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # 그래프를 이미지로 저장
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # 이미지를 base64로 인코딩하여 반환
    graph_base64 = base64.b64encode(image_png).decode('utf-8')
    return graph_base64

## 눈 깜박임 횟수를 막대그래프로 시각화하는 함수
def plot_bar_graph(x, y, title, x_label, y_label):
    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(ticks=range(len(x)))  # x 축에 눈 깜박임 시점(프레임)을 표시

    # 그래프를 이미지로 저장
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # 이미지를 base64로 인코딩하여 반환
    graph_base64 = base64.b64encode(image_png).decode('utf-8')
    return graph_base64



# EyeCheck 클래스는 눈 깜박임과 얼굴 변화 분석을 위한 기능을 제공
class EyeCheck:
    def __init__(self):
        # 눈 및 얼굴 분석에 사용할 임계값 설정
        self.VISIBILITY_THRESHOLD = 0.5
        self.PRESENCE_THRESHOLD = 0.5

        # Mediapipe에서 눈 영역을 표시하는 랜드마크 인덱스 리스트
        self.Landmark_eye = [
            33, 7, 163, 144, 145, 153, 154, 155,
            133, 246, 161, 160, 159, 158, 157, 173,
            263, 249, 390, 373, 374, 380, 381, 382,
            362, 466, 388, 387, 386, 385, 384, 398
        ]

        # 모델을 파일에서 불러옴
        model_path = r'D:\.LGR\Proj\PythonServer\app\models\face\eye_model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # compile=False 옵션으로 모델을 로드하여, 불필요한 lr 관련 경고 해결
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # 모델을 로드한 후 다시 컴파일 (Adam 옵티마이저 사용)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.IMG_SIZE = (34, 26)  # 이미지 크기 설정
        self.mp_drawing = mp.solutions.drawing_utils  # Mediapipe의 그리기 유틸리티 초기화

    # 눈 부분 이미지를 자르는 함수
    def crop_eye(self, img, eye_points):
        # 눈 랜드마크의 최소, 최대 좌표 계산
        x1, y1 = np.amin(eye_points, axis=0)
        x2, y2 = np.amax(eye_points, axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # 눈 이미지의 가로, 세로 크기 계산
        w = (x2 - x1) * 1.2
        h = w * self.IMG_SIZE[1] / self.IMG_SIZE[0]

        # 눈 이미지의 마진 설정
        margin_x, margin_y = w / 2, h / 2

        # 눈 이미지의 좌상단 및 우하단 좌표 계산
        min_x, min_y = int(cx - margin_x), int(cy - margin_y)
        max_x, max_y = int(cx + margin_x), int(cy + margin_y)

        # 자른 이미지의 좌표를 정수로 변환
        eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)
        # 이미지를 자른 눈 부분을 반환
        eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]
        return eye_img, eye_rect

    # 랜드마크 데이터를 numpy 배열로 변환하는 함수
    def to_ndarray(self, landmark_dict):
        return np.array([x for i, x in landmark_dict.items() if i in self.Landmark_eye])

    # 왼쪽 눈 이미지를 모델에 전달하여 예측하는 함수
    def eye_preL(self, eye_img_l):
        eye_img_l = cv2.resize(eye_img_l, dsize=self.IMG_SIZE)  # 이미지를 설정된 크기로 조정
        eye_input_l = eye_img_l.reshape((1, self.IMG_SIZE[1], self.IMG_SIZE[0], 1)).astype(np.float32) / 255.
        pred_l = self.model.predict(eye_input_l)  # 예측 수행
        state_l = '0' if pred_l > 0.02 else '1'  # 예측 결과에 따라 상태 설정
        return int(state_l)

    # 오른쪽 눈 이미지를 모델에 전달하여 예측하는 함수
    def eye_preR(self, eye_img_r):
        eye_img_r = cv2.resize(eye_img_r, dsize=self.IMG_SIZE)  # 이미지를 설정된 크기로 조정
        eye_input_r = eye_img_r.reshape((1, self.IMG_SIZE[1], self.IMG_SIZE[0], 1)).astype(np.float32) / 255.
        pred_r = self.model.predict(eye_input_r)  # 예측 수행
        state_r = '0' if pred_r > 0.02 else '1'  # 예측 결과에 따라 상태 설정
        return int(state_r)

    # Mediapipe 결과에서 얼굴 랜드마크 좌표를 추출하는 함수
    def landmark_dict(self, results, width, height):
        face_landmark = {}
        # 얼굴 랜드마크를 순차적으로 처리
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # 랜드마크의 visibility와 presence 값이 임계값 이상인 경우에만 처리
                if ((landmark.HasField('visibility') and
                     landmark.visibility < self.VISIBILITY_THRESHOLD) or
                        (landmark.HasField('presence') and
                         landmark.presence < self.PRESENCE_THRESHOLD)):
                    continue
                # 랜드마크 좌표를 픽셀 좌표로 변환
                landmark_px = self.mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y, width, height)
                if landmark_px:
                    face_landmark[idx] = landmark_px
        return face_landmark

    # 얼굴의 특정 지점을 기준으로 얼굴 변화를 계산하는 함수
    def face_change(self, dict):
        # 얼굴 너비 기준선 계산 (예: 두 눈 사이 거리)
        w_line = int(distance.euclidean(dict[227], dict[447]))
        # 입의 변화 계산
        mouth = int(distance.euclidean(dict[61], dict[291]))
        res_mouth = int((mouth / w_line) * 100)
        # 광대뼈 거리 계산
        cheekbones = int(distance.euclidean(dict[50], dict[280]))
        res_cheekbones = int((cheekbones / w_line) * 100)
        # 눈썹 변화 계산
        brow = int(distance.euclidean(dict[107], dict[336]))
        res_brow = int((brow / w_line) * 100)
        # 팔자주름 계산
        Nasolabial_Folds = int(distance.euclidean(dict[205], dict[425]))
        Nasolabial_Folds_res = int((Nasolabial_Folds / w_line) * 100)

        return res_mouth, res_cheekbones, res_brow, Nasolabial_Folds_res

    # 임시 파일을 삭제하는 함수
    def del_file(self, path):
        try:
            os.remove(path)  # 파일 삭제
        except:
            pass  # 예외 발생 시 무시

    # 비디오 파일을 처리하여 눈 깜박임 및 얼굴 변화를 계산하는 함수
    def face_run(self, video_file):
        # 초기 변수 설정
        eye_cnt = 0
        frame = 0
        eye = 0
        eye_list = []
        Nasolabial_Folds_list = []
        brow_list = []
        cheekbones_list = []
        mouth_list = []

        # 비디오 파일을 열고 속성 설정
        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        m_fps = cap.get(cv2.CAP_PROP_FPS) * 60  # 1분 단위의 프레임 설정
        face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        check = EyeCheck()  # EyeCheck 객체 생성

        # 비디오가 열려 있는 동안 프레임을 하나씩 읽어서 처리
        while cap.isOpened():
            ret, image = cap.read()  # 프레임 읽기

            if not ret:  # 비디오 끝에 도달하면 루프 종료
                print("End frame")
                eye_list.append(int(eye_cnt / 2))  # 깜박임 수를 추가
                break

            frame += 1
            if frame % m_fps == 0:
                eye_list.append(int(eye_cnt / 2))
                eye_cnt = 0
                frame = 0

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 회색조로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
            results = face_mesh.process(image)  # Mediapipe로 얼굴 랜드마크 추출

            image.flags.writeable = True  # 이미지를 다시 쓰기 가능 상태로 변경
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지를 다시 BGR로 변환

            eye_image = image.copy()  # 눈 부분 이미지를 복사

            if results.multi_face_landmarks:  # 얼굴 랜드마크가 있으면 처리
                idx_to_coordinates = check.landmark_dict(results, width, height)  # 랜드마크 좌표 추출
                eye_np = check.to_ndarray(idx_to_coordinates)  # 눈 랜드마크 추출
                eye_img_l, eye_rect_l = check.crop_eye(gray, eye_points=eye_np[0:16])  # 왼쪽 눈 이미지 추출
                eye_img_r, eye_rect_r = check.crop_eye(gray, eye_points=eye_np[16:32])  # 오른쪽 눈 이미지 추출

                # 왼쪽 및 오른쪽 눈 예측 수행
                state_r = check.eye_preR(eye_img_r)
                state_l = check.eye_preL(eye_img_l)

                # 눈 상태에 따라 깜박임 여부 계산
                state = 1 if state_l == 1 or state_r == 1 > 0.05 else 0

                if eye != state:
                    eye = state
                    eye_cnt += 1
                else:
                    eye = state

                # 얼굴 변화 계산
                res_mouth, res_cheekbones, res_brow, Nasolabial_Folds_res = check.face_change(idx_to_coordinates)

                # 계산된 결과를 리스트에 추가
                mouth_list.append(res_mouth)
                cheekbones_list.append(res_cheekbones)
                brow_list.append(res_brow)
                Nasolabial_Folds_list.append(Nasolabial_Folds_res)

            if cv2.waitKey(1) == ord('q'):  # 'q' 키를 누르면 루프 종료
                eye_list.append(int(eye_cnt / 2))
                break

        face_mesh.close()  # Mediapipe face mesh 종료
        cap.release()  # 비디오 캡처 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

        # 얼굴 변화 데이터를 JSON 형식으로 반환
        face_json = {
            'eye': {
                'x': len(eye_list),
                'y': eye_list
            },
            'face': {
                'mouth': {
                    'x': len(mouth_list),
                    'y': mouth_list
                },
                'cheekbones_list': {
                    'x': len(cheekbones_list),
                    'y': cheekbones_list
                },
                'brow': {
                    'x': len(brow_list),
                    'y': brow_list
                },
                'nasolabial_folds': {
                    'x': len(Nasolabial_Folds_list),
                    'y': Nasolabial_Folds_list
                },
                'm_fps': m_fps
            }
        }

        # 눈 깜박임 횟수를 막대그래프로 시각화
        eye_bar_graph = plot_bar_graph(list(range(len(face_json['eye']['y']))), face_json['eye']['y'], 'Eye Blinking Over Time',
                                       'Time', 'Blink Count')

        # 얼굴 변화 데이터를 선 그래프로 시각화
        mouth_graph = plot_line_graph(list(range(len(face_json['face']['mouth']['y']))), face_json['face']['mouth']['y'],
                                      'Mouth Movement', 'Time', 'Mouth Distance')
        cheekbones_graph = plot_line_graph(list(range(len(face_json['face']['cheekbones_list']['y']))),
                                           face_json['face']['cheekbones_list']['y'], 'Cheekbones Movement', 'Time',
                                           'Cheekbones Distance')
        brow_graph = plot_line_graph(list(range(len(face_json['face']['brow']['y']))), face_json['face']['brow']['y'],
                                     'Brow Movement', 'Time', 'Brow Distance')
        nasolabial_folds_graph = plot_line_graph(list(range(len(face_json['face']['nasolabial_folds']['y']))),
                                                 face_json['face']['nasolabial_folds']['y'], 'Nasolabial Folds Movement',
                                                 'Time', 'Nasolabial Folds Distance')

        # 그래프 데이터를 JSON에 포함
        face_json['graphs'] = {
            'eye_bar_graph': eye_bar_graph,
            'mouth_graph': mouth_graph,
            'cheekbones_graph': cheekbones_graph,
            'brow_graph': brow_graph,
            'nasolabial_folds_graph': nasolabial_folds_graph
        }

        check.del_file(video_file)  # 임시 비디오 파일 삭제

        return face_json  # 결과 반환


# 비디오 파일의 내용을 받아서 처리하는 함수
def process_video(file_contents: bytes) -> Dict:
    # 임시 비디오 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(file_contents)  # 비디오 데이터 작성
        temp_file_path = temp_file.name  # 파일 경로 저장

    print(f"Temporary file path: {temp_file_path}")  # 임시 파일 경로 출력

    try:
        eye_check = EyeCheck()  # EyeCheck 객체 생성
        result = eye_check.face_run(temp_file_path)  # 비디오 파일 처리
    finally:
        # 파일이 존재하는지 확인하고 삭제
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)  # 파일 삭제
        else:
            print(f"File not found: {temp_file_path}")

    return result  # 처리 결과 반환
