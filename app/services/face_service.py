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
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 표정 분석 결과에 따라 코멘트를 생성하는 함수
def generate_expression_comment(mouth_score, cheekbones_score, brow_score, nasolabial_folds_score):
    comments = []

    # 입 움직임에 따른 코멘트
    if mouth_score <= 25:
        comments.append("입 움직임이 거의 없으며, 말할 때 조용하고 감정 표현이 적습니다.")
        comments.append("입술의 움직임이 거의 없기 때문에 비언어적인 표현도 감소하는 것으로 보입니다.")
    elif 25 < mouth_score <= 50:
        comments.append("입 움직임이 다소 적어 말이 조용하고 차분한 편입니다.")
        comments.append("자신감이 부족하거나, 차분한 태도를 유지하려는 경향일 수 있습니다.")
    elif 50 < mouth_score <= 75:
        comments.append("입 움직임이 적당하며, 일반적인 대화 중에 감정 표현이 잘 드러납니다.")
        comments.append("말하는 동안 적절한 비언어적 표현이 이루어지고 있습니다.")
    else:
        comments.append("입 움직임이 크며, 매우 활발하게 말하고 있는 것으로 보입니다.")
        comments.append("자신감이 높으며 적극적으로 의사소통을 하고 있습니다.")

    # 광대뼈 움직임에 따른 코멘트
    if cheekbones_score <= 25:
        comments.append("광대뼈 움직임이 거의 없으며, 미소가 적거나 무표정 상태일 가능성이 큽니다.")
        comments.append("감정적인 반응이 거의 드러나지 않아 다소 딱딱한 인상을 줄 수 있습니다.")
    elif 25 < cheekbones_score <= 50:
        comments.append("광대뼈 움직임이 다소 적어, 미소가 드문 편입니다.")
        comments.append("미소가 억제되었거나, 상황에 따라 미묘하게 표현된 것일 수 있습니다.")
    elif 50 < cheekbones_score <= 75:
        comments.append("광대뼈 움직임이 적당하며, 미소가 자연스럽게 나타납니다.")
        comments.append("긍정적인 감정이 적절히 드러나며 친근한 인상을 줍니다.")
    else:
        comments.append("광대뼈 움직임이 커서 웃음이 많고 긍정적인 표정이 두드러집니다.")
        comments.append("웃음이 크고 자연스러워, 매우 즐거운 상태를 보여줍니다.")

    # 눈썹 움직임에 따른 코멘트
    if brow_score <= 25:
        comments.append("눈썹 움직임이 거의 없어, 감정 표현이 잘 드러나지 않습니다.")
        comments.append("얼굴 전반에 감정 변화가 거의 없어 침착해 보입니다.")
    elif 25 < brow_score <= 50:
        comments.append("눈썹 움직임이 적어, 감정 변화가 미미하게 나타납니다.")
        comments.append("눈썹의 약간의 움직임은 감정을 미묘하게 드러내고 있습니다.")
    elif 50 < brow_score <= 75:
        comments.append("눈썹 움직임이 적당하며, 감정 표현이 자연스럽습니다.")
        comments.append("눈썹을 통한 감정 표현이 상황에 적절하게 이루어집니다.")
    else:
        comments.append("눈썹 움직임이 크며, 놀라거나 긴장된 상태를 보여줍니다.")
        comments.append("감정이 강하게 드러나며, 순간적인 반응이 뚜렷합니다.")

    # 팔자주름 변화에 따른 코멘트
    if nasolabial_folds_score <= 25:
        comments.append("팔자주름이 거의 나타나지 않아 감정 표현이 적거나 차분한 상태입니다.")
        comments.append("미소나 감정 표현이 최소한으로 유지된 듯 보입니다.")
    elif 25 < nasolabial_folds_score <= 50:
        comments.append("팔자주름이 다소 나타나며, 감정 표현이 부분적으로 보입니다.")
        comments.append("감정 표현이 제한적이지만, 소소한 감정 변화를 반영하고 있습니다.")
    elif 50 < nasolabial_folds_score <= 75:
        comments.append("팔자주름이 적당히 나타나며, 감정 표현이 잘 드러납니다.")
        comments.append("긍정적인 감정이나 즐거움이 비교적 자연스럽게 표현되고 있습니다.")
    else:
        comments.append("팔자주름이 많이 나타나며, 감정 표현이 매우 강하게 드러납니다.")
        comments.append("웃음이나 감정 표현이 강하며 매우 생동감 있는 표정입니다.")

    # 종합적인 상태 평가
    if mouth_score > 75 and cheekbones_score > 75:
        comments.append("전반적으로 매우 긍정적이고 활발한 상태로 보입니다.")
        comments.append("활발한 표정과 감정 표현이 신체 전반에서 자연스럽게 나타나고 있습니다.")
    if brow_score > 75 or nasolabial_folds_score > 75:
        comments.append("긴장한 상태일 수 있으며, 감정적으로 불안해 보일 수 있습니다.")
        comments.append("눈썹이나 팔자주름의 큰 움직임은 감정적 긴장감을 반영할 수 있습니다.")

        # 코멘트를 '*'로 구분된 문자열로 반환
    return '*'.join(comments)


# 눈 깜박임 횟수에 따른 코멘트를 생성하는 함수
def generate_eye_blink_comment(total_blinks, duration_minutes=1):
    # 이상적인 눈 깜박임 횟수 범위를 설정 (일반적으로 1분에 15~20회)
    ideal_blinks_per_minute = (15, 20)

    comments = []  # 여러 개의 코멘트를 리스트로 저장

    if total_blinks < ideal_blinks_per_minute[0]:
        comments.append(f"눈 깜박임 횟수가 {total_blinks}회로, 평균보다 적어 긴장된 상태일 수 있습니다. 1분에 15회에서 20회 깜빡이는 것이 적당합니다.")
        comments.append(f"이는 스트레스나 집중력 증가로 인해 눈을 덜 깜빡이는 상태를 반영할 수 있습니다.")
    elif total_blinks > ideal_blinks_per_minute[1]:
        comments.append(f"눈 깜박임 횟수가 {total_blinks}회로, 평균보다 많아 피곤하거나 매우 불안한 상태일 수 있습니다. 1분에 15회에서 20회 깜빡이는 것이 적당합니다.")
        comments.append(f"눈의 과도한 깜빡임은 눈의 피로감이나 감정적 불안감을 나타낼 수 있습니다.")
    else:
        comments.append(f"눈 깜빡임 횟수가 {total_blinks}회로, 적정 범위 내에 있으며 안정적인 상태입니다.")
        comments.append(f"눈 깜빡임이 정상 범위이므로 집중력과 감정 상태가 비교적 안정적이라고 할 수 있습니다.")

    # 코멘트를 '*'로 구분된 문자열로 반환
    return '*'.join(comments)


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
                eye_list.append(eye_cnt / 2)  # 깜박임 수를 추가
                break

            frame += 1
            if frame % m_fps == 0:
                eye_list.append(eye_cnt / 2)
                eye_cnt = 0
                frame = 0

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 이미지를 회색조로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지를 RGB로 변환
            results = face_mesh.process(image)  # Mediapipe로 얼굴 랜드마크 추출

            image.flags.writeable = True  # 이미지를 다시 쓰기 가능 상태로 변경
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 이미지를 다시 BGR로 변환

            #eye_image = image.copy()  # 눈 부분 이미지를 복사

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
                eye_list.append(eye_cnt / 2)
                break

        face_mesh.close()  # Mediapipe face mesh 종료
        cap.release()  # 비디오 캡처 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 윈도우 닫기

        # 표정 분석 코멘트를 생성하여 문자열로 반환
        expression_comment = generate_expression_comment(
            mouth_list[-1] if mouth_list else 0,
            cheekbones_list[-1] if cheekbones_list else 0,
            brow_list[-1] if brow_list else 0,
            Nasolabial_Folds_list[-1] if Nasolabial_Folds_list else 0
        )

        # 얼굴 변화 데이터를 JSON 형식으로 반환
        face_json = {
            'eye': {
                'total_blinks': sum(eye_list),  # 눈 깜박임 횟수의 총합을 반환
                'average_blinks': sum(eye_list) / len(eye_list) if eye_list else 0,  # 평균 깜박임 횟수 반환
                'comment': generate_eye_blink_comment(sum(eye_list))
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
            },
             'expression_comment': expression_comment  # 표정 분석 코멘트는 문자열로 반환
        }

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
