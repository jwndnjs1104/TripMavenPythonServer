from google.cloud import vision
from google.oauth2 import service_account
import base64
import os

class ObjectDetectionService:
    def __init__(self):
        self.credentials = os.environ['GOOGLE_APPLICATION_CREDENTIALS']

    def authenticate_service_account(self):
        credentials = service_account.Credentials.from_service_account_file(self.credentials)
        scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
        return scoped_credentials

    def detect_init(self, base64Encoded):
        credentials = self.authenticate_service_account()
        image_content = base64.b64decode(base64Encoded)
        client = vision.ImageAnnotatorClient(credentials=credentials)
        image = vision.Image(content=image_content)
        return image, client

    def object_detect(self, base64Encoded):
        image, client = self.detect_init(base64Encoded)
        response = client.label_detection(image=image)
        labels = response.label_annotations
        responseLabels = []
        for label in labels:
            responseLabels.append(f'객체: {label.description},  정확도: {label.score}\n')
        return responseLabels
