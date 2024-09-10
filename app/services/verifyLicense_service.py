# 자격증 진위확인 사이트
# https://www.q-net.or.kr/qlf006.do?id=qlf00601&gSite=Q&gId=

# 웹 드라이버 생성을 위한 모듈
from selenium import webdriver

# 웹 드라이버 생성을 위한 서비스 클래스
from selenium.webdriver.chrome.service import Service
# 키보드 키값이 정의된 클래스
#from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import Select

# 위치 지정자(셀렉터)를 위한 클래스
from selenium.webdriver.common.by import By
# 지정한 시간동안 요소를 못 찾을때 발생하는 예외
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# 표준 라이브러리
import os, time
import base64
from io import BytesIO
from PIL import Image

#ocr
from app.services.ocr_service import OCRService
ocr_service = OCRService()

class VerifyLicenseService:

    def createDriver(self):
        # WebDriver 객체 생성
        driver_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chromedriver.exe')
        service = Service(executable_path=driver_path)
        options = webdriver.ChromeOptions()
        #options.add_argument("--headless")  # 헤드리스 모드
        options.add_argument("--window-size=1920x1080")  # 창 크기 설정

        driver = webdriver.Chrome(service=service, options=options)
        driver.implicitly_wait(4)  # 랜덤하게 3~5초 사이의 초로 지연 설정

        # 사이트 띄우기
        driver.get('https://www.q-net.or.kr/iss004.do?id=iss00401&gSite=Q&gId=')
        # 자격증 진위확인
        # https://www.q-net.or.kr/qlf006.do?id=qlf00601&gSite=Q&gId=
        # 확인서 진위확인
        # https://www.q-net.or.kr/iss004.do?id=iss00401&gSite=Q&gId=

        return driver

    #자격증 진위확인
    def verify(self, name, regnum1, regnum2):
        try:
            driver = self.createDriver()

            # 자격증 종류 셀렉트 박스 찾기
            selectbox = driver.find_element(By.CSS_SELECTOR, '#content > div.content > form:nth-child(2) > div.tbl_normal.nmlType3 > table > tbody > tr:nth-child(1) > td > select')

            # Select 객체 생성 후 선택(텍스트로 선택)
            select = Select(selectbox)
            select.select_by_visible_text('상장형 자격증')

            # 이름, 숫자 앞자리, 숫자 뒷자리 얻고 넣기
            nameInput = driver.find_element(By.CSS_SELECTOR,'#hgulNm2')
            nameInput.send_keys(name)

            no1Input = driver.find_element(By.CSS_SELECTOR, '#hrdNo1')
            no1Input.send_keys(regnum1)

            no2Input = driver.find_element(By.CSS_SELECTOR, '#hrdNo2')
            no2Input.send_keys(regnum2)

            # 확인 버튼 누르기
            button = driver.find_element(By.CSS_SELECTOR, '#content > div.content > form:nth-child(2) > div > div.btn_center > button.btn2.btncolor2 > span')
            button.click()

            # alert 창에서 결과 확인(자격증이 있을때 알람을 몰라서 지금은 확인 불가능)
            alert = driver.switch_to.alert
            if alert.text == '발급일로부터 90일 이후 자료는 조회가 불가능합니다.':
                #조회된 자격증 번호가 없을때 False 반환
                print(False)
                return False
            return True

        except NoSuchElementException as e:
            return {"errorMsg":e.msg}
        finally:
            #3초 후 브라우저 닫기
            time.sleep(3)
            if driver:
                driver.quit()


    # 확인서 진위확인
    def verify2(self, name, regnum1, regnum2):
        try:
            driver = self.createDriver()
            driver.maximize_window()
            print(name, regnum1, regnum2)

            # 이름, 숫자 앞자리, 숫자 뒷자리 얻고 넣기
            nameInput = driver.find_element(By.CSS_SELECTOR, '#nm')
            nameInput.send_keys(name)

            no1Input = driver.find_element(By.CSS_SELECTOR, '#content > div.content > div.tbl_type2.mb10 > table > tbody > tr > td:nth-child(4) > span > input:nth-child(2)')
            no1Input.send_keys(regnum1)

            no2Input = driver.find_element(By.CSS_SELECTOR, '#content > div.content > div.tbl_type2.mb10 > table > tbody > tr > td:nth-child(4) > span > input:nth-child(3)')
            no2Input.send_keys(regnum2)

            # 확인 버튼 누르기
            button = driver.find_element(By.CSS_SELECTOR,
                                         '#content > div.content > div.tbl_type2.mb10 > table > tbody > tr > td:nth-child(4) > span > button')
            # 기존 창의 핸들을 저장
            main_window_handle = driver.current_window_handle
            print('이전 드라이버', driver)

            button.click()
            time.sleep(3)  # 새 창이 열리도록 대기 (더 나은 방법은 WebDriverWait을 사용하는 것)

            # 모든 창의 핸들을 가져옴, # 새로 열린 창으로 전환
            all_window_handles = driver.window_handles
            print(all_window_handles)
            new_window_handle = [handle for handle in all_window_handles if handle != main_window_handle][0]
            print(new_window_handle)
            driver.switch_to.window(new_window_handle)
            print('새창 드라이버', driver)

            try:
                # Alert가 나타날 때까지 기다림
                WebDriverWait(driver, 10).until(EC.alert_is_present())

                # Alert로 전환하고 텍스트 확인
                alert = driver.switch_to.alert
                print(f"Alert text: {alert.text}")
                time.sleep(1)  # 잠시 대기

                # alert 창에서 결과 확인
                if '정상적으로 확인' in alert.text:
                    return {'isSuccess': True}
                else:
                    return {'isSuccess': False}
            except:
                print("No alert found or alert handling failed.")

        except NoSuchElementException as e:
            print(e.msg)
            return {'isSuccess':False}
        finally:
            #pass
            #3초 후 브라우저 닫기
            time.sleep(2)
            if driver:
                driver.quit()

