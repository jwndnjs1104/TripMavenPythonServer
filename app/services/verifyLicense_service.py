# 자격증 진위확인 사이트
# https://www.q-net.or.kr/qlf006.do?id=qlf00601&gSite=Q&gId=
# 웹 드라이버 생성을 위한 모듈
from selenium import webdriver
# 웹 드라이버 생성을 위한 서비스 클래스
from selenium.webdriver.chrome.service import Service
# 키보드 키값이 정의된 클래스
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import Select

# 위치 지정자(셀렉터)를 위한 클래스
from selenium.webdriver.common.by import By
# 지정한 시간동안 요소를 못 찾을때 발생하는 예외
from selenium.common.exceptions import NoSuchElementException
# 표준 라이브러리
import os, random

class VerifyLicenseService:
    def verify(self, name, regnum1, regnum2):
        try:
            # 1.WebDriver 객체 생성
            driver_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chromedriver.exe')
            service = Service(executable_path=driver_path)
            options = webdriver.ChromeOptions()
            options.add_experimental_option('detach', True)
            driver = webdriver.Chrome(service=service, options=options)
            # 2.implicitly_wait(초)로 최대 지정한 초까지 요소가 나타날때까지 조건없이 기다리기
            # (지정 초까지 요소 못 찾을시 예외처리하고자 하는 경우)
            driver.implicitly_wait(random.randint(3, 5))  # 랜덤하게 3~5초 사이의 초로 지연 설정

            # 3.사이트 띄우기
            driver.get('https://www.q-net.or.kr/qlf006.do?id=qlf00601&gSite=Q&gId=')

            # 4.자격증 종류 셀렉트 박스 찾기
            selectbox = driver.find_element(By.CSS_SELECTOR, '#content > div.content > form:nth-child(2) > div.tbl_normal.nmlType3 > table > tbody > tr:nth-child(1) > td > select')

            # 5. Select 객체 생성
            select = Select(selectbox)

            # 6. 셀렉트 박스에서 옵션 선택 (텍스트로 선택)
            select.select_by_visible_text('상장형 자격증')

            # 7. 이름, 숫자 앞자리, 숫자 뒷자리 입력요소 얻기
            # 얻은 요소에다가 이름, 자격증 번호 넣기
            nameInput = driver.find_element(By.CSS_SELECTOR,'#hgulNm2')
            nameInput.send_keys(name)

            no1Input = driver.find_element(By.CSS_SELECTOR, '#hrdNo1')
            no1Input.send_keys(regnum1)

            no2Input = driver.find_element(By.CSS_SELECTOR, '#hrdNo2')
            no2Input.send_keys(regnum2)

            # 8. 확인 버튼 누르기
            button = driver.find_element(By.CSS_SELECTOR, '#content > div.content > form:nth-child(2) > div > div.btn_center > button.btn2.btncolor2 > span')
            button.click()
            # #content > div.content > form:nth-child(2) > div > div.btn_center > button.btn2.btncolor2 > span

            alert = driver.switch_to.alert
            if alert.text == '발급일로부터 90일 이후 자료는 조회가 불가능합니다.':
                #조회된 자격증 번호가 없을때 False 반환
                print(False)
                return False

            return True

        # 요소를 못 찾을때
        # implicitly_wait(조건 없이 지정 시간동안 대기)는 NoSuchElementException 발생
        # explicitly_wait(조건을 만족할때까지 대기)는 TimeoutException 예외가 발생
        except NoSuchElementException as e:
            print('찾는 요소가 없어요:', e)
            return False
        finally:
            pass
            # 3초 후 브라우저 닫기
            # import time
            # time.sleep(3)
            # if driver:
            #     driver.quit()
