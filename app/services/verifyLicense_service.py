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
# 표준 라이브러리
import os, time

class VerifyLicenseService:
    def verify(self, name, regnum1, regnum2):
        try:
            # WebDriver 객체 생성
            driver_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'chromedriver.exe')
            service = Service(executable_path=driver_path)
            options = webdriver.ChromeOptions()
            options.add_argument("--headless")  # 헤드리스 모드
            options.add_argument("--window-size=1920x1080")  # 창 크기 설정

            driver = webdriver.Chrome(service=service, options=options)
            driver.implicitly_wait(3)  # 랜덤하게 3~5초 사이의 초로 지연 설정

            # 사이트 띄우기
            driver.get('https://www.q-net.or.kr/qlf006.do?id=qlf00601&gSite=Q&gId=')

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
