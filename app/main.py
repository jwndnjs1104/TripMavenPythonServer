from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware #fastapi 서버의 CORS 설정
import asyncio
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from app.models.users import Users
from app.db.session import get_db

app = FastAPI()

# Spring Boot 서버의 도메인을 여기에 추가
origins = [
    "http://localhost:9099" # 예시로, Spring Boot 서버가 실행되는 도메인
]

#스프링에서 오는 요청 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 출처 목록
    allow_credentials=True,  # 쿠키, 인증 정보 등을 포함한 요청 허용
    allow_methods=["*"],  # 모든 HTTP 메서드 허용(GET, POST, PUT, DELETE 등)
    allow_headers=["*"]  # 모든 헤더 허용
)

#기본 엔트포인트 설정
@app.get("/")
async def read_root():
    return {"message": "Hello World"}

#db 연결 예시 코드
@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)): #세션 객체 의존성 주입 받는다, db연결을 위한 세션 객체임
    #db.query(Users)는 Users테이블에 대해서 쿼리를 실행한다는 뜻
    #filter(Users.id == user_id)는 where 절에 해당한다고 보면 됨
    #first()는 결과의 첫 행만을 반환한다는 뜻
    #만약 쿼리 결과가 없으면 None을 반환
    user = db.query(Users).filter(Users.id == user_id).first()

    # 모든 사용자 데이터를 조회하는 쿼리
    # 결과가 없으면 빈 리스트 반환
    # users = db.query(Users).filter(Users.id > 0).all()

    # User와 Address를 조인하여 모든 사용자와 그들의 주소를 가져오는 쿼리
    # query = db.query(User, Address).join(Address, User.id == Address.user_id).all()


    # 특정 사용자(예: username이 'johndoe'인 사용자)의 주소만 가져오기
    # query = db.query(Address).join(User).filter(User.username == 'johndoe').all()
    # 위에서 조인 컬럼을 지정하지 않았는데, 테이블 정의시 미리 relationship으로 미리 정의하면
    # 조인 컬럼을 직접 지정하지 않고 자동으로 처리함
    # class User(Base):
    #     __tablename__ = 'users'
    #
    #     id = Column(Integer, primary_key=True, index=True)
    #     username = Column(String, index=True)
    #     email = Column(String, unique=True, index=True)
    #
    #     addresses = relationship("Address", back_populates="user")
    #
    # class Address(Base):
    #     __tablename__ = 'addresses'
    #
    #     id = Column(Integer, primary_key=True, index=True)
    #     user_id = Column(Integer, ForeignKey('users.id'))
    #     address = Column(String)
    #
    #     user = relationship("User", back_populates="addresses")


    # Left outer join 예시: 모든 사용자와, 해당 사용자가 있을 경우 주소를 가져옴
    # query = db.query(User, Address).outerjoin(Addrtess, User.id == Address.user_id).all()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

#====================================================================================================
#실질적인 api사용하기 위해서는 아래와 같이 하면 될 것 같음

#api 사용
#api 패키지 내에서 파일 가져오기
from app.api import ocr
from app.api import stt
# 라우터를 등록
# main.py에서 기본 라우팅 하는게 아니라 api패키지에 있는 각 파일에서 APIRouter객체를 이용해 라우팅하고 main에서 라우터 등록
app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
app.include_router(stt.router, prefix="/stt", tags=["STT"])

#큐넷 자격증 진위확인
#성명
#생년월일 (주민등록번호 앞 6자리)
#자격증번호 (12345678901A)
#발급연월일 (20050101)
#자격증내지번호 (0901234567) #2009년 8월 3일 이후 발행자격증은 반드시 기재
#셀레니움 사용해서 하면 될거같은데...
#https://www.q-net.or.kr/qlf006.do?id=qlf00601&gSite=Q&gId=

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)