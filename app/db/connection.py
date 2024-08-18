from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Oracle 데이터베이스 연결 URL 설정
# 형식: oracle+cx_oracle://user:password@hostname:port/?service_name=service_name
DATABASE_URL = "oracle+cx_oracle://TRIPMAVEN:TRIPMAVEN@121.133.84.38:1521/?service_name=XEPDB1"

# 데이터베이스 엔진 생성
engine = create_engine(
    DATABASE_URL,
    pool_recycle=280,  # 연결을 재사용하는 시간(초)
    pool_size=20,      # 연결 풀의 최대 크기
    max_overflow=40,   # 풀 크기 초과 시 추가로 생성할 수 있는 연결 수
    echo=True          # SQLAlchemy가 실행하는 SQL 쿼리를 출력하려면 True로 설정
)

# 세션 로컬 객체 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)



