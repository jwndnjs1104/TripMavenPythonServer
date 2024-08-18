from sqlalchemy import Column, Integer, String
from app.db.base_class import Base

class Users(Base):
    __tablename__ = 'users'  # 이미 존재하는 테이블의 이름

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    full_name = Column(String(100))
