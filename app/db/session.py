from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.connection import SessionLocal

# 의존성 주입을 통해 세션을 제공
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()