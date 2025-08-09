# app/db/database.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# 1. .env 파일 로드
# 이 함수는 프로젝트 루트에 있는 .env 파일의 변수들을 환경 변수로 로드합니다.
load_dotenv()

# 2. 환경 변수에서 데이터베이스 연결 URL 직접 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")

# 데이터베이스 URL이 설정되지 않았을 경우를 대비한 예외 처리
if not DATABASE_URL:
    raise ValueError("데이터베이스 연결을 위한 DATABASE_URL 환경 변수가 설정되지 않았습니다.")

# 3. 데이터베이스 엔진 생성
# 가져온 DATABASE_URL을 사용하여 엔진을 설정합니다.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# 4. 세션 메이커(Session Maker) 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 5. 모든 DB 모델의 공통 부모가 될 Base 클래스 정의
Base = declarative_base()

# 6. 의존성 주입용 함수 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()