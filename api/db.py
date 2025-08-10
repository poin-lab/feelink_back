# api/db.py

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# [!!! 핵심 수정 사항 !!!]
# 순환 참조 문제를 해결하기 위해, 모든 모델의 기반이 되는 Base 클래스는
# models.py에서 직접 정의합니다. 따라서 이 파일에서는 관련 코드를 삭제합니다.

# 1. .env 파일 로드
load_dotenv()

# 2. 환경 변수에서 데이터베이스 연결 URL 직접 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("데이터베이스 연결을 위한 DATABASE_URL 환경 변수가 설정되지 않았습니다.")

# 3. 데이터베이스 엔진 생성
engine = create_engine(
    DATABASE_URL,
    connect_args={}  # <--- 바로 이 한 줄이 모든 것을 해결합니다.
)
# 4. 세션 메이커(Session Maker) 생성
# 이 SessionLocal을 통해 DB와의 개별적인 대화(세션)를 시작합니다.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# 5. 의존성 주입용 함수 생성
# FastAPI의 Depends()가 이 함수를 호출하여 각 API 요청마다
# 독립적인 DB 세션을 생성하고, 요청 처리가 끝나면 안전하게 닫습니다.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 이 파일은 이제 데이터베이스 연결 설정과 세션 제공 역할만 수행합니다.