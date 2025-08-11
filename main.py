# main.py
import logging
from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI
from dotenv import load_dotenv
# 같은 폴더에 있는 router.py 파일에서 'router' 라는 변수를 임포트합니다.
from router import router
from api import ai_service


# --- 애플리케이션 생명주기(Lifespan) 정의 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱의 시작과 종료 시점에 실행될 작업을 정의합니다.
    """
    # --- 1. 애플리케이션 시작 시 ---
    # ai_service에 있는 모델 초기화 함수를 호출합니다.
    ai_service.initialize_ai_model()
    
    # 이제 앱이 요청을 받을 준비가 되었습니다.
    yield
    
    # --- 2. 애플리케이션 종료 시 ---
    # 앱이 종료될 때 리소스를 정리합니다.
    ai_service.close_ai_model()

# .env 파일은 이 파일과 같은 위치에 있으므로 여기서 로드합니다.
load_dotenv()

# --- FastAPI 앱 객체 생성 및 Lifespan 연결 ---
app = FastAPI(
    lifespan=lifespan,
    title="Feelink Chatbot Server"
)

# router 변수 자체를 포함시킵니다.
app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "ok"}