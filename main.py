# main.py
import logging
from contextlib import asynccontextmanager
import sys
from fastapi import FastAPI
from dotenv import load_dotenv
# 같은 폴더에 있는 router.py 파일에서 'router' 라는 변수를 임포트합니다.
from router import router
from api import ai_service


# --- FastAPI 앱 객체 생성 및 Lifespan 연결 ---
app = FastAPI(
    title="Feelink Chatbot Server"
)

# router 변수 자체를 포함시킵니다.
app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "ok"}