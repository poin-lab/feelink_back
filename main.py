# main.py
import logging
import sys
from fastapi import FastAPI
from dotenv import load_dotenv
# 같은 폴더에 있는 router.py 파일에서 'router' 라는 변수를 임포트합니다.
from router import router

logging.basicConfig(
    level=logging.INFO,  # INFO 레벨 이상의 로그를 모두 기록 (DEBUG로 변경 가능)
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# .env 파일은 이 파일과 같은 위치에 있으므로 여기서 로드합니다.
load_dotenv()

app = FastAPI(title="Feelink Chatbot Server")

# router 변수 자체를 포함시킵니다.
app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "ok"}