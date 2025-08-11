# router.py

# --- 1. 외부 라이브러리 임포트 ---
import time
from fastapi import (
    APIRouter,
    Depends,
    BackgroundTasks,
    File,
    Form,
    UploadFile,
    HTTPException
)
from fastapi.responses import Response # favicon 처리를 위해 추가
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import uuid

# --- 2. 내부 모듈 임포트 ---
# 이 경로는 프로젝트 구조에 따라 달라질 수 있습니다.
from api.db import get_db
from api import ai_service

# --- 3. API 라우터 및 모델 정의 ---
router = APIRouter()

class ContinueChatRequest(BaseModel):
    conversation_id: str = Field(..., description="이전 대화에서 받은 고유한 대화 ID")
    user_question: str = Field(..., description="이미지에 대한 추가 질문")

class ChatResponse(BaseModel):
    conversation_id: str
    answer: str

# --- 4. API 엔드포인트 정의 ---
@router.post("/chat/start", summary="새로운 대화 시작", response_model=ChatResponse)
async def start_chat(
    background_tasks: BackgroundTasks,
    image_file: UploadFile = File(..., description="분석을 요청할 스크린샷 이미지 파일"),
    user_question: str = Form(..., description="이미지에 대한 첫 번째 질문"),
    db: Session = Depends(get_db)
):
    """
    새로운 이미지와 질문으로 대화를 시작합니다.
    AI의 답변과 함께 고유한 대화 ID를 반환합니다.
    """
    # --- 시간 측정 로직 시작 ---
    t0 = time.time()
    print("\n--- API 요청 처리 시작 ---")

    try:
        # DB 세션 준비까지 걸린 시간 측정
        t1 = time.time()
        print(f"1. FastAPI 준비 및 DB 세션 생성 소요 시간: {t1 - t0:.4f}초")

        # 핵심 비즈니스 로직 호출
        response_data = await ai_service.start_new_chat_session(
            db=db,
            background_tasks=background_tasks,
            image_file=image_file,
            user_question=user_question
        )
        
        # 핵심 로직(AI 호출 포함) 실행에 걸린 시간 측정
        t2 = time.time()
        print(f"2. AI 서비스 핵심 로직 소요 시간: {t2 - t1:.4f}초")
        
        print(f"★ 총 API 응답 소요 시간 (백그라운드 제외): {t2 - t0:.4f}초 ★")
        return response_data
        
    except HTTPException as e:
        # 이미 처리된 HTTP 예외는 그대로 다시 발생시킵니다.
        raise e
    except Exception as e:
        # 그 외의 모든 예외를 처리합니다.
        print(f"[ROUTER ERROR] /chat/start: {e}")
        raise HTTPException(status_code=500, detail="서버 내부에서 예상치 못한 오류가 발생했습니다.")


@router.post("/chat/continue", summary="기존 대화 이어가기", response_model=ChatResponse)
async def continue_chat(
    request: ContinueChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    기존 대화 ID를 사용하여 대화를 계속합니다.
    """
    try:
        response_data = await ai_service.continue_existing_chat(
            db=db,
            background_tasks=background_tasks,
            conversation_id=request.conversation_id,
            user_question=request.user_question
        )
        return response_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ROUTER ERROR] /chat/continue: {e}")
        raise HTTPException(status_code=500, detail="서버 내부에서 예상치 못한 오류가 발생했습니다.")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    웹 브라우저의 불필요한 favicon.ico 요청에 대한 404 로그를 방지합니다.
    """
    return Response(status_code=204)