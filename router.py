# router.py

# --- 1. 외부 라이브러리 임포트 ---
from fastapi import (
    APIRouter,
    Depends,
    BackgroundTasks, # FastAPI 의존성이므로 기본값 할당 불필요
    File,
    Form,
    UploadFile,
    HTTPException
)
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import uuid

# --- 2. 내부 모듈 임포트 ---
# !!! 경고: 이 import 경로는 GitHub 저장소 기준으로 작성되었습니다.
# !!! 사용자님의 실제 프로젝트 구조에 맞게 이 부분을 직접 확인하고 수정하셔야 합니다.
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
    # [!!! 논리 수정 사항 !!!]
    # BackgroundTasks는 FastAPI가 주입해주는 의존성이므로, 기본값을 직접 생성하지 않습니다.
    background_tasks: BackgroundTasks,
    image_file: UploadFile = File(..., description="분석을 요청할 스크린샷 이미지 파일"),
    user_question: str = Form(..., description="이미지에 대한 첫 번째 질문"),
    db: Session = Depends(get_db)
):
    """
    (API 설명 생략)
    """
    try:
        response_data = await ai_service.start_new_chat_session(
            db=db,
            background_tasks=background_tasks,
            image_file=image_file,
            user_question=user_question
        )
        return response_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ROUTER ERROR] /chat/start: {e}")
        raise HTTPException(status_code=500, detail="서버 내부에서 예상치 못한 오류가 발생했습니다.")


@router.post("/chat/continue", summary="기존 대화 이어가기", response_model=ChatResponse)
async def continue_chat(
    request: ContinueChatRequest,
    # 여기도 마찬가지로 기본값 없이 선언합니다.
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    (API 설명 생략)
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