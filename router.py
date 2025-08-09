# router.py
# --- 1. 필요한 라이브러리 및 모듈 임포트 ---
from fastapi import (
    APIRouter,          # API 라우터를 생성하기 위해 필요
    Depends,            # 의존성 주입을 위해 필요 (예: DB 세션)
    BackgroundTasks,    # 백그라운드 작업을 위해 필요
    File,               # 파일 업로드를 위해 필요 (`image_file`)
    Form,               # 폼 데이터 입력을 위해 필요 (`user_question`)
    UploadFile,         # 업로드된 파일의 타입 힌트
    HTTPException       # HTTP 예외를 발생시키기 위해 필요
)
from sqlalchemy.orm import Session # DB 세션의 타입 힌트
from pydantic import BaseModel, Field # 요청 Body의 데이터 모델을 정의하기 위해 필요
import uuid

# --- 내부 모듈 임포트 ---
from api import db # DB 연결을 위한 의존성 주입 함수
from api import ai_service # 우리의 핵심 비즈니스 로직

# --- 2. API 라우터 생성 ---
# 이 라우터에 정의된 모든 경로는 나중에 main.py에서 '/api/v1' 같은 접두사와 함께 등록됩니다.
router = APIRouter()

# --- 3. 요청 Body를 위한 데이터 모델 정의 ---
# '대화 이어가기'는 JSON 형식으로 요청을 받으므로, Pydantic 모델로 구조를 정의합니다.
class ContinueChatRequest(BaseModel):
    conversation_id: str = Field(..., description="이전 대화에서 받은 고유한 대화 ID")
    user_question: str = Field(..., description="이미지에 대한 추가 질문")

# 응답 모델 정의
class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
# --- 4. API 엔드포인트 정의 ---

@router.post("/chat/start", summary="새로운 대화 시작", response_model=ChatResponse)
async def start_chat(
    # FastAPI는 아래 파라미터들을 보고, 요청의 종류와 필요한 자원을 파악하여 자동으로 주입해줍니다.
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    image_file: UploadFile = File(..., description="분석을 요청할 스크린샷 이미지 파일"),
    user_question: str = Form(..., description="이미지에 대한 첫 번째 질문")
):
    """
    [API 설명]
    이미지와 첫 번째 질문을 받아 새로운 대화를 시작합니다.
    성공 시, AI의 답변과 앞으로 대화를 이어갈 때 사용할 고유한 'conversation_id'를 반환합니다.
    DB 저장은 백그라운드에서 비동기적으로 처리됩니다.
    """
    try:
        # 모든 복잡한 로직은 ai_service가 처리합니다.
        # 라우터는 그저 적절한 인자들을 넘겨주는 '중개자' 역할을 합니다.
        response_data = await ai_service.start_new_chat_session(
            db=db,
            background_tasks=background_tasks,
            image_file=image_file,
            user_question=user_question
        )
        # 서비스로부터 받은 결과를 그대로 클라이언트에게 반환합니다.
        return response_data
        
    except HTTPException as e:
        # ai_service에서 발생시킨 HTTPException은 그대로 클라이언트에게 전달합니다.
        raise e
    except Exception as e:
        # 예상치 못한 다른 모든 오류에 대한 처리
        # 보안을 위해 상세한 오류 내용을 클라이언트에게 직접 노출하지 않습니다.
        print(f"[ROUTER ERROR] /chat/start: {e}")
        raise HTTPException(status_code=500, detail="서버 내부에서 예상치 못한 오류가 발생했습니다.")


@router.post("/chat/continue", summary="기존 대화 이어가기",response_model=ChatResponse)
async def continue_chat(
    request: ContinueChatRequest, # 위에서 정의한 Pydantic 모델을 사용하여 요청 Body를 받음
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    [API 설명]
    'conversation_id'와 추가 질문을 받아, 이전 대화의 맥락을 이어갑니다.
    이미지 파일은 다시 보내지 않습니다.
    """
    try:
        # Pydantic 모델에서 값을 추출하여 서비스 함수에 전달합니다.
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