import logging
import httpx
from typing import Optional

# 1. BackgroundTasks를 fastapi에서 임포트해야 합니다.
from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile, HTTPException

from api import ai_service
from api import notification_service

# 로거 설정
log = logging.getLogger("uvicorn")

# APIRouter 인스턴스 생성
router = APIRouter()


@router.post("/start_chat")
async def start_chat_endpoint(
    # 2. 엔드포인트 함수가 BackgroundTasks를 파라미터로 받도록 합니다.
    #    FastAPI가 이 부분을 보고 자동으로 BackgroundTasks 객체를 여기에 넣어줍니다.
    background_tasks: BackgroundTasks,
    request: Request,
    user_question: str = Form(...),
    image_file: UploadFile = File(...),
):
    """
    새로운 대화를 시작하는 엔드포인트.
    """
    

    log.info(
        f"[ROUTER] /start_chat received from={request.client.host} | "
        f"file={'yes' if image_file else 'no'} | "
        f"q_len={len(user_question)}"
    )

    # 3. ai_service 함수를 호출할 때, 주입받은 background_tasks를 그대로 전달합니다.
    return await ai_service.start_new_chat_session(
        background_tasks=background_tasks,
        image_file=image_file,
        user_question=user_question,
    )


@router.post("/continue_chat")
async def continue_chat_endpoint(
    # 여기도 마찬가지로 BackgroundTasks를 받도록 수정합니다.
    background_tasks: BackgroundTasks,
    conversation_id: str = Form(...),
    user_question: str = Form(...),
):
    """
    기존 대화를 이어가는 엔드포인트.
    """
    log.info(
        f"[ROUTER] /continue_chat received | cid={conversation_id} | q_len={len(user_question)}"
    )

    # ai_service 함수를 호출할 때 background_tasks를 전달합니다.
    return await ai_service.continue_chat_endpoint(
        background_tasks=background_tasks,
        conversation_id=conversation_id,
        user_question=user_question,
    )


@router.post("/test") #start_chat으로 마찬가지로 분석한내용 이거는 ios로 알림 보내는거 지금 저장된 intial과 token으로
async def start_test(
    # 2. 엔드포인트 함수가 BackgroundTasks를 파라미터로 받도록 합니다.
    #    FastAPI가 이 부분을 보고 자동으로 BackgroundTasks 객체를 여기에 넣어줍니다.
    background_tasks: BackgroundTasks,
    request: Request,
    user_question: str = Form(...),
    image_file: UploadFile = File(...),
):
    """
    새로운 대화를 시작하는 엔드포인트.
    """
    

    log.info(
        f"[ROUTER] /start_chat received from={request.client.host} | "
        f"file={'yes' if image_file else 'no'} | "
        f"q_len={len(user_question)}"
    )

    # 3. ai_service 함수를 호출할 때, 주입받은 background_tasks를 그대로 전달합니다.
    return await ai_service.start_test(
        background_tasks=background_tasks,
        image_file=image_file,
        user_question=user_question
    )

@router.post("/continue_test")
async def continue_chat_test(
    # 여기도 마찬가지로 BackgroundTasks를 받도록 수정합니다.
    background_tasks: BackgroundTasks,
    conversation_id: str = Form(...),
    user_question: str = Form(...),
):
    """
    기존 대화를 이어가는 엔드포인트.
    """
    log.info(
        f"[ROUTER] /continue_chat received | cid={conversation_id} | q_len={len(user_question)}"
    )

    # ai_service 함수를 호출할 때 background_tasks를 전달합니다.
    return await ai_service.continue_test(
        background_tasks=background_tasks,
        conversation_id=conversation_id,
        user_question=user_question,
    )


@router.post("/register_device")
async def register_device_endpoint(
    installation_id: Optional[str] = Form("123"),
    platform: Optional[str] = Form("apns"),
    device_token: str = Form(...),
    tags: Optional[str] = Form(None),
):
    """
    기기를 Notification Hub에 등록하거나 업데이트하는 엔드포인트.
    
    :param installation_id: 기기를 식별하는 고유 ID (예: UUID).
    :param platform: 'apns' (Apple), 'gcm' (Android/Firebase).
    :param device_token: APNs 또는 FCM에서 받은 디바이스 푸시 토큰.
    :param tags: 이 기기에 할당할 태그 리스트 (콤마로 구분된 문자열).
    """
    # 태그가 문자열로 전달되면 리스트로 변환
    tag_list = tags.split(",") if tags else []

    # Notification Hub에 설치 정보를 등록/업데이트
    success, status_code = await notification_service.create_or_update_installation(
        installation_id=installation_id,
        platform=platform,
        device_token=device_token,
        tags=tag_list,
    )       

    if not success:
        raise HTTPException(status_code=status_code, detail="Installation 등록 실패")

    return {"message": "Installation 등록/업데이트 성공", "status_code": status_code}

