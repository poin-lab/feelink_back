import logging
from typing import Optional

# 1. BackgroundTasks를 fastapi에서 임포트해야 합니다.
from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile

from api import ai_service

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
    image_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    """
    새로운 대화를 시작하는 엔드포인트.
    """
    # 클라이언트가 image_file 필드에 URL을 넣는 경우를 대비한 보정 로직
    if image_url is None and image_file is None:
        form = await request.form()
        maybe_url = form.get("image_url") or form.get("image_file")
        if isinstance(maybe_url, str) and (
            maybe_url.startswith("http://") or maybe_url.startswith("https://")
        ):
            image_url = maybe_url

    log.info(
        f"[ROUTER] /start_chat received from={request.client.host} | "
        f"file={'yes' if image_file else 'no'} | url={'yes' if image_url else 'no'} | "
        f"q_len={len(user_question)}"
    )

    # 3. ai_service 함수를 호출할 때, 주입받은 background_tasks를 그대로 전달합니다.
    return await ai_service.start_new_chat_session(
        background_tasks=background_tasks,
        image_file=image_file,
        user_question=user_question,
        image_url=image_url,
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
    return await ai_service.continue_existing_chat(
        background_tasks=background_tasks,
        conversation_id=conversation_id,
        user_question=user_question,
    )