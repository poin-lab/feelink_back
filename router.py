# router.py
from fastapi import APIRouter, UploadFile, File, Form, Request
from typing import Optional
from api import ai_service
import logging

router = APIRouter()
log = logging.getLogger("uvicorn")

@router.post("/start_chat")
async def start_chat_endpoint(
    request: Request,
    user_question: str = Form(...),
    image_file: Optional[UploadFile] = File(None),   # 파일 없어도 허용
    image_url: Optional[str] = Form(None),           # URL도 허용
):
    """
    - 파일 업로드(image_file) 또는 이미지 URL(image_url) 모두 지원
    - 일부 클라이언트가 image_file 키에 URL 문자열을 넣어 보내는 경우도 보정
    """
    # 폼 원본에서 보정: image_url이 없고, image_file 자리에 문자열 URL이 온 경우 잡아줌
    if image_url is None and image_file is None:
        form = await request.form()
        maybe_url = form.get("image_url") or form.get("image_file")
        if isinstance(maybe_url, str) and (maybe_url.startswith("http://") or maybe_url.startswith("https://")):
            image_url = maybe_url

    log.info(
        f"[start_chat] from={request.client.host} "
        f"file={'yes' if image_file else 'no'} url={'yes' if image_url else 'no'} "
        f"q_len={len(user_question)}"
    )

    return await ai_service.start_new_chat_session(
        image_file=image_file,
        user_question=user_question,
        image_url=image_url,
    )

@router.post("/continue_chat")
async def continue_chat_endpoint(
    conversation_id: str = Form(...),
    user_question: str = Form(...)
):
    return await ai_service.continue_existing_chat(conversation_id, user_question)