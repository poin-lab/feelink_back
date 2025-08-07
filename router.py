# router.py

from fastapi import APIRouter, UploadFile, File, Form, Response, HTTPException
from fastapi.responses import StreamingResponse
from api import ai_service # 수정된 ai_service 모듈 임포트

router = APIRouter()

# --- 스트리밍을 지원하는 엔드포인트 ---
@router.post("/start_chat_stream")
async def start_chat_stream_endpoint(
    response: Response, # FastAPI가 헤더 설정을 위해 response 객체를 주입
    image_file: UploadFile = File(...),
    user_question: str = Form(...)
):
    # 1. 서비스 로직 호출
    convo_id, stream_gen = await ai_service.start_new_chat_session_stream(image_file, user_question)

    # 2. 응답 헤더에 대화 ID 설정
    response.headers["X-Conversation-ID"] = convo_id

    # 3. 스트리밍 응답 반환
    return StreamingResponse(stream_gen, media_type="text/plain; charset=utf-8")


@router.post("/continue_chat_stream")
async def continue_chat_stream_endpoint(
    conversation_id: str = Form(...),
    user_question: str = Form(...)
):
    # 1. 서비스 로직 호출
    convo_id, stream_gen = await ai_service.continue_existing_chat_stream(conversation_id, user_question)

    # 2. 대화 ID 유효성 검사
    if convo_id is None:
        raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")

    # 3. 스트리밍 응답 반환 (여기서는 헤더 설정 불필요)
    return StreamingResponse(stream_gen, media_type="text/plain; charset=utf-8")