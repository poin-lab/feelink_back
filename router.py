# router.py
from fastapi import APIRouter, UploadFile, File, Form
# 'api' 폴더 안에 있는 'ai_service' 모듈을 임포트합니다.
from api import ai_service

router = APIRouter()

@router.post("/start_chat")
async def start_chat_endpoint(
    image_file: UploadFile = File(...),
    user_question: str = Form(...)
):
    return await ai_service.start_new_chat_session(image_file, user_question)

@router.post("/continue_chat")
async def continue_chat_endpoint(
    conversation_id: str = Form(...),
    user_question: str = Form(...)
):
    return await ai_service.continue_existing_chat(conversation_id, user_question)