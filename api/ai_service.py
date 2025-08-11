import os
import uuid

# 외부 라이브러리
import google.generativeai as genai
from fastapi import UploadFile, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# 내부 모듈 (DB 연동 시 필요)
from api.models import Conversation

# --- 1. Google Gemini 클라이언트 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# --- 2. AI 모델 객체를 저장할 전역 변수 ---
model = None


# --- 3. 애플리케이션 생명주기(Lifespan) 관리 함수 ---
def initialize_ai_model():
    """AI 모델을 초기화하고 전역 model 변수에 할당합니다."""
    global model
    if model is None:
        print("[LIFESPAN] AI 모델 초기화를 시작합니다...")
        try:
            model = genai.GenerativeModel(
                model_name='gemini-2.5-flash-lite',
                system_instruction=
                """
                너는 이미지를 설명하는 AI야
                간단히 이미지에 맞는 종류를 명확히 말해.
                답은 5초안에 나와야해.
                """,
                generation_config={
                    "max_output_tokens": 150,
                    "temperature": 0.6
                }
            )
            print("[LIFESPAN] AI 모델 초기화 완료.")
        except Exception as e:
            print(f"[LIFESPAN ERROR] AI 모델 초기화 중 심각한 오류 발생: {e}")
            raise e

def close_ai_model():
    """앱 종료 시 모델 관련 리소스를 정리합니다."""
    global model
    model = None
    print("[LIFESPAN] AI 모델 리소스를 정리했습니다.")


# --- 4. 이미지 처리 관련 함수 ---
async def _upload_image_to_storage(image_file: UploadFile) -> str:
    file_extension = image_file.filename.split('.')[-1] if image_file.filename else 'png'
    image_id = uuid.uuid4()
    image_url = "url주소"
    print(f"[IMAGE UPLOAD] '{image_file.filename}' -> {image_url}")
    return image_url


# --- 5. AI 상호작용 함수 ---
async def _get_ai_response_and_history(image_part: dict, user_question: str, previous_history: list = None) -> tuple[str, list]:
    if not model:
        print("[AI ERROR] AI 모델이 초기화되지 않았습니다.")
        raise HTTPException(status_code=503, detail="AI 서비스가 현재 사용 불가능합니다.")

    chat_session = model.start_chat(history=previous_history or [])
    prompt_parts = [image_part, user_question] if image_part else [user_question]
    response = await chat_session.send_message_async(prompt_parts)
    ai_answer = response.text

    history_to_save = [
        {"role": msg.role, "parts": [part.text for part in msg.parts]}
        for msg in chat_session.history
    ]
    return ai_answer, history_to_save


# --- 6. 데이터베이스 상호작용 함수 (★★핵심 변경점: async 제거★★) ---
def _save_conversation_to_db(db: Session, log_data: dict):
    """
    [책임] 대화 기록을 데이터베이스에 새로 저장합니다. (동기 함수)
    """
    try:
        db_conversation = Conversation(
            conversation_id=log_data["conversation_id"],
            image_url=log_data["image_url"],
            history=log_data["history"],
            user_id=None
        )
        db.add(db_conversation)
        db.commit()
        db.refresh(db_conversation)
        print(f"[DB SAVE] 대화 {log_data['conversation_id']}가 성공적으로 저장되었습니다.")
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] 대화 저장 중 오류 발생: {e}")

def _update_conversation_in_db(db: Session, conv_id: uuid.UUID, updated_history: list):
    """
    [책임] 기존 대화 기록을 새로운 내용으로 업데이트합니다. (동기 함수)
    """
    try:
        conversation = db.query(Conversation).filter(Conversation.conversation_id == conv_id).first()
        if conversation:
            conversation.history = updated_history
            db.commit()
            print(f"[DB UPDATE] 대화 {conv_id}가 성공적으로 업데이트되었습니다.")
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] 대화 업데이트 중 오류 발생: {e}")


# --- 7. 핵심 서비스 로직 ---
async def start_new_chat_session(
    db: Session,
    background_tasks: BackgroundTasks,
    image_file: UploadFile,
    user_question: str
):
    conversation_id = uuid.uuid4()
    try:
        image_url = await _upload_image_to_storage(image_file)
        image_part = {"mime_type": image_file.content_type, "data": await image_file.read()}
        
        # AI 호출은 await로 기다립니다 (이것이 주된 작업).
        ai_answer, history_to_save = await _get_ai_response_and_history(image_part, user_question)
        
        log_data = {
            "conversation_id": conversation_id,
            "image_url": image_url,
            "history": history_to_save
        }
        
        # DB 저장은 동기 함수이므로, BackgroundTasks가 별도 스레드에서 처리합니다.
        background_tasks.add_task(_save_conversation_to_db, db, log_data)
        
        return {
            "conversation_id": str(conversation_id),
            "answer": ai_answer
        }
    except Exception as e:
        print(f"ERROR in start_new_chat_session: {e}")
        raise HTTPException(status_code=500, detail=f"AI 서비스 처리 중 오류 발생: {str(e)}")

async def continue_existing_chat(
    db: Session,
    background_tasks: BackgroundTasks,
    conversation_id: str,
    user_question: str
):
    try:
        conv_uuid = uuid.UUID(conversation_id)
        conversation = db.query(Conversation).filter(Conversation.conversation_id == conv_uuid).first()
        if not conversation:
            raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")
        
        ai_answer, updated_history = await _get_ai_response_and_history(
            image_part=None,
            user_question=user_question,
            previous_history=conversation.history
        )
        
        background_tasks.add_task(_update_conversation_in_db, db, conv_uuid, updated_history)
        
        return {
            "conversation_id": str(conversation_id),
            "answer": ai_answer
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"ERROR in continue_existing_chat: {e}")
        raise HTTPException(status_code=500, detail=f"AI 서비스 처리 중 오류 발생: {str(e)}")