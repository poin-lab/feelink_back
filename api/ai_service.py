import os
import uuid

# 외부 라이브러리
import google.generativeai as genai
from fastapi import UploadFile, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session
from dotenv import load_dotenv


# 내부 모듈 (DB 연동 시 필요)


# --- 1. Google Gemini 클라이언트 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)



# --- 2.5. 대화 생성 설정 ---
model = genai.GenerativeModel(
    # --- 모델 이름 설정 ---
    model_name='gemini-2.5-flash-lite',
    
    # --- 여기가 수정된 부분입니다! (균형 잡힌 버전) ---
    system_instruction= 
    """
    너는 시각장애인을 위한 AI 화면 묘사기 'FEELINK'이다. 너의 핵심 임무는 사용자가 화면에 대해 질문하기 전에, 먼저 화면 전체를 **내부적으로 분석하여 답변을 준비**하는 것이다. 너의 가장 중요한 원칙은 **'질문받은 내용에만 답변'** 하는 것이며, 분석 내용을 절대 먼저 말하거나 질문과 무관한 정보를 덧붙이지 않는다.

    ### [작동 방식]

    1.  **내부 분석 (절대 출력하지 않음):**
        *   화면의 전체 목적을 파악한다.
        *   화면을 '상단', '중앙', '하단'으로 나누어 각각의 핵심 시각 요소, 텍스트, 그리고 포함된 그림/사진의 존재를 정리한다. 이 정보는 오직 답변을 위한 배경 데이터로만 사용한다.

    2.  **질의응답:**
    *   사용자의 질문이 들어오면, 1단계의 분석 내용 중에서 **정확히 질문에 해당하는 정보만**을 찾아 1-2 문장으로 답한다.
    *   **(그림 묘사)** 그림에 대한 질문에는 해당 그림의 시각적 요소만 묘사한다.
    *   질문이 없으면 먼저 말하지 않고 대기한다.

    ### [절대 규칙]

    *   **질문 범위 절대 준수:** 질문받지 않은 내용은 절대 먼저 말하지 않는다. **예를 들어, '버튼이 몇 개야?'라는 질문에는 버튼의 '개수'만 답하고, 그 버튼들의 기능이나 위치는 사용자가 묻기 전까지 절대 설명하지 않는다.**
    *   **분석 내용 비공개:** 어떤 경우에도 '상단', '중앙', '하단'으로 나눈 분석 내용을 사용자에게 먼저 설명하지 않는다.
    *   **신속성과 간결성:** 모든 답변은 최대 150토큰을 넘지 않도록 극도로 간결하게 유지한다.
    *   **객관적 사실:** 화면에 명확히 보이는 정보(텍스트, 아이콘, 그림의 내용 등) 외에 추측이나 의견을 포함하지 않는다.
    *   **범위 제한:** 화면 묘사와 관련 없는 질문에는 "죄송합니다, 저는 화면을 묘사하는 역할만 수행할 수 있습니다."라고만 답한다.
    """,
    # ------------------------------------
    
    generation_config={
        "max_output_tokens": 150,
        "temperature": 0.2 # 유추를 막기 위해 창의성을 0으로 고정
    }
)
   


# --- 3. 이미지 처리 관련 함수 --- 
async def _upload_image_to_storage(image_file: UploadFile) -> str: 
    #   나중에 이 함수 내부를 실제 Azure Blob Storage 업로드 코드로 교체해야 합니다.
    #   파일 이름에서 확장자를 추출하고, UUID를 생성하여 이미지 URL을 만듭니다.
    file_extension = image_file.filename.split('.')[-1] if image_file.filename else 'png' 
    
    image_id = uuid.uuid4() # 이미지 ID 생성
    image_url = "url주소" #이미지 url 생성
    
    # 개발 중 확인을 위한 로그
    print(f"[IMAGE UPLOAD] '{image_file.filename}' -> {image_url}")
    return image_url




# --- 4. AI 상호작용 함수 ---

async def _get_ai_response_and_history(image_part: dict, user_question: str, previous_history: list = None) -> tuple[str, list]:
    """
    [책임] Gemini AI 모델에 요청을 보내고, 답변과 전체 대화 기록을 받아옵니다.
    [목적] AI와의 모든 통신을 이 함수 안에서만 처리하여 로직을 중앙 집중화합니다.

    Args:
        image_part (dict): 분석할 이미지 데이터. 첫 질문에만 사용됩니다.
        user_question (str): 사용자의 질문.
        previous_history (list, optional): 이전 대화 기록. 대화 이어가기 시 사용됩니다.

    Returns:
        tuple: (AI의 답변 텍스트, DB에 저장하기 좋은 형태로 변환된 전체 대화 기록)
    """
    if not model:
        raise HTTPException(status_code=503, detail="AI 서비스가 현재 사용 불가능합니다.")

    chat_session = model.start_chat(history=previous_history or [])
    prompt_parts = [image_part, user_question] if image_part else [user_question]
    
    response = await chat_session.send_message_async(prompt_parts)
    ai_answer = response.text

    # Gemini 라이브러리의 history 객체는 직접 저장하기 복잡하므로,
    # DB에 저장하기 쉬운 JSON(딕셔너리 리스트) 형태로 변환합니다.
    history_to_save = [
        {"role": msg.role, "parts": [part.text for part in msg.parts]}
        for msg in chat_session.history
    ]

    return ai_answer, history_to_save


# --- 5. 데이터베이스 상호작용 함수 ---

async def _save_conversation_to_db(db: Session, log_data: dict):
    """
    [책임] 대화 기록을 데이터베이스에 새로 저장합니다.
    [실행] 비동기(백그라운드)로 실행되어 사용자 응답 시간에 영향을 주지 않습니다.
    """
    try:
        db_conversation = Conversation(
            conversation_id=log_data["conversation_id"],
            image_url=log_data["image_url"],
            history=log_data["history"],
            user_id=None # 로그인 기능 구현 전까지는 NULL
        )
        db.add(db_conversation)
        db.commit()
        db.refresh(db_conversation)
        print(f"[DB SAVE] 대화 {log_data['conversation_id']}가 성공적으로 저장되었습니다.")
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] 대화 저장 중 오류 발생: {e}")

#   대화 기록 업데이트
async def _update_conversation_in_db(db: Session, conv_id: uuid.UUID, updated_history: list):
    """
    [책임] 기존 대화 기록을 새로운 내용으로 업데이트합니다.
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



# --- 6. 핵심 서비스 로직 (API 라우터가 직접 호출하는 메인 함수들) ---

async def start_new_chat_session(
    db: Session, 
    background_tasks: BackgroundTasks,
    image_file: UploadFile, 
    user_question: str
):
    """
    [서비스] 새로운 대화를 시작하는 전체 과정을 조율(Orchestrate)합니다.
    """
    # 1. 고유한 대화 ID를 미리 생성합니다.
    conversation_id = uuid.uuid4()
    
    try:
        # 2. 이미지 처리
        image_url = await _upload_image_to_storage(image_file)
        image_part = {"mime_type": image_file.content_type, "data": await image_file.read()}

        # 3. AI 호출하여 답변과 대화 기록 얻기
        ai_answer, history_to_save = await _get_ai_response_and_history(image_part, user_question)

        # 4. DB에 저장할 데이터 묶음(로그) 생성
        log_data = {
            "conversation_id": conversation_id,
            "image_url": image_url,
            "history": history_to_save
        }

        # 5. DB 저장 작업을 '백그라운드'로 예약합니다.
        #    사용자는 이 작업이 끝날 때까지 기다리지 않습니다.
        background_tasks.add_task(_save_conversation_to_db, db, log_data)

        # 6. 사용자에게 즉시 반환할 최종 응답을 만듭니다.
        return {
            "conversation_id": str(conversation_id),
            "answer": ai_answer
        }

    except Exception as e:
        # 모든 예외를 처리하여 서버가 다운되는 것을 방지합니다.
        print(f"ERROR in start_new_chat_session: {e}")
        raise HTTPException(status_code=500, detail=f"AI 서비스 처리 중 오류 발생: {str(e)}")


async def continue_existing_chat(
    db: Session,
    background_tasks: BackgroundTasks,
    conversation_id: str, 
    user_question: str
):
    """
    [서비스] 기존 대화를 이어가는 과정을 조율합니다.
    """
    try:
        # 1. 문자열 ID를 UUID 객체로 변환하여 DB에서 조회
        conv_uuid = uuid.UUID(conversation_id)
        conversation = db.query(Conversation).filter(Conversation.conversation_id == conv_uuid).first()
        
        # 2. 대화 기록이 없으면 404 에러 발생
        if not conversation:
            raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")

        # 3. AI 호출 (이전 기록을 전달하고, 이미지는 보내지 않음)
        ai_answer, updated_history = await _get_ai_response_and_history(
            image_part=None, 
            user_question=user_question, 
            previous_history=conversation.history
        )

        # 4. DB 업데이트 작업을 백그라운드로 예약합니다.
        background_tasks.add_task(_update_conversation_in_db, db, conv_uuid, updated_history)

        # 5. 사용자에게 최종 응답을 만듭니다.
        return {
            "conversation_id": str(conversation_id),
            "answer": ai_answer
        }

    except HTTPException as e:
        # 이미 처리된 HTTP 예외는 그대로 다시 발생시킵니다.
        raise e
    except Exception as e:
        # 그 외의 모든 예외를 처리합니다.
        print(f"ERROR in continue_existing_chat: {e}")
        raise HTTPException(status_code=500, detail=f"AI 서비스 처리 중 오류 발생: {str(e)}")