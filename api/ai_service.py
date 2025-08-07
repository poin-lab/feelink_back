import os
import uuid
import google.generativeai as genai
from fastapi import UploadFile, HTTPException
from dotenv import load_dotenv

# --- 1. Google Gemini 클라이언트 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. '기억'을 저장할 서버 메모리 ---
conversations_memory = {}

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
   

# --- 대화 시작 로직 (예외 처리 없음) ---
async def start_new_chat_session(image_file: UploadFile, user_question: str):
    image_data = await image_file.read()
    image_part = {"mime_type": image_file.content_type, "data": image_data}
    prompt_parts = [image_part, user_question]

    chat_session = model.start_chat(history=[])
    response = await chat_session.send_message_async(prompt_parts)
    ai_answer = response.text

    conversation_id = str(uuid.uuid4())
    conversations_memory[conversation_id] = chat_session
    
    return {"conversation_id": conversation_id, "answer": ai_answer}

# --- 대화 이어가기 로직 (예외 처리 없음) ---
async def continue_existing_chat(conversation_id: str, user_question: str):
    if conversation_id not in conversations_memory:
        # 이 부분은 예외 처리가 아니라, 필수적인 로직 검증입니다.
        raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")
    
    chat_session = conversations_memory[conversation_id]
    response = await chat_session.send_message_async(user_question)
    ai_answer = response.text
    
    return {"answer": ai_answer}