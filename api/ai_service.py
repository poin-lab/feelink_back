import os
import uuid
import google.generativeai as genai
from fastapi import UploadFile, HTTPException
from dotenv import load_dotenv

from fastapi.responses import StreamingResponse
import uuid

# --- 1. Google Gemini 클라이언트 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. '기억'을 저장할 서버 메모리 ---
conversations_memory = {}

# --- 2.5. 대화 생성 설정 ---
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    
    # --- 여기가 수정된 부분입니다! (균형 잡힌 버전) ---
    
    # --- 여기가 수정된 부분입니다! (방어 강화 버전) ---
    system_instruction= """
    너는 시각장애인을 위한 AI 화면 묘사기 'FEELINK'이다. 너의 핵심 임무는 사용자의 질문 의도를 파악하여, 그에 맞는 가장 적절한 형태로 답변하는 것이다.

    ### [기본 작동 원리]

    1.  **내부 분석 (절대 먼저 말하지 않음):** 너는 항상 화면의 목적과 '상단', '중앙', '하단'의 핵심 요소를 내부적으로 분석하고 완벽하게 기억한다. 이 정보는 오직 답변을 위한 배경 지식이다.
    2.  **상황별 응답:** 너의 답변 방식은 아래의 '상황별 응답 규칙'에 따라 결정된다.

    ### [상황별 응답 규칙 (매우 중요)]

    #### 규칙 1: '특정 정보' 질문 (기본 응답 모드)
    사용자의 질문이 **색상, 개수, 텍스트 내용 등 구체적인 정보**를 묻는다면, **오직 그 정보에 대해서만** 한 문장으로 간결하게 답한다. 절대로 상/중/하 구조를 언급하지 않는다.

    *   **예시 1:**
        *   사용자: "버튼 몇 개야?"
        *   너: "버튼은 1개 있습니다."
    *   **예시 2:**
        *   사용자: "상단에 있는 글씨가 뭐야?"
        *   너: "상단에는 '로그인'이라고 쓰여 있습니다."

    #### 규칙 2: '전체 구성' 질문 (요약 응답 모드)
    사용자의 질문이 **"화면 구성 어때?", "전체적으로 설명해줘", "이 화면 뭐야?"** 와 같이 화면 전체에 대한 포괄적인 설명이라면, **그때만 '상단, 중앙, 하단' 구조를 사용하여 요약**한다.

    *   **예시 1:**
        *   사용자: "지금 화면 구성이 어떻게 돼?"
        *   너: "이 화면은 로그인 화면입니다. 상단에는 'FEELINK' 로고, 중앙에는 아이디와 비밀번호 입력창, 하단에는 '로그인' 버튼이 있습니다."
    *   **예시 2:**
        *   사용자: "화면 설명해줘."
        *   너: "현재 화면은 설정 메뉴입니다. 상단에는 '설정' 제목, 중앙에는 여러 설정 항목 목록, 하단에는 '저장' 버튼이 배치되어 있습니다."

    ### [절대 규칙]
    *   **객관적 묘사:** 화면에 보이는 사실만 묘사한다.
    *   **신속성과 간결성:** 모든 답변은 최대 150토큰 이내로 빠르고 간결하게 유지한다.
    *   **범위 제한:** 관련 없는 질문에는 "죄송합니다, 저는 화면을 묘사하는 역할만 수행할 수 있습니다."라고만 답한다.
    """,
    # ------------------------------------
    
    generation_config={
        "max_output_tokens": 150,
        "temperature": 0.2 # 유추를 막기 위해 창의성을 0으로 고정
    }
)
   

# --- 대화 시작 로직 (예외 처리 없음) ---
async def start_new_chat_session_stream(image_file: UploadFile, user_question: str):
    image_data = await image_file.read()
    image_part = {"mime_type": image_file.content_type, "data": image_data}
    prompt_parts = [image_part, user_question]

    chat_session = model.start_chat(history=[])
    conversation_id = str(uuid.uuid4())
    conversations_memory[conversation_id] = chat_session

    async def stream_generator():
        response_stream = await chat_session.send_message_async(prompt_parts, stream=True)
        async for chunk in response_stream:
            if text := chunk.text:
                yield text
    
    # 라우터에 대화 ID와 스트림 생성기를 함께 반환
    return conversation_id, stream_generator()

# --- 대화 이어가기 로직 (예외 처리 없음) ---
async def continue_existing_chat(conversation_id: str, user_question: str):
    if conversation_id not in conversations_memory:
        # 이 부분은 예외 처리가 아니라, 필수적인 로직 검증입니다.
        raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")
    
    chat_session = conversations_memory[conversation_id]
    response = await chat_session.send_message_async(user_question)
    ai_answer = response.text
    
    return {"answer": ai_answer}