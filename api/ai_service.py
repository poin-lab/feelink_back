# --- 외부 라이브러리 임포트 ---
import io  # 이미지 데이터를 메모리 상에서 바이너리 스트림으로 다루기 위해 사용
import logging  # 서버 로그를 기록하기 위해 사용
import os  # 환경 변수(.env)를 읽어오기 위해 사용
import uuid  # 고유한 대화 ID를 생성하기 위해 사용
from typing import Optional, Tuple  # 타입 힌팅(코드 명확성)을 위해 사용

import google.generativeai as genai  # Google Gemini AI 모델 사용
import httpx  # 비동기 HTTP 요청을 보내기 위해 사용 (이미지 URL 다운로드용)
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 로드하기 위해 사용
from fastapi import BackgroundTasks, HTTPException, UploadFile  # FastAPI 프레임워크 기능 사용
from PIL import Image  # 이미지 파일을 열고, 검증하고, 다른 포맷으로 변환하기 위해 사용

# --- 내부 모듈 임포트 ---
# 우리가 직접 만든 데이터베이스 서비스 모듈을 가져옵니다.
# 스토리지 서비스는 현재 사용하지 않으므로 임포트에서 제외했습니다.
from api import db_service , storage_service  # 데이터베이스 서비스 모듈을 가져옵니다.
from api import notification_service  # 알림 서비스 모듈을 가져옵니다.

from api.db_service import get_db_session  # 비동기 DB 세션을 관리하는 헬퍼 함수를 가져옵니다.
# --- 기본 설정 ---
# 로거 인스턴스 생성 (uvicorn 서버의 로거를 사용)
log = logging.getLogger("uvicorn")
# .env 파일에 정의된 환경 변수를 로드
load_dotenv()


# --- 1. Google Gemini AI 클라이언트 설정 ---
# .env 파일에서 GOOGLE_API_KEY 값을 읽어옵니다.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# API 키가 없으면 서버 실행을 중단시켜, 설정 오류를 즉시 인지하도록 합니다.
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
# 읽어온 API 키로 Gemini 라이브러리를 초기화합니다.
genai.configure(api_key=GOOGLE_API_KEY)


# --- 2. AI 모델 정의 및 설정 ---
# 사용할 Gemini 모델을 지정합니다. 'flash'는 빠르고 비용 효율적인 모델입니다.
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-001',  # 사용할 모델 이름
    # [!! 시스템 프롬프트 !!] AI의 역할과 정체성을 정의하는 매우 중요한 부분입니다.
    system_instruction="""
너는 시각장애인을 위한 AI 화면 해설사 'FEELINK'이다. 너의 출력은 **오직 음성(TTS)으로만 변환**된다는 사실을 명심하고, 모든 답변을 **자연스러운 대화체 문장**으로 생성해야 한다.

---
### [출력의 절대 원칙: 간결함 (Brevity First)]
**가장 중요한 규칙이다.** 너의 모든 답변은 음성으로 변환되므로, 듣는 사람이 지루하지 않도록 **반드시 간결해야 한다.**
*   **길이 목표:** 모든 최종 답변은 **150자 내외**로 요약하는 것을 목표로 한다.
*   **핵심 우선:** 장황한 설명을 피하고, 가장 중요한 정보부터 전달한다. (참고: 아래 [핵심 개체 식별 의무]에 따라 유명 인물을 식별하는 것은 '가장 중요한 정보'에 해당한다.)

---
### [메타 규칙: 생각과 말의 분리]
너의 모든 분석 과정, 예를 들어 '화면을 분석한 결과 콘텐츠 중심 화면으로 판단했습니다'와 같은 **내부적인 생각의 과정은 절대로 출력해서는 안 된다.** 사용자는 오직 **최종 분석 결과**만을 자연스러운 문장으로 듣기 원한다.

---
### [내부 판단 로직: 3단계 분석 및 콘텐츠 절대 우선주의 (사용자에게 절대 언급 금지)]
너의 판단 과정은 다음 단계를 순서대로 따른다. 이 모든 과정은 절대 외부에 노출되어서는 안 된다.

**[1단계: 뷰어(Viewer) 모드 식별 - 최우선 규칙]**
*   가장 먼저 화면이 카메라나 갤러리처럼 **중심 콘텐츠를 '보는' 것에만 집중**하고, 주변 UI(셔터, 편집 도구 등)가 보조적인 **'뷰어 모드'**인지 판단한다.
*   만약 '뷰어 모드'로 판단되면, **무조건 [B. 콘텐츠 중심 화면]으로 분류**하고, 응답 시 **주변의 보조 UI는 완전히 무시**한다.

**[2단계: 화면의 기본 유형 분석 (콘텐츠 우선 원칙)]**
1단계의 '뷰어 모드'에 해당하지 않을 경우에만 다음을 따른다.
*   **복합 화면:** 화면에 **상호작용 가능한 UI 요소와 명확한 중심 콘텐츠가 함께 존재**한다면(예: 쇼핑 앱 홈, 뉴스 앱 목록), 이는 **[B. 콘텐츠 중심 화면]으로 분류**한다.
*   **순수 유형 판별:**
    *   **[B. 콘텐츠 중심 화면]:** 화면의 거의 전체가 하나의 사진이나 그림으로만 이루어진 경우.
    *   **[A. UI 화면]:** 의미 있는 중심 콘텐츠 없이, UI 요소로만 구성된 경우.

**[3단계: 사용자 질문에 따른 최종 응답 모드 결정]**
분석 결과를 바탕으로, 사용자의 질문 의도를 최우선으로 고려하여 최종 응답 방식을 결정한다.
*   **구체적인 질문:** 질문이 UI 요소에 관한 것이면 [모드 A] 스타일로, 콘텐츠에 관한 것이면 [모드 B] 스타일로 해당 부분만 설명한다.
*   **일반적인 질문:** 1, 2단계에서 분석한 화면의 기본 유형(A 또는 B)에 해당하는 모드를 사용하여 전체 화면을 설명한다.

---
### [핵심 원칙: 텍스트 처리 - "읽지 말고, 요약하라"]
화면에 텍스트가 있을 경우, 단순히 글자를 그대로 읽지 말고 그 **의미와 목적을 파악하여 요약**해야 한다.
*   **1. 짧은 기능성 텍스트 (버튼, 라벨 등):** 명확하게 읽어준다.
*   **2. 중간 길이 텍스트 (제목, 광고 문구 등):** 무엇에 대한 내용인지 설명한다.
*   **3. 긴 텍스트 (본문, 상세 설명 등):** **절대 그대로 읽지 않고**, "무엇에 대한 글인지" 한 문장으로 요약한다.

---
### [모드 A: UI 화면 브리핑]
화면이 **순수한 UI 중심**이라고 판단되면, 화면의 전체적인 구조와 주요 기능들을 위에서부터 아래로 간결하게 설명한다. 모든 요소는 동일한 비중으로 간략하게 언급한다.
*   **(예시):** "와이파이 설정 화면입니다. 상단에는 '연결' 스위치가 있고, 중앙에는 사용 가능한 와이파이 목록이 보입니다."

---
### [모드 B: 콘텐츠 중심 화면 묘사]
화면이 **콘텐츠 중심(뷰어 모드, 복합 화면, 순수 콘텐츠)**이라고 판단되면, 다음 지침에 따라 설명한다.

*   **[묘사의 절대 원칙]**
    *   **1. 시각적 사실 기반:** 오직 화면에 **명확한 시각적 근거가 있는 것만 묘사**한다. 절대 추측하거나 상상해서는 안 된다.
    *   **2. 마크다운 금지:** 모든 답변은 완전한 문장 형태여야 한다.

*   **[핵심 개체 식별 의무 (Entity Identification Mandate) - 최우선 묘사 규칙]**
    *   **너의 설명에서 가장 가치 있는 정보는 핵심 인물이나 사물을 식별하는 것이다.** 이는 단순한 묘사를 넘어 사용자에게 실질적인 가치를 제공하는 핵심 임무다.
    *   따라서, 사진에 유명인, 특정 브랜드, 유명 건축물로 **보이는 대상이 있다면, 반드시 식별을 시도해야 한다.**
    *   **판단 근거:** 명확한 시각적 단서(얼굴, 유니폼, 로고 등)를 기반으로 한다.
    *   **표현 방식:** **"~인 것 같습니다"** 또는 **"~으로 보입니다"** 와 같이 조심스럽게 추론한다.
    *   **실패 처리:** 근거가 부족하면 **"누구인지는 명확하지 않네요"** 라고 명확히 한계를 인정한다.

*   **[복합 화면 묘사 공식: 강약 조절]**
    만약 화면이 **UI와 이미지가 섞인 '복합 화면'**이라면, 다음 공식을 따른다.
    *   **1. 전체 맥락 제시:** 화면의 목적이나 앱의 종류를 한 문장으로 먼저 언급한다.
    *   **2. 핵심 콘텐츠 압축 묘사:** 화면 중심 이미지의 **가장 핵심적인 시각적 디테일을 중심으로 간결하지만 생생하게 묘사**한다. (주의: 여기서도 [출력의 절대 원칙: 간결함]을 반드시 지켜야 한다.)
    *   **3. 주변 UI 간결 요약:** 주변 UI는 '무엇이 있다'는 정보만 간결하게 언급한다.

*   **[질문 유도]**
    *   모든 묘사의 마지막에는 "더 궁금한 점이 있으신가요?" 라고 질문을 덧붙인다.

*   **(예시 - 복합 화면):**
    "쇼핑 앱 메인 화면으로 보입니다. 중앙에는 **해변 모래사장 위에 놓인 갈색 샌들 광고 이미지**가 크게 있네요. 상단에는 검색창이, 하단에는 홈, 카테고리 탭이 있습니다. 더 궁금한 점이 있으신가요?"

*   **(예시 - 뷰어 모드 / 순수 콘텐츠):**
    "축구 경기장이네요. 흰색 유니폼의 선수가 드리블을 치고있고, 붉은 유니폼의 선수 3명이 막고 있어요. **흰색 유니폼은 손흥민 선수인 것 같습니다.** 더 궁금한 점이 있으신가요?"
""",
# temperature는 약간 높여서 묘사의 창의성을 허용하는 것도 좋음

    generation_config={
        "max_output_tokens": 150,  # 답변의 최대 길이를 150 토큰으로 제한 (간결성 유지)
        "temperature": 0.3,  # 생성의 무작위성. 낮을수록 AI가 더 사실에 기반하고 일관된 답변을 생성합니다.
    },
)


# =====================================================================================
# 유틸리티 함수 섹션: 이미지 처리 등 보조적인 작업을 수행하는 함수들
# =====================================================================================

async def _fetch_image_from_url(image_url: str) -> Tuple[bytes, str]:
    """주어진 URL에서 이미지를 비동기적으로 다운로드합니다."""
    try:
        # httpx 클라이언트를 사용하여 비동기로 URL에 GET 요청을 보냅니다.
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url, follow_redirects=True)
            resp.raise_for_status()  # HTTP 오류 (4xx, 5xx)가 발생하면 예외를 일으킵니다.
            # 응답 헤더에서 'Content-Type' (e.g., 'image/jpeg')을 추출합니다.
            content_type = resp.headers.get("content-type", "image/png").split(";")[0].strip().lower()
            return resp.content, content_type
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"이미지 URL 다운로드 실패: {e}")

def _reencode_image_to_png(img_bytes: bytes) -> bytes:
    """
    이미지 데이터를 PNG 포맷으로 변환(정규화)합니다.
    - 이유: 다양한 이미지 포맷(JPG, GIF 등)을 일관된 PNG로 통일하여 처리 안정성을 높입니다.
    - 부가 효과: 유효하지 않은 이미지 데이터일 경우 여기서 오류가 발생하여 사전 검증이 가능합니다.
    """
    try:
        # Pillow 라이브러리로 메모리상의 이미지 바이트를 엽니다.
        with Image.open(io.BytesIO(img_bytes)) as im:
            buf = io.BytesIO()  # 새로운 메모리 버퍼를 만듭니다.
            im.save(buf, format="PNG")  # 버퍼에 이미지를 PNG 형식으로 저장합니다.
            return buf.getvalue()  # 버퍼의 전체 바이트 데이터를 반환합니다.
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 이미지 데이터: {e}")

def _make_image_part(img_bytes: bytes) -> dict:
    """Gemini API가 요구하는 멀티모달(이미지+텍스트) 입력 형식에 맞게 이미지 데이터를 포장합니다."""
    return {"mime_type": "image/png", "data": img_bytes}

async def _fetch_image_from_url(url: str) -> bytes:
    """주어진 URL에서 이미지 데이터를 비동기적으로 다운로드합니다."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status() # 200 OK가 아니면 예외 발생
            return response.content
        except httpx.RequestError as e:
            log.error(f"URL에서 이미지 다운로드 실패: {url} - {e}")
            raise HTTPException(status_code=400, detail="이미지 URL에 접근할 수 없습니다.")



async def start_new_chat_session(
    background_tasks: BackgroundTasks,
    user_question: str,
    image_file: UploadFile,
):
    """
    [안정 버전] 이미지를 먼저 업로드 한 후 Gemini 분석을 하고, DB 저장은 백그라운드로 처리합니다.
    """
    if not image_file:
        raise HTTPException(status_code=400, detail="이미지 파일을 반드시 제공해야 합니다.")

    # 1. 이미지 처리 및 스토리지에 업로드 (사용자 응답 전에 먼저 실행)
    try:
        image_bytes = await image_file.read()
        
        # storage_service를 직접 호출하여 업로드가 끝날 때까지 기다립니다.
        generated_image_url = await storage_service.upload_image_and_get_url(
            file_bytes=image_bytes,
            original_filename=image_file.filename
        )
    except Exception as e:
        log.error(f"이미지 처리 및 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")
        
    # 2. Gemini API 호출
    try:
        image_part = _make_image_part(image_bytes)
        chat_session = model.start_chat(history=[])
        response = await chat_session.send_message_async([image_part, user_question])
        ai_answer = response.text
    except Exception as e:
        log.error(f"Gemini API 호출 실패: {e}")
        raise HTTPException(status_code=500, detail="AI 응답 생성 중 오류가 발생했습니다.")

    # 3. 응답 데이터 및 백그라운드 작업 준비
    conversation_id = uuid.uuid4()
    initial_history = [
        {"role": "user", "parts": [user_question]},
        {"role": "model", "parts": [ai_answer]},
    ]
    
    # 4. DB 저장 작업만 백그라운드로 보냅니다.
    # 이 방식은 복잡한 세션 주입이 필요 없습니다.
    background_tasks.add_task(
        db_service.create_new_conversation,
        conversation_id=conversation_id,
        image_url=generated_image_url, 
        initial_history=initial_history
    )

    log.info(f"[AI] /start_chat 응답 완료. cid={conversation_id} DB 작업 백그라운드 실행")
    
    # 5. 사용자에게 최종 응답 반환
    return {"conversation_id": str(conversation_id), "answer": ai_answer}


async def continue_chat_endpoint(
    background_tasks: BackgroundTasks,
    conversation_id: str,
    user_question: str
):
    """기존 대화를 이어가는 핵심 함수."""
    try:
        # 문자열로 받은 대화 ID를 UUID 객체로 변환합니다. 형식이 틀리면 에러가 발생합니다.
        convo_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 형식의 대화 ID입니다.")

    # 1. 데이터베이스에서 이전 대화 기록 조회
    # db_service를 통해 해당 ID의 대화 정보를 가져옵니다.
    conversation = await db_service.get_conversation_history(convo_uuid)
    if not conversation:
        # 대화 기록이 없으면 404 Not Found 오류를 반환합니다.
        raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")
    
    
    try:
        image_bytes = await _fetch_image_from_url(conversation.image_url)
    except HTTPException as e:
        # 다운로드 실패 시, 받은 예외를 그대로 다시 발생시킵니다.
        raise e
        
    # 다운로드한 이미지 바이트로 Gemini가 이해할 수 있는 이미지 파트를 만듭니다.
    image_part = {
        "mime_type": "image/png", # 또는 저장된 타입에 맞게
        "data": image_bytes      # 'uri' 대신 'data' 키를 사용합니다.
    }
    # 2. Gemini API 호출 (이전 대화 맥락 포함)
    try:
        # [!!핵심!!] DB에서 가져온 'history'를 사용해 AI 모델의 대화 세션을 복원합니다.
        # 이렇게 하면 AI가 이전 대화 내용을 기억하고 답변을 생성합니다.
        # ★★ 이때, 이미지는 다시 보내지 않고 오직 텍스트 기록만 사용합니다. ★★
        chat_session = model.start_chat(history=conversation.history)
        response = await chat_session.send_message_async([image_part, user_question])
        ai_answer = response.text
    except Exception as e:
        log.error(f"Gemini API 호출 실패 (continue): {e}")
        raise HTTPException(status_code=500, detail=f"AI 응답 생성 중 오류: {e}")

    # 3. 백그라운드 작업 등록
    # DB에 업데이트할 새로운 대화 턴(사용자 질문 + AI 답변)을 준비합니다.
    new_turn = [
        {"role": "user", "parts": [user_question]},
        {"role": "model", "parts": [ai_answer]},
    ]
    # 이전 대화 기록에 이 새로운 턴을 추가하도록 DB 업데이트 작업을 백그라운드로 예약합니다.
    background_tasks.add_task(
        db_service.update_conversation_history, # 실행할 함수
        convo_uuid,                             # 함수에 전달할 인자 1
        new_turn                                # 함수에 전달할 인자 2
    )

    log.info(f"[AI] /continue_chat 응답 완료. cid={conversation_id} DB 업데이트 백그라운드 실행")

    # 4. 사용자에게 최종 응답 즉시 반환
    return {"conversation_id": conversation_id, "answer": ai_answer}




async def start_test(
    background_tasks: BackgroundTasks,
    user_question: str,
    image_file: Optional[UploadFile] = None,
    image_url: Optional[str] = None,
):
    """
    [안정 버전] 이미지를 먼저 업로드 한 후 Gemini 분석을 하고, DB 저장은 백그라운드로 처리합니다.
    """
    if not image_file:
        raise HTTPException(status_code=400, detail="이미지 파일을 반드시 제공해야 합니다.")

    # 1. 이미지 처리 및 스토리지에 업로드 (사용자 응답 전에 먼저 실행)
    try:
        image_bytes = await image_file.read()
        
        # storage_service를 직접 호출하여 업로드가 끝날 때까지 기다립니다.
        generated_image_url = await storage_service.upload_image_and_get_url(
            file_bytes=image_bytes,
            original_filename=image_file.filename
        )
    except Exception as e:
        log.error(f"이미지 처리 및 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail="이미지 처리 중 오류가 발생했습니다.")
        
    # 2. Gemini API 호출
    try:
        image_part = _make_image_part(image_bytes)
        chat_session = model.start_chat(history=[])
        response = await chat_session.send_message_async([image_part, user_question])
        ai_answer = response.text
    except Exception as e:
        log.error(f"Gemini API 호출 실패: {e}")
        raise HTTPException(status_code=500, detail="AI 응답 생성 중 오류가 발생했습니다.")

    # 3. 응답 데이터 및 백그라운드 작업 준비
    conversation_id = uuid.uuid4()
    initial_history = [
        {"role": "user", "parts": [user_question]},
        {"role": "model", "parts": [ai_answer]},
    ]
    
    # 4. DB 저장 작업만 백그라운드로 보냅니다.
    # 이 방식은 복잡한 세션 주입이 필요 없습니다.
    background_tasks.add_task(
        db_service.create_new_conversation,
        conversation_id=conversation_id,
        image_url=generated_image_url, 
        initial_history=initial_history
    )

    background_tasks.add_task(
        notification_service.send_notification_as_single_message,
        message=ai_answer,  # 함수에 전달할 인자
        ms_conversation_id=conversation_id  # 추가 인자
    )

    log.info(f"[AI] /start_test 응답 완료. cid={conversation_id} DB 작업 백그라운드 실행")
    

   
    # 5. 사용자에게 최종 응답 즉시 반환
    # 백그라운드 작업이 시작되었는지 여부와 상관없이, AI 답변은 바로 클라이언트에게 전달됩니다.
    return {"conversation_id": str(conversation_id)}




async def continue_test(
    background_tasks: BackgroundTasks,
    conversation_id: str,
    user_question: str
):
    """기존 대화를 이어가는 핵심 함수."""
    try:
        # 문자열로 받은 대화 ID를 UUID 객체로 변환합니다. 형식이 틀리면 에러가 발생합니다.
        convo_uuid = uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 형식의 대화 ID입니다.")

    # 1. 데이터베이스에서 이전 대화 기록 조회
    # db_service를 통해 해당 ID의 대화 정보를 가져옵니다.
    conversation = await db_service.get_conversation_history(convo_uuid)
    if not conversation:
        # 대화 기록이 없으면 404 Not Found 오류를 반환합니다.
        raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")
    
    
    try:
        image_bytes = await _fetch_image_from_url(conversation.image_url)
    except HTTPException as e:
        # 다운로드 실패 시, 받은 예외를 그대로 다시 발생시킵니다.
        raise e
        
    # 다운로드한 이미지 바이트로 Gemini가 이해할 수 있는 이미지 파트를 만듭니다.
    image_part = {
        "mime_type": "image/png", # 또는 저장된 타입에 맞게
        "data": image_bytes      # 'uri' 대신 'data' 키를 사용합니다.
    }
    # 2. Gemini API 호출 (이전 대화 맥락 포함)
    try:
        # [!!핵심!!] DB에서 가져온 'history'를 사용해 AI 모델의 대화 세션을 복원합니다.
        # 이렇게 하면 AI가 이전 대화 내용을 기억하고 답변을 생성합니다.
        # ★★ 이때, 이미지는 다시 보내지 않고 오직 텍스트 기록만 사용합니다. ★★
        chat_session = model.start_chat(history=conversation.history)
        response = await chat_session.send_message_async([image_part, user_question])
        ai_answer = response.text
    except Exception as e:
        log.error(f"Gemini API 호출 실패 (continue): {e}")
        raise HTTPException(status_code=500, detail=f"AI 응답 생성 중 오류: {e}")

    # 3. 백그라운드 작업 등록
    # DB에 업데이트할 새로운 대화 턴(사용자 질문 + AI 답변)을 준비합니다.
    new_turn = [
        {"role": "user", "parts": [user_question]},
        {"role": "model", "parts": [ai_answer]},
    ]
    # 이전 대화 기록에 이 새로운 턴을 추가하도록 DB 업데이트 작업을 백그라운드로 예약합니다.
    background_tasks.add_task(
        db_service.update_conversation_history, # 실행할 함수
        convo_uuid,                             # 함수에 전달할 인자 1
        new_turn                                # 함수에 전달할 인자 2
    )

    background_tasks.add_task(
        notification_service.send_notification_as_single_message,
        message=ai_answer,  # 함수에 전달할 인자
        ms_conversation_id=conversation_id  # 추가 인자
    )
    log.info(f"[AI] /continue_test 응답 완료. cid={conversation_id} DB 업데이트 백그라운드 실행")

    # 4. 사용자에게 최종 응답 즉시 반환 
    return {"conversation_id": conversation_id}