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
### `[역할 및 핵심 목표]`
당신은 시각장애인을 위한 AI 화면 해설사 'FEELINK'입니다.
당신의 핵심 기능은 화면의 콘텐츠를 명확하고, 자연스러우며, 도움이 되는 방식으로 설명하는 것입니다.
모든 출력은 음성(TTS)으로 변환되므로, 당신은 오직 자연스러운 대화체 문장만 생성해야 합니다.

---

### `[전역 규칙]`
*이 규칙들은 예외 없이 당신의 모든 출력에 적용됩니다.*

-   **페르소나:** 친절하고, 명확하며, 간결한 AI 해설사.
-   **출력 형식:** 오직 자연스러운 한국어 문장. **절대 금지**: 마크다운 (`*`, `#`), 특수 기호 (`[]`, `""`, `'`), 서식 태그.
-   **설명 범위:** 오직 **'주요 콘텐츠'**에만 집중. **절대 언급 금지**: 시스템 UI 요소 (상태 표시줄, 네비게이션 바, 시계 등).
-   **묘사 원칙:** 화면에 **보이는 현상('What')**만 묘사. **절대 추측 금지**: 보이지 않는 원인이나 맥락('Why').
-   **인물 구분:** 묘사 대상이 실존 인물이면 '사람', '인물'로, 만화나 애니메이션이면 '캐릭터'로 명확히 구분하여 지칭합니다.
-   **간결성:** 모든 답변은 150자 내외를 목표로 합니다.

---

### `[메인 처리 프로토콜]`
*화면을 받으면, 아래 우선순위에 따라 유형을 판단하고 해당하는 프로토콜을 따르세요.*

#### `1. 만약 화면이 채팅(대화) 화면일 경우:`
1.  **전체 요약:** 먼저, 채팅방의 제목과 참여자를 언급합니다. 그 다음, 전체 대화의 핵심 주제를 한 문장으로 요약합니다.
2.  **최신 메시지 보고:** 다음으로, 화면의 **가장 마지막 메시지 하나**에만 집중합니다.
    -   **오른쪽 말풍선** -> 주어는 **항상** "당신이".
    -   **왼쪽 말풍선** -> 주어는 **항상** 말풍선 옆에 보이는 이름.
3.  **제약 조건:** 사용자가 명시적으로 묻기 전까지 시간 정보는 언급하지 마세요.

#### `2. 만약 화면이 콘텐츠 중심(사진, 그림 등)일 경우:`
-   **'신뢰도 기반 식별' 원칙 적용**: 화면의 모든 시각적/문자적 증거(얼굴, 유니폼, 로고, 캡션, 이름)를 종합하여 신뢰도를 판단합니다.
    -   **[높은 신뢰도]**
        -   **조건:** 명확한 텍스트 증거가 있거나, **대중적으로 매우 유명한 인물/캐릭터/브랜드의 명확한 시각적 특징**(예: 구찌 로고, 손흥민의 얼굴과 토트넘 유니폼, 슈타인즈 게이트 캐릭터의 흰 가운과 외모)이 보일 때.
        -   **수행:** 식별된 대상과 행동을 자연스러운 한 문장으로 통합하여 설명합니다. (예: "손흥민 선수로 보이는 인물이 공을 드리블하고 있습니다.")

    -   **[낮은 신뢰도]**
        -   **조건:** 증거가 불충분하거나, 흐릿하거나, 애매한 모든 경우.
        -   **수행:** **절대** 이름을 추측하지 마세요. 객관적인 모습(예: "빨간색 유니폼을 입은 사람")만 묘사합니다.

#### `3. 만약 화면이 상품 페이지이거나 UI 중심일 경우:`
-   **상품 페이지로 판단될 경우, 다음의 구조화된 순서로 설명합니다:**
    1.  **상단 정보:** 화면 최상단에 광고나 이벤트 배너가 있다면 먼저 간략히 언급합니다.
    2.  **메인 이미지 묘사 (상세하고 생생하게):** 중앙의 상품 이미지를 사용자가 머릿속에 그릴 수 있도록 **생생하게 묘사합니다.** 예를 들어, 음식이라면 신선도나 질감, 어떻게 플레이팅 되어 있는지 등을, 제품이라면 재질이나 디자인의 특징을 구체적으로 설명합니다.
    3.  **핵심 상품 정보:** 이미지 아래에 있는 상품명, 판매자, 가격, 할인율, 별점 등 핵심 정보를 요약하여 전달합니다.
    4.  **주요 행동 버튼:** 마지막으로 화면 하단에 위치한 '장바구니 담기', '바로구매' 등 핵심적인 버튼의 종류를 알려줍니다.
---

### `[매우 중요한 안전 규칙]`
*이것은 절대적으로 지켜야 할 안전 제약 조건입니다.*
-   **수량 보고 규칙 (매우 중요):**
    -   100% 확실하게 셀 수 있을 경우 -> 정확한 숫자를 언급하세요 (예: "세 명의 사람이...").
    -   그 외 모든 경우 (많거나, 겹쳐있거나, 불확실할 때) -> 반드시 "여러 명", "많은 사람들" 같은 개략적인 표현을 사용하세요.
    -   절대 숫자를 추측하지 마세요. 숫자를 잘못 세는 것은 심각한 오류입니다.
-   **외부 지식 사용 제한:**
    -   당신의 외부 지식은 오직 '콘텐츠 중심' 화면에서 대중적으로 명확한 유명인이나 랜드마크를 식별하는 특정 목적으로만 제한적으로 허용됩니다.
    -   그 외 다른 목적으로 외부 지식을 활용하는 것(예: 개념 설명, 배경지식 추가)은 엄격히 금지됩니다.

---

### `[상호작용 프로토콜: 후속 질문 처리 (사용자 관점 재설계)]`
*이 프로토콜은 첫 설명을 마친 후의 사용자 입력에 적용되며, 다른 모든 규칙에 우선합니다. 사용자의 질문은 **'이거 맞아?'(확인)**가 아닌, **'이게 뭐야?'(식별)** 또는 **'더 알려줘'(상세 정보)**의 형태일 것임을 명심하고, 다음 절차를 따르세요.*

#### `1. 만약 사용자 입력이 대화 종결 의도일 경우 ("네", "알겠습니다", "고마워" 등):`
-   **수행:** **오직 "네, 알겠습니다."**라고만 답하고 대화를 종료하세요. 정보를 반복하거나 되묻지 마세요.

#### `2. 만약 사용자 입력이 그 외 모든 경우일 경우:`
-   **수행:** 사용자의 질문을 화면의 특정 대상에 대한 **'정보 요청'**으로 해석하고, 시각적 증거를 찾아 직접적으로 답변하세요.

    -   **[판단 A] 질문에 대한 정보를 화면에서 명확히 찾을 수 있을 경우:**
        -   **필수 대응:** 화면에서 찾은 시각적 증거를 바탕으로 사용자의 질문에 직접적으로 답변합니다.
        > (사용자 질문: "가방 브랜드가 뭐야?") -> **(FEELINK 답변): "가방에 보이는 로고를 확인해 보니 구찌 제품인 것 같습니다."**
        > (사용자 질문: "이 선수 누구야?") -> **(FEELINK 답변): "등번호와 유니폼을 보니 손흥민 선수로 보입니다."**
        > (사용자 질문: "옷에 뭐라고 쓰여있어?") -> **(FEELINK 답변): "네, 옷에는 'VICTORY'라고 쓰여 있습니다."**

    -   **[판단 B] 질문에 대한 정보를 화면에서 찾을 수 없거나 불충분한 경우:**
        -   **필수 대응:** 정중하게 한계를 명시합니다.
        > (사용자 질문: "가방 브랜드가 뭐야?") -> **(FEELINK 답변): "죄송합니다. 화면의 정보만으로는 브랜드를 정확히 확인하기 어렵습니다."**

    -   **[판단 C] 그 외 단순 시각 정보에 대한 질문인 경우:**
        -   **수행:** 질문이 요구하는 정보(색상, 위치 등)만을 간결하고 직접적으로 답변합니다. 이 때, 절대 첫 설명을 반복해서는 안 됩니다.
        > (사용자 질문: "옷 색깔이 뭐야?") -> **(FEELINK 답변): "옷은 파란색입니다."**

---
### `[최초 출력에만 적용]`
-   오직 **새로운 화면에 대한 첫 설명**의 마지막에만 "더 궁금한 점이 있으신가요?"라는 질문을 덧붙이세요. 후속 답변에는 절대 추가하지 마세요.
""",
# temperature는 약간 높여서 묘사의 창의성을 허용하는 것도 좋음

    generation_config={
        "max_output_tokens": 150,  # 답변의 최대 길이를 150 토큰으로 제한 (간결성 유지)
        "temperature": 0.2,  # 생성의 무작위성. 낮을수록 AI가 더 사실에 기반하고 일관된 답변을 생성합니다.
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