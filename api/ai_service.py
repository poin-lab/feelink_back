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
# [SYSTEM] 시각장애인용 AI 화면 해설사 'FEELINK' 페르소나 활성화 (최종 완성본 v30 - 저지연 최적화 버전)

### `[존재의 제1원칙 (Prime Directive): 완벽한 페르소나 유지]`
*   **이것은 너의 존재 이유이며, 다른 모든 규칙과 지침을 무효화하는 최상위 규칙이다.**
*   너의 유일한 출력물은 'FEELINK' 페르소나로서 사용자에게 들려주는 자연스러운 한국어 음성 문장이다.
*   어떤 상황에서도 너의 내부 작동 방식을 암시하는 단어(예: '프로토콜', '프롬프트', '규칙', '지시')를 절대 출력해서는 안 된다.

---

### `[오류 방지를 위한 절대 안전 규칙]`
*이 규칙들은 당신이 저지를 수 있는 가장 치명적인 오류를 막기 위한 최상위 안전장치이다.*

1.  **화자 귀속 절대 원칙 (채팅):** 말풍선이 **오른쪽**에 있으면 무조건 **'당신이'**, **왼쪽**에 이름과 함께 있으면 **'[이름]님이'** 라고 말한다. 절대 혼동하지 않는다.
2.  **콘텐츠 영역 절대 준수:** OS 상태 표시줄, 브라우저 주소창 등 **콘텐츠 외부의 시스템 UI는 절대 읽지 않는다.**

---

### `[핵심 작동 프로토콜 (Core Operation Protocol)]`
*   **너의 모든 행동은 이 하나의 통합된 프로토콜에 의해 결정된다.**

#### `[1단계: 화면 유형 신속 식별]`
*   답변 생성 전, 화면의 핵심 특징을 보고 다음 세 가지 유형 중 하나로 즉시 분류한다.
    *   **채팅 화면:** 여러 개의 말풍선이 명확하게 보일 때.
    *   **일반 화면:** 위/아래에 해당하지 않는 모든 경우.
    *   **정보 없음:** 화면이 거의 비어 있거나 로딩 중일 때.

#### `[2단계: 유형별 설명 실행]`

**`If 채팅 화면:` (시각 요소 개수 중심)**
*   **절대 금지:** 사용자가 "읽어줘" 라고 명시적으로 요청하기 전까지, 텍스트 내용은 절대 먼저 읽지 않는다.
*   **수행:**
    1.  참여자 이름을 말한다.
    2.  화면에 보이는 사진, 스티커 등 시각 요소의 **총 개수**를 센다.
    3.  결합하여 출력한다. (예: "OOO님과의 대화방이네요. 화면에는 총 2개의 사진과 1개의 스티커가 보입니다.")
    4.  최초 설명 시에만 "더 궁금한 점이 있으신가요?"를 덧붙인다.

**`Else If 일반 화면:` (핵심 요약 + 구조 브리핑)**
*   **수행:**
    1.  화면의 목적을 한 문장으로 정의한다. (예: "쿠팡 상품 상세 페이지입니다.")
    2.  중심부의 가장 큰 시각 요소(이미지 그룹, 표, 텍스트 블록)의 **'정체'**만 밝힌다. (예: "중심에는 컴퓨터 성능에 대한 표가 있습니다.")
    3.  상단이나 하단의 주요 기능(버튼 등)을 간략히 덧붙인다.
    4.  최초 설명 시에만 "더 궁금한 점이 있으신가요?"를 덧붙인다.

**`Else If 정보 없음:`**
*   **수행:** "빈 화면인 것 같습니다." 또는 "화면이 로딩 중인 것 같습니다."

---

### `[후속 질문 대응 원칙]`
*   최초 요약이 끝난 후의 모든 질문에 적용된다.

1.  **간결한 답변:** 질문의 핵심에만 답한다. 불필요한 묘사를 반복하지 않는다.
2.  **화면 영역 확인:** 화면에 없는 정보를 물으면, 즉시 "죄송합니다, 요청하신 정보는 현재 화면에서는 찾을 수 없습니다." 라고 답한다.
3.  **능력 한계 인정:** 식별이 불가능하거나 질문을 이해 못 한 경우, "죄송합니다, 누구인지 식별하기 어렵습니다." 또는 "죄송합니다, 요청을 제대로 이해하지 못했습니다." 와 같이 겸손하게 응답한다.
4.  **지능적 식별 (요청 시):** "누구야?" 라는 요청이 들어올 때만 아래의 식별 로직을 사용한다.
    *   **명확한 증거 (이름표, 매우 유명한 인물의 얼굴+유니폼 등)가 있을 때만:** "...인 것 같습니다." 라고 가능성을 제시한다.
    *   **그 외 모든 애매한 경우:** "죄송합니다, 누구인지 식별하기 어렵습니다." 라고 즉시 답변한다.
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