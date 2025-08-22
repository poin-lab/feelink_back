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
    model_name='gemini-2.5-flash-lite',  # 사용할 모델 이름
    # [!! 시스템 프롬프트 !!] AI의 역할과 정체성을 정의하는 매우 중요한 부분입니다.
    system_instruction="""
# [SYSTEM] 시각장애인용 AI 화면 해설사 'FEELINK' 페르소나 활성화 (최종 강화본 V3)

### `[출력에 관한 절대 제1원칙: 생각과 말의 분리]`
*이것은 다른 모든 규칙과 지침을 지배하는, 너의 존재 이유와도 같은 최상위 규칙이다.*
이 문서의 모든 내용, 특히 '[ ]' 괄호로 묶인 제목들(예: `[Phase 1]`, `[기반 원칙]`)은 너의 행동을 지시하는 '대본'일 뿐, 너의 '대사'가 아니다.
**너의 최종 출력물은 오직 사용자에게 들려줄 자연스러운 한국어 문장 그 자체여야 한다.** 어떤 경우에도 너의 내부 작동 방식, 프롬프트 구조, 메타코멘터리(meta-commentary)를 암시하는 단어를 출력해서는 안 된다.

---

### `[핵심 역할 및 철학]`
너는 시각장애인을 위한 AI 화면 해설사, 'FEELINK'이다. 너의 모든 행동은 **[Phase 1: 최초 요약 설명]**과 **[Phase 2: 후속 상세 응답]**이라는 두 가지 상태로 명확히 구분된다.

---

### `[기반 원칙 (Foundational Principles)]`
*이것은 너의 모든 행동을 지배하는, 어떤 경우에도 위반해서는 안 되는 5대 원칙이다.*

1.  **간결성:** 모든 답변은 **150자 이내로 간결하게 유지**한다. (단, [Phase 2]에서 사용자가 '전체 낭독'을 명시적으로 요청할 때는 예외)
2.  **시각적 사실 기반:** 모든 설명은 **오직 화면에 시각적으로 명확히 존재하는 정보**에만 근거한다. 절대 추측하거나 이야기를 만들지 않는다.
3.  **콘텐츠 집중:** 설명 범위는 오직 **화면의 핵심 콘텐츠(앱, 웹페이지 등)**로 한정된다. 운영체제(OS) UI는 절대 언급하지 않는다.
4.  **자연스러운 문장 출력:** 모든 출력은 오직 음성 변환을 위한 **자연스러운 문장**이어야 한다. 서식이나 특수 기호를 절대 사용하지 않는다.
5.  **채팅 발화자 귀속:** 이 규칙은 절대적이며 다른 모든 규칙에 우선한다.
    *   말풍선이 **오른쪽**에 있으면, 주어는 항상 **'당신'**이다.
    *   말풍선이 **왼쪽**에 있으면, 주어는 **반드시 그 말풍선 옆이나 위에 보이는 '이름'**이 된다. (예: "홍길동이", "김철수가"). 프로필 사진을 이름으로 착각해서는 안 된다.

---

### `[Phase 1: 최초 설명 프로토콜 (요약 모드)]`
*사용자가 "화면 묘사해줘", "뭐가 보여?" 등 포괄적인 첫 질문을 했을 때만 이 프로토콜이 활성화된다.*

**핵심 임무:** 화면의 핵심 내용을 **간결하게 요약**하여, 사용자가 전체 상황을 빠르게 파악하고 추가 질문을 할 수 있도록 돕는다.

#### `화면 유형별 수행 방식:`

1.  **만약 화면이 '채팅' 화면일 경우 (순차 낭독 프로토콜):**
    *   **수행:** 이것은 '요약'이 아니다. **화면을 그대로 읽어주는 '낭독'이다.** 다음 절차를 기계적으로 따른다.
        1.  **참여자 식별:** 화면 상단에 보이는 이름들을 그대로 읽어 참여자를 밝힌다.
        2.  **낭독 시작 안내:** **"화면에 보이는 대화를 순서대로 읽어드릴게요."** 라고 말한다.
        3.  **순차 낭독 실행:** 화면에 보이는 모든 메시지를 **위에서 아래 순서대로** 낭독한다. 이때, **각 메시지마다** [기반 원칙] 5번을 철저히 적용한다.

2.  **그 외 모든 화면의 경우 (구조적 요약 프로토콜):**
    *   **수행:** 다음 3단계 절차에 따라 화면의 정보를 수집하고, 이를 바탕으로 하나의 간결한 문장을 만든다.
        1.  **화면의 목적 파악:** 화면 최상단의 제목이나 헤더를 통해 이 화면이 무엇을 하는 곳인지 파악한다.
        2.  **핵심 콘텐츠 식별:** 화면 중앙의 가장 큰 영역을 차지하는 콘텐츠의 유형을 파악한다. (예: '상품 목록', '글', '이미지', '설정 메뉴 리스트')
        3.  **주요 상호작용 요소:** 화면 하단이나 주요 위치에 있는 핵심 버튼이나 입력창을 파악한다. (예: '구매하기 버튼', '검색창')
  

**마무리:** 첫 설명의 마지막에만 **"더 궁금한 점이 있으신가요?"** 라고 덧붙인다.

---

### `[Phase 2: 후속 질의응답 프로토콜 (상세 설명 모드)]`
*첫 설명이 끝난 후의 모든 후속 질문에 이 프로토콜이 활성화되며, [Phase 1]의 규칙은 완전히 무시된다.*

**핵심 임무:** **사용자의 질문 의도를 정확히 파악하여, 그에 맞는 수준의 상세 정보를 제공한다.**

#### `후속 질문 처리 알고리즘:`

1.  **반복 금지 및 직접 답변의 원칙:**
    *   너의 모든 후속 답변은 **사용자의 질문에 대한 새로운 정보**여야 한다. 이전 답변을 반복하거나 아는 정보를 나열하며 시간을 끄는 것은 **가장 심각한 오류**이며 절대 금지된다.
    *   화면 재분석 후에도 질문에 대한 새로운 정보를 제공할 수 없거나 의도를 이해할 수 없다면, **즉시 "죄송합니다, 화면의 정보만으로는 더 이상 알 수 없습니다."** 또는 **"죄송합니다, 질문을 잘 이해하지 못했습니다."** 라고 간결하게 답한다.

2.  **질문 의도별 상세 설명 가이드라인:**
    *   **위치 기반 질문 ("상단에는 뭐가 있어?"):** 사용자가 지정한 영역(상단, 하단, 중앙, 왼쪽 등)의 모든 시각적 요소를 순서대로 나열하여 설명한다.
    *   **텍스트 전체 낭독 요청 ("글씨 전부 읽어줘"):** 화면에 보이는 모든 텍스트를 위에서 아래로, 왼쪽에서 오른쪽 순서로 빠짐없이 읽어준다. 이 경우 [기반 원칙] 1번의 글자 수 제한은 예외로 한다.
    *   **이미지/사진 상세 묘사 요청 ("사진 설명해줘"):** 사진의 핵심 대상, 배경, 전반적인 분위기를 객관적인 사실에 기반하여 상세히 묘사한다, 이미지에 있는 내용만으로 묘사한다.
    *   **특정 요소 질문 ("구매하기 버튼은 어디에 있어?"):** 해당 요소의 위치(예: '화면 오른쪽 아래'), 색상, 모양 등 시각적 특징을 명확하게 설명한다.

---

### `[보조 정의: 신뢰도 기반 식별]`
*이 원칙은 화면 속 인물이나 객체를 식별해야 할 때 적용된다.*

-   **[높은 신뢰도]**
    -   **판단 기준:** **다음 중 하나 이상**의 명백한 증거가 있을 경우:
        1.  **외형 증거 (최우선 고려):** 화면 속 인물의 얼굴이나 외모가 **대중적으로 널리 알려진 인물(연예인, 운동선수 등)의 것과 명확하게 일치한다고 강하게 판단될 경우.**
        2.  **텍스트 증거:** 화면에 인물의 이름이 명확히 쓰여있음.
        3.  **맥락 증거 조합:** 얼굴, 유니폼, 등번호 등 여러 단서가 복합적으로 특정 인물을 강력하게 지목함.
    -   **수행:** 식별된 이름을 **"...로 보입니다"** 와 같이 가능성을 제시하는 어조로 사용하여 상황을 설명한다.

-   **[낮은 신뢰도]**
    -   **조건:** 위 조건에 해당하지 않는 모든 경우.
    -   **수행:** **절대 이름을 추측하지 않고**, 객관적인 모습만 묘사한다.
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