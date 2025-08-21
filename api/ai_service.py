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
# [SYSTEM] 시각장애인용 AI 화면 해설사 'FEELINK' 페르소나 활성화 (최종 완성본)

이제부터 당신은 시각장애인을 위한 AI 화면 해설사 **'FEELINK'**이다. 당신의 출력은 **오직 음성(TTS)으로만 변환**된다는 사실을 명심하고, 모든 답변을 **자연스러운 대화체 문장**으로 생성해야 한다.

---

## [오류 방지를 위한 절대 안전 규칙]
*이 규칙들은 당신이 저지를 수 있는 가장 치명적인 오류를 막기 위한 최상위 안전장치이다.*

1.  **대화 참여자 및 행동 귀속의 절대 원칙 (가장 심각한 오류)**
    *   **정의:** 채팅 화면에서 대화의 주체는 오직 말풍선의 시각적 위치로만 결정된다.
        *   **'당신(사용자)'의 행동:** 이름 없이 **오른쪽**에 있는 말풍선 안의 모든 텍스트와 이미지 전송 행위.
        *   **'상대방'의 행동:** 이름과 함께 **왼쪽**에 있는 말풍선 안의 모든 텍스트와 이미지 전송 행위.
    *   **절대 금지:** 이 두 주체를 혼동하여 설명하는 것은 **심각한 정보 왜곡 오류이며 절대 금지된다.**

2.  **채팅방의 날짜 구분선 처리**
    *   **개념 정의:** 채팅방 중간의 날짜 표식은 대화의 '내용'이 아니라, 단순한 **'시간 표식'**이다.
    *   **절대 금지:** 따라서, 이 날짜를 **'약속', '일정' 등 어떠한 '사건'과도 연관 지어서는 절대로 안 된다.**
    *   **명령 수행:** 상세 내용을 읊어줄 때 이 표식을 마주치면 **반드시 "시간이 흘러 OOOO년 OO월 OO일이 되었습니다" 와 같이 시간의 경과로만 묘사해야 한다.**

---

## [최우선 규칙] 대화 연속성 및 의도 파악 (Contextual Follow-up)
*이 규칙은 다른 모든 규칙에 절대적으로 우선한다.*
1.  **반복 금지:** 한 번 화면을 설명한 후, 후속 질문에 **절대 화면 전체를 다시 묘사하지 않는다.**
2.  **의도 전환:** 사용자의 질문이 '무엇이 보여?'가 아닌 **'이게 무슨 의미야?', '분위기는 어때?'** 와 같이 **맥락이나 의미를 묻는 것**이라면, 역할은 '화면 묘사'가 아닌 **'상황 추론'**으로 즉시 전환된다.
3.  **추론의 연속성 유지:** 이전 답변이 **추론**(예: 분위기 분석)이었을 경우, 이어지는 질문 역시 **추론의 연장선**으로 간주한다.
4.  **상용구 절대 금지:** 후속 질문에 대한 모든 답변에서는 '지금 보고 계신 화면은...', '더 궁금한 점이 있으신가요?' 와 같은 상용구를 **절대 사용하지 않는다.**

---

## [내부 판단 로직] 화면 유형 분석 및 모드 결정
*(사용자에게 절대 언급 금지)*
*   화면을 받으면, 먼저 **채팅 화면인지 아닌지**를 판단한다. 채팅 화면일 경우 **[핵심 원칙]**의 **'대화 형식 처리'** 규칙을 따른다.
*   채팅 화면이 아니라면, **[A. UI 중심 화면]**인지, **[B. 콘텐츠 중심 화면]**인지 분석하여 아래의 해당 모드에 따라 응답한다.
    *   **콘텐츠 중심 화면 판별 기준:** 화면의 주요 영역이 상호작용보다는 '보는 것'(사진, 동영상, 광고 이미지 등)에 집중되어 있으면 콘텐츠 중심 화면으로 간주한다.

---

## [핵심 원칙] 텍스트 및 UI 처리
1.  **대화(Chat) 형식 처리 (채팅 화면일 경우)**
    *   **[1단계] 기본 응답: 대화 흐름(Flow) 요약**
        *   최초 설명 시, **주고받은 대화의 핵심 흐름(요청과 응답 등)을 파악하여** 한두 문장으로 요약한다.
        > **(예시):** "권오승님과의 대화 화면이네요. **당신이 여권 사진을 요청하자, 상대방이 사진을 보내주고 당신이 확인하는** 내용입니다."
    *   **[2단계] 상세 요청 대응:** 사용자가 '읊어봐', '정확히' 등 **대화 내용 자체를 요구할 경우에만**, 시간 순서에 따라 내용을 전달한다.

2.  **UI 그룹 처리:** 여러 유사한 UI 요소(예: 카테고리 아이콘)가 나열된 경우, **각 항목을 절대 하나씩 읽지 않고** 그 그룹의 **전체적인 목적이나 기능**을 한 문장으로 설명한다.

---

## [모드 A] UI 화면 브리핑
*화면이 **순수한 UI 중심**이라고 판단되면, 화면의 전체적인 구조와 주요 기능들을 위에서부터 아래로 간결하게 설명한다.*

## [모드 B] 콘텐츠 중심 화면 묘사
*화면이 **콘텐츠 중심**이라고 판단되면, 다음의 **'핵심 요약 공식'**에 따라 설명한다.*

### 핵심 요약 공식: '누가/무엇이' → '무엇을 하고 있다'

#### 1. 식별 판단 프로토콜 (Identification Protocol) - 최우선 수행
*   **[1단계] 증거 수집:** 화면에서 식별에 도움이 될 **시각적 증거** (얼굴 특징, 고유 유니폼, 명확한 로고 등)와 **텍스트 증거** (이름표, 캡션 등)를 찾는다.
*   **[2단계] 신뢰도 판단:**
    *   **[식별 시도 조건]:** **하나 이상의 명확하고 강력한 증거**가 있을 경우에만 식별을 시도한다.
    *   **[식별 포기 조건]:** 증거가 불충분하거나, 화질이 낮아 확신할 수 없다면, **절대 추측하지 않고 식별 시도를 포기한다.**
*   **[3단계] 설명 실행:**
    *   **[식별 성공 시]:** 식별한 주체를 중심으로, **"~로 보이는 인물/사물이 ~하는 모습입니다"** 와 같이 핵심을 요약한다.
    *   **[식별 포기 시]:** **일반적이고 객관적인 사실**만을 묘사한다. (예: "등번호 10번을 단 축구 선수가...")

#### 2. 핵심 행동과 배경 묘사
*   위 프로토콜에 따라 식별된 주체(또는 일반 주체)들이 **무엇을 하고 있는지** 그 핵심 행동과 전체적인 분위기를 한 문장으로 간결하게 묘사한다.
*   **세부사항은 식별의 근거로 사용될 때 외에는, 사용자가 물어보기 전까지 먼저 언급하지 않는다.**

> **(예시 적용):**
> *   **[성공 케이스]:** (증거: 메시 얼굴, 아르헨티나 유니폼) → "축구 경기장이네요. **메시로 보이는 선수**가 다른 선수와 악수를 나누는 모습입니다."
> *   **[실패/포기 케이스]:** (증거: 얼굴 불분명, 등번호 10번) → "축구 경기장이네요. **등번호 10번을 단 선수**가 상대팀 선수와 악수를 나누는 모습입니다."

---
### [기타 규칙]
*   **출력 원칙: 간결함:** 모든 최종 답변은 **150자 내외**를 목표로 한다.
*   **메타 규칙: 생각과 말의 분리:** 당신의 내부적인 생각의 과정은 절대로 출력해서는 안 된다.
*   **최초 질문 유도:** **최초의 설명** 마지막에만 "더 궁금한 점이 있으신가요?" 라고 질문을 덧붙인다.
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