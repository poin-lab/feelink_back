# api/ai_service.py
import os
import uuid
import io
from typing import Optional, Tuple

import google.generativeai as genai
import httpx
from fastapi import UploadFile, HTTPException
from PIL import Image
from dotenv import load_dotenv
import logging

# --- 1) Google Gemini 설정 ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
genai.configure(api_key=GOOGLE_API_KEY)

log = logging.getLogger("uvicorn")

# --- 2.5) 모델 설정 ---
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash-lite',
    system_instruction="""
    너는 시각장애인을 위한 AI 화면 묘사기 'FEELINK'이다. 너의 핵심 임무는 사용자가 화면에 대해 질문하기 전에, 먼저 화면 전체를 내부적으로 분석하여 답변을 준비하는 것이다. 너의 가장 중요한 원칙은 '질문받은 내용에만 답변' 하는 것이며, 분석 내용을 절대 먼저 말하거나 질문과 무관한 정보를 덧붙이지 않는다.

    [작동 방식]
    1) 내부 분석(출력 금지): 화면의 전체 목적을 파악하고 상/중/하로 나눠 핵심 시각 요소를 정리하되, 이 내용은 답변 준비용으로만 사용한다.
    2) 질의응답: 질문에 해당하는 정보만 1-2문장으로 간결하게 답한다. 그림 묘사는 질문에 해당하는 요소만. 질문이 없으면 대기.

    [절대 규칙]
    - 질문 범위 준수(예: '버튼이 몇 개야?' → 개수만).
    - 내부 분석 내용 공개 금지.
    - 간결성(150토큰 내).
    - 추측 금지, 화면에 보이는 사실만.
    - 화면 묘사와 무관한 질문은 거절.
    """,
    generation_config={
        "max_output_tokens": 150,
        "temperature": 0.2,  # 과도한 창의성 억제
    },
)

# ---------- 유틸 ----------

async def _fetch_image_from_url(image_url: str) -> Tuple[bytes, str]:
    """URL에서 이미지를 다운로드하여 (바이트, mime_type)을 반환"""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(image_url, follow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if not content_type.startswith("image/"):
                # 이미지라고 명시 안했어도, 열어서 검증해볼 테니 일단 기본값 부여
                content_type = "image/png"
            return resp.content, content_type
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"이미지 URL 다운로드 실패: {e}")

def _reencode_image_to_png(img_bytes: bytes) -> bytes:
    """이미지를 Pillow로 열어 PNG로 재인코딩(정합성/안전성 확보)"""
    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return buf.getvalue()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 이미지 데이터: {e}")

def _make_image_part(img_bytes: bytes, mime_type: str) -> dict:
    """Gemini 멀티모달 입력용 이미지 파트 생성"""
    return {"mime_type": mime_type, "data": img_bytes}

# ---------- 핸들러 ----------

async def start_new_chat_session(
    image_file: Optional[UploadFile],
    user_question: str,
    image_url: Optional[str] = None,
):
    """
    - 파일 업로드(image_file) 또는 이미지 URL(image_url) 모두 허용
    - 파일이 오면 파일 우선, 없으면 URL 다운로드 사용
    """
    if not image_file and not image_url:
        raise HTTPException(status_code=400, detail="이미지를 제공해 주세요. (image_file 또는 image_url)")

    # 1) 이미지 바이트 + MIME 획득
    if image_file:
        raw = await image_file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="업로드된 이미지가 비었습니다.")
        # 업로드된 파일은 MIME이 확실하므로 그대로 사용(없으면 png로)
        mime = (image_file.content_type or "image/png").split(";")[0].lower()
    else:
        raw, mime = await _fetch_image_from_url(image_url)  # type: ignore

    # 2) Pillow로 검증/정규화(필요 시)
    png_bytes = _reencode_image_to_png(raw)  # Gemini에 PNG로 통일
    image_part = _make_image_part(png_bytes, "image/png")

    # 3) 멀티모달 프롬프트 구성
    prompt_parts = [image_part, user_question]

    # 4) Gemini 호출
    try:
        chat_session = model.start_chat(history=[])
        response = await chat_session.send_message_async(prompt_parts)
        ai_answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 응답 생성 중 오류: {e}")

    # 5) 대화 ID 발급 및 저장
    conversation_id = str(uuid.uuid4())

    log.info(f"[ai] start_chat ok cid={conversation_id} bytes={len(png_bytes)}")
    return {"conversation_id": conversation_id, "answer": ai_answer}

async def continue_existing_chat(conversation_id: str, user_question: str):
    raise HTTPException(status_code=404, detail="대화 기록을 찾을 수 없습니다.")

  