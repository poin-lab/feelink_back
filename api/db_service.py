import logging
import os
from typing import AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID
from datetime import timedelta,datetime

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.future import select


# Pydantic 스키마와 SQLAlchemy 모델을 올바른 경로에서 import 합니다.
# 경로는 프로젝트 구조에 맞게 조정해야 할 수 있습니다.
from schema import UserCreate, RefreshTokenCreate # 스키마 이름을 명확히 함

from api.models import Conversation, User , RefreshToken
# 로거 설정
log = logging.getLogger("uvicorn")

# --- 1. 데이터베이스 연결 설정 (수정 없음) ---
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")

engine = create_async_engine(DATABASE_URL, echo=True)

AsyncSessionFactory = async_sessionmaker(
    engine,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession,
)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI 의존성 주입을 위한 DB 세션 제너레이터.
    요청마다 세션을 생성하고, 요청이 끝나면 자동으로 세션을 닫습니다.
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
        except Exception as e:
            log.error(f"DB 세션 중 오류 발생, 롤백 수행: {e}")
            await session.rollback()
            raise


# --- 3. 대화 관련 CRUD 함수 (세션 관리 방식 수정) ---

async def create_new_conversation(
    session: AsyncSession, # 👈 session을 첫 번째 인자로 받도록 수정
    conversation_id: UUID,
    image_url: Optional[str],
    initial_history: List[Dict[str, Union[str, List[str]]]],
    user_id: Optional[UUID] = None,
) -> None:
    """새로운 대화 기록을 데이터베이스에 저장합니다. (INSERT)"""
    new_convo = Conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        image_url=image_url,
        history=initial_history,
    )
    # 👇 'async with' 블록 제거, 전달받은 session 사용
    session.add(new_convo)
    await session.commit()
    log.info(f"[DB] 새 대화 저장 완료: cid={conversation_id}")


async def get_conversation_history(
    session: AsyncSession, # 👈 session을 첫 번째 인자로 받도록 수정
    conversation_id: UUID
) -> Optional[Conversation]:
    """주어진 ID로 대화 전체를 조회합니다. (SELECT)"""
    query = select(Conversation).where(Conversation.conversation_id == conversation_id)
    # 👇 'async with' 블록 제거, 전달받은 session 사용
    result = await session.execute(query)
    conversation = result.scalar_one_or_none()
    if conversation:
        log.info(f"[DB] 대화 조회 성공: cid={conversation_id}")
        return conversation
    else:
        log.warning(f"[DB] 대화 조회 실패: cid={conversation_id} 없음")
        return None


async def update_conversation_history(
    session: AsyncSession, # 👈 session을 첫 번째 인자로 받도록 수정
    conversation_id: UUID,
    new_turn: List[Dict[str, Union[str, List[str]]]]
) -> bool:
    """기존 대화에 새로운 질문/답변 쌍을 추가하여 업데이트합니다. (UPDATE)"""
    # 👇 'async with' 블록 제거, 전달받은 session 사용
    conversation = await session.get(Conversation, conversation_id)
    if not conversation:
        log.error(f"[DB] 업데이트 실패: cid={conversation_id} 없음")
        return False

    updated_history = conversation.history + new_turn
    conversation.history = updated_history

    await session.commit()
    log.info(f"[DB] 대화 업데이트 완료: cid={conversation_id}")
    return True
    

# --- 4. 계정 관련 CRUD 함수 (세션 관리 방식 일관성 유지) ---

# 이 함수는 이미 올바르게 작성되어 있었습니다. (수정 없음)
async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    """주어진 이메일로 사용자를 조회합니다."""
    query = select(User).where(User.email == email)
    result = await session.execute(query)
    user = result.scalar_one_or_none()
    
    if user:
        log.info(f"[DB] 사용자 조회 성공: email={email}")
        return user
    else:
        log.info(f"[DB] 사용자 조회 결과 없음: email={email}")
        return None
        

# 이 함수도 이미 올바르게 작성되어 있었습니다. (UserCreate 필드 수정 및 반환 타입 명확화)
# 'hashed_password'는 서비스 계층에서 처리해서 넘겨주는 것이 더 좋습니다.
async def create_user(session: AsyncSession, user_data: UserCreate, hashed_password: str) -> User:
    """새로운 사용자를 생성합니다. 비밀번호는 미리 해싱되어야 합니다."""
    new_user = User(
        email=user_data.email,
        name=user_data.name,
        phonenum=user_data.phonenum,
        hashed_password=hashed_password,
        # is_active는 모델에서 기본값을 가지므로 명시적으로 넣지 않아도 될 수 있습니다.
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    log.info(f"[DB] 사용자 생성 완료: email={user_data.email}")
    return new_user # 👈 id 대신 User 객체 전체를 반환하는 것이 더 유연합니다.


# 이 함수도 잘못된 패턴을 사용하고 있었으므로 수정합니다.
async def create_new_refresh_token(
    session: AsyncSession,
    user_id: UUID,
    token: str,
    expires: timedelta
) -> None:
    """새로운 리프레시 토큰을 데이터베이스에 저장합니다."""
    # 👈 [핵심 수정 2] timedelta를 실제 만료 시각(datetime)으로 변환합니다.
    expires_at = datetime.utcnow() + expires
    
    new_token = RefreshToken(
        user_id=user_id,
        token=token,
        expires_at=expires_at # 변환된 datetime 객체를 사용
    )
    session.add(new_token)
    await session.commit()
    log.info(f"[DB] 새 리프레시 토큰 저장 완료: user_id={user_id}")