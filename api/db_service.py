import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID
from datetime import timedelta
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.future import select
from schema import UserCreate , RefreshToken
from api.models import Conversation , User  # 우리가 만든 모델을 임포트

# 로거 설정
log = logging.getLogger("uvicorn")

# --- 1. 데이터베이스 연결 설정 ---
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


# --- 2. 비동기 세션 관리를 위한 컨텍스트 매니저 ---
@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    session: AsyncSession = AsyncSessionFactory()
    try:
        yield session
    except Exception as e:
        log.error(f"DB 세션 중 오류 발생, 롤백 수행: {e}")
        await session.rollback()
        raise
    finally:
        await session.close()


# --- 3. 대화 관련 CRUD 함수 ---

# !! 바로 이 함수가 없거나 이름이 달라서 오류가 발생한 것입니다 !!
async def create_new_conversation(
    conversation_id: UUID,
    image_url: Optional[str],
    initial_history: List[Dict[str, Union[str, List[str]]]],
    user_id: Optional[UUID] = None,
) -> None:
    """
    새로운 대화 기록을 데이터베이스에 저장합니다. (INSERT)
    """
    new_convo = Conversation(
        conversation_id=conversation_id,
        user_id=user_id,
        image_url=image_url,
        history=initial_history, # type: ignore
    )
    async with get_db_session() as session:
        session.add(new_convo)
        await session.commit()
        log.info(f"[DB] 새 대화 저장 완료: cid={conversation_id}")


async def get_conversation_history(conversation_id: UUID) -> Optional[Conversation]:
    """
    주어진 ID로 대화 전체를 조회합니다. (SELECT)
    """
    query = select(Conversation).where(Conversation.conversation_id == conversation_id)
    async with get_db_session() as session:
        result = await session.execute(query)
        conversation = result.scalar_one_or_none()
        if conversation:
            log.info(f"[DB] 대화 조회 성공: cid={conversation_id}")
            return conversation
        else:
            log.warning(f"[DB] 대화 조회 실패: cid={conversation_id} 없음")
            return None


async def update_conversation_history(
    conversation_id: UUID, new_turn: List[Dict[str, Union[str, List[str]]]]
) -> bool:
    """
    기존 대화에 새로운 질문/답변 쌍을 추가하여 업데이트합니다. (UPDATE)
    """
    async with get_db_session() as session:
        conversation = await session.get(Conversation, conversation_id)
        if not conversation:
            log.error(f"[DB] 업데이트 실패: cid={conversation_id} 없음")
            return False

        updated_history = conversation.history + new_turn # type: ignore
        conversation.history = updated_history

        await session.commit()
        log.info(f"[DB] 대화 업데이트 완료: cid={conversation_id}")
        return True
    

# 계정관련
async def get_user_by_email(session: AsyncSession, email: str) -> Optional[User]:
    query = select(User).where(User.email == email)
    async with get_db_session() as session:
        result = await session.execute(query)
        user = result.scalar_one_or_none()
        if user:
            log.info(f"[DB] 사용자 조회 성공: email={email}")
            return user
        else:
            log.warning(f"[DB] 사용자 조회 실패: email={email} 없음")
            return None
        

async def create_user(session: AsyncSession, user_data: UserCreate) -> User:
    new_user = User(
        email=user_data.email,
        hashed_password=user_data.password,
        is_active=True
    )
    async with get_db_session() as session:
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        log.info(f"[DB] 사용자 생성 완료: email={user_data.email}")
        return new_user.id


async def create_new_refresh_token(session: AsyncSession, user_id: UUID, token: str , expires: timedelta) -> None:
    new_token = RefreshToken(
        user_id=user_id,
        token=token,
        expires=expires
    )
    async with get_db_session() as session:
        session.add(new_token)
        await session.commit()
        log.info(f"[DB] 새 리프레시 토큰 저장 완료: user_id={user_id}")