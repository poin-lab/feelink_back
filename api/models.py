import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# 1. 기본 Base 클래스 정의
# 모든 모델 클래스가 상속받을 기본 클래스입니다.
# SQLAlchemy가 이 클래스를 상속받는 모든 클래스를 테이블 모델로 인식하게 합니다.
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    # JSONB 컬럼을 Python의 dict 타입으로 매핑하기 위한 기본 설정
    type_annotation_map = {
        dict: JSONB
    }


# ---------------------------------------------------------------------------
# 2. 'users' 테이블 모델
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    # 컬럼 정의
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    phonenum: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False, server_default="local")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="TRUE")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # 관계 정의 (User 입장에서 자신과 연결된 다른 테이블들을 가리킴)
    # 'lazy="selectin"'은 User를 조회할 때 관련 fcm_devices와 conversations를 효율적으로 함께 로드하는 옵션입니다.
    fcm_devices: Mapped[List["FCMDevice"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin"
    )


# ---------------------------------------------------------------------------
# 3. 'fcm_devices' 테이블 모델
# ---------------------------------------------------------------------------
class FCMDevice(Base):
    __tablename__ = "fcm_devices"

    # 컬럼 정의 (SERIAL은 정수형 PK로 지정하면 자동으로 처리됨)
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    fcm_token: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    device_type: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # 관계 정의 (FCMDevice 입장에서 자신을 소유한 User를 가리킴)
    user: Mapped["User"] = relationship(back_populates="fcm_devices")


# ---------------------------------------------------------------------------
# 4. 'conversations' 테이블 모델
# ---------------------------------------------------------------------------
class Conversation(Base):
    __tablename__ = "conversations"

    # 컬럼 정의
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    # 프로토타입 단계에서는 user_id가 없을 수 있으므로 nullable=True로 설정
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    image_url: Mapped[Optional[str]] = mapped_column(Text)
    # 대화 기록은 JSONB 타입으로 저장 (Python에서는 dict로 처리)
    history: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # 관계 정의 (Conversation 입장에서 자신을 소유한 User를 가리킴)
    user: Mapped[Optional["User"]] = relationship(back_populates="conversations")