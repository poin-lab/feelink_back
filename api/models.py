# models.py

import uuid
from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base

# [!!! 핵심 수정 사항 !!!]
# 데이터베이스 모델의 공통 부모가 될 Base 클래스를 여기서 직접 정의합니다.
# 이제 이 파일은 다른 내부 모듈(db.py)에 의존하지 않는 독립적인 설계도가 됩니다.
Base = declarative_base()


class User(Base):
    """
    사용자 정보를 저장하는 테이블의 SQLAlchemy 모델 (설계도)
    """
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    hashed_password = Column(String, nullable=False)
    phonenum = Column(String(20), unique=True, nullable=False)
    provider = Column(String(50), default='local', nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Conversation 테이블과의 관계 설정 (User는 여러 개의 Conversation을 가질 수 있다)
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    """
    AI와의 대화 기록을 저장하는 테이블의 SQLAlchemy 모델 (설계도)
    """
    __tablename__ = 'conversations'

    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # users 테이블의 id를 참조하는 외래 키
    # 로그인 기능 구현 전까지는 NULL 값을 허용합니다.
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True, index=True)

    image_url = Column(Text, nullable=True) # Blob Storage에 저장된 이미지의 URL

    # PostgreSQL의 JSONB 타입을 사용하여 효율적인 JSON 데이터 저장/조회
    history = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # User 테이블과의 관계 설정 (Conversation은 하나의 User에 속한다)
    user = relationship("User", back_populates="conversations")