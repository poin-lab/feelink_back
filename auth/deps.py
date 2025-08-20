# auth/deps.py
import os
from datetime import datetime, timedelta
from typing import Optional
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from dotenv import load_dotenv

from api.db_service import get_db_session, get_user_by_email
from api.models import User
from schema import TokenData

# 환경변수 또는 기본값
load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("치명적 오류: JWT_SECRET 환경변수가 설정되지 않았습니다.")

JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24h
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30d
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
# [수정] tokenUrl을 새로 만든 로그인 엔드포인트 경로로 수정
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"}) # [수정] type 필드 추가
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "refresh"}) # [수정] type 필드 추가
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_db_session)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])

        if payload.get("type") != "access": # [수정] 토큰 타입 검증
            logging.error(f"엑세스토큰 타입 오류: {payload.get('type')}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type; expected 'access'",
            )

        email: Optional[str] = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)

    except JWTError:
        raise credentials_exception

    user = await get_user_by_email(session, email=token_data.email)
    if user is None:
        raise credentials_exception

    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user