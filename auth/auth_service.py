# auth/auth_service.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import timedelta

from api import db_service
from api.models import User
from auth.deps import hash_password, verify_password, create_access_token, create_refresh_token

from auth import deps
from schema import SignUpBody, TokenResponse , UserCreate



async def signup(session: AsyncSession, user_data: UserCreate) -> User:
    """
    회원가입 비즈니스 로직.
    """
    existing_user = await db_service.get_user_by_email(session, email=user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="이미 등록된 이메일입니다.",
        )
    
    hashed_pass = hash_password(user_data.password)
    
    new_user = await db_service.create_user(
        session, user_data=user_data, hashed_password=hashed_pass
    )
    return new_user.id


# 함수 이름을 'login'으로 원복했습니다.
async def login(session: AsyncSession, email: str, password: str) -> dict:
    """
    로그인 비즈니스 로직.
    """
    user = await db_service.get_user_by_email(session, email=email)

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 올바르지 않습니다.",
        )
        
    access_token = create_access_token(data={"sub": user.email})
    refresh_token = create_refresh_token(data={"sub": user.email, "user_id": str(user.id)}) # user_id도 payload에 포함 권장

    await db_service.create_new_refresh_token(session, user_id=user.id, token=refresh_token , expires=timedelta(minutes=deps.REFRESH_TOKEN_EXPIRE_MINUTES))

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
