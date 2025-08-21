# auth_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
import logging

# ‼️ 세션을 가져오는 함수는 db.session 또는 유사한 경로에서 직접 가져오는 것이 더 좋습니다.
from api.db_service import get_db_session # 경로가 정확한지 확인 필요

from auth import auth_service
from schema import TokenResponse, UserCreate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("uvicorn") # 👈 uvicorn 로거를 사용하는 것이 더 표준적입니다.

auth_router = APIRouter(prefix="/auth", tags=["Auth"])

@auth_router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup_endpoint(
    user_data: UserCreate,
    session: AsyncSession = Depends(get_db_session)
):
    log.info(f"회원가입 시도: email={user_data.email}")
    
    # 주입받은 'session' 객체를 서비스 함수로 전달합니다.
    user_id = await auth_service.signup(
        session=session, 
        user_data=user_data
    )
    return {"message": "회원가입 성공", "user_id": user_id}


@auth_router.post("/login", response_model=TokenResponse)
async def login_endpoint(
    # form_data는 이 방식이 맞습니다.
    form_data: OAuth2PasswordRequestForm = Depends(),
    # --- 👇 [핵심 수정 2] ---
    # 로그인에서도 마찬가지로 Depends를 파라미터에서 사용하여 세션을 주입받습니다.
    session: AsyncSession = Depends(get_db_session)
):
    log.info(f"로그인 시도: email={form_data.username}")

    # 주입받은 'session' 객체를 login 서비스 함수로 전달합니다.
    return await auth_service.login(
        session=session,
        email=form_data.username,
        password=form_data.password
    )