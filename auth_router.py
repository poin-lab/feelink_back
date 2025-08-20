# auth_router.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from api.db_service import get_db_session
from auth import auth_service
from schema import TokenResponse, UserCreate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# 이 라우터는 인증이 필요 없는 API들을 다룹니다.
router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup_endpoint(
    user_data: UserCreate,
):
    log.info(f"회원가입 시도: email={user_data.email}")
    return await auth_service.signup(
        session=Depends(get_db_session), user_data=user_data
    )
"""
email: EmailStr
name: str
password: str
phonenum: str 
"""

@router.post("/login", response_model=TokenResponse)
async def login_endpoint(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    log.info(f"로그인 시도: email={form_data.username}")

    return await auth_service.login(
        session=Depends(get_db_session),
        email=form_data.username,
        password=form_data.password
    )

"""
HTTP Method: POST
URL: [서버 주소]/auth/login 
2. 헤더 (Headers)
요청 본문이 폼 데이터 형식임을 서버에 알려주기 위해 아래 헤더를 반드시 포함해야 합니다.
Content-Type: application/x-www-form-urlencoded
3. 본문 (Body)
요청 본문(Body)에는 다음 두 필드를 키-값 형태로 구성하여 전송합니다.
username (string, 필수): 사용자의 이메일 주소 또는 아이디
password (string, 필수): 사용자의 비밀번호
"""
