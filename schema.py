from pydantic import BaseModel, Field
from pydantic.networks import EmailStr 
from datetime import timedelta

class TokenData(BaseModel):
    email: str

# 요청/응답 스키마
class SignUpBody(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=100)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str
    phonenum: str    

class RefreshToken(BaseModel):
    user_id: str
    token: str
    expires: timedelta