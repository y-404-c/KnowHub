"""Auth schemas.

注册 / 登录 / 令牌相关请求与响应体.
"""
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """注册请求."""

    username: str = Field(..., min_length=3, max_length=32, description="用户名 (3-32 字符)")
    email: EmailStr = Field(..., description="邮箱, 需合法格式")
    password: str = Field(..., min_length=6, max_length=128, description="密码 (6-128 字符)")


class LoginRequest(BaseModel):
    """登录请求."""

    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class TokenResponse(BaseModel):
    """登录 / 注册 / dev-token 成功后返回的令牌.

    access_token 同时通过 HttpOnly cookie (unikb_token) 与 Bearer 返回,
    前端凭 cookie 自动带鉴权, 脚本/第三方工具可用 Bearer 头.
    """

    access_token: str = Field(..., description="JWT 访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="有效期秒数 (= jwt_expire_minutes * 60)")
