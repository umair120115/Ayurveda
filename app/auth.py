import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import ValidationError
from dotenv import load_dotenv

#importing useful from other files
# from .database import get_collection
from database import get_collection
# from .models1 import UserInDB,TokenData
from models1 import UserInDB,TokenData
# from .security import verify_password
from security import verify_password

load_dotenv()

#-----  JWT Settings ----
SECRET_KEY=os.getenv("SECRET_KEY")
ALGORITHM=os.getenv("ALGORITHM","HS256")
ACCESS_TOKEN_EXPIRE_TIMES=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

if not SECRET_KEY:
    raise ValueError("SECRET KEY environment variable not set!")


# ----  OAuth Scheme --
#Should points to the token endpoint in main.py
oauth2_scheme=OAuth2PasswordBearer(tokenUrl="token")

#--- JWT Functions -----
def create_access_token(data: dict, expires_delta: Optional[timedelta]=None):
    to_encode=data.copy()
    if expires_delta:
        expire=datetime.now(timezone.utc) + expires_delta
    else:
        expire=datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_TIMES)
    to_encode.update({"exp":expire})
    encoded_jwt=jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM)
    return encoded_jwt

#-- Database Interaction for Auth --
async def get_user_by_username(username:str)-> Optional[UserInDB]:
    """ Retrieves user from DB by username"""
    users_collection=get_collection("users")    #Collection name defined here
    user_doc=await users_collection.find_one({"username":username})
    if user_doc:
        try:
            return UserInDB(**user_doc)  # validate DB against Pydantic model
        except ValidationError as e:
            print(f"DB data validation error for user P{username} : {e}")
            return None
    return None


# authenticate user from DBiona
async def authenticate_user(username: str, password: str)-> Optional[UserInDB]:
    """Authenticate a user. Return user object if valid, else None."""

    user=await get_user_by_username(username)
    if not user:
        return {"user":None,
                "message":"No account available with provided username!"}
    if not verify_password(password,user.hashed_password):
        return {"user":None, "message":"Invalid Credentials!"}

    return {'user':user,"message":"Welcome to tha platform!"}


# -- Dependency to get current authenticated user --
async def get_current_user(token: Annotated[str,Depends(oauth2_scheme)])-> UserInDB:
    credentials_exception=HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials!",
        headers={"WWW-Authenticate": "Bearer"},

    )
    try:
        playload=jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None= playload.get("username")
        if username is None:
            raise credentials_exception
        token_data=TokenData(username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    user= await get_user_by_username(token_data.username)
    if user is None:
        #incase of user deleted account
        raise credentials_exception
    return user

# -- For getting current active user 
async def get_current_active_user(
    current_user: Annotated[UserInDB, Depends(get_current_user)]
) -> UserInDB:
    # Add logic here if you have an 'is_active' flag in UserInDB
    # if not getattr(current_user, 'is_active', True):
    #     raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
        

