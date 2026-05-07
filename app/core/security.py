from jose import jwt, JWTError
from datetime import datetime, timedelta
import os
SECRET_KEY = os.getenv("JWT_SECRET_KEY") 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

def create_token(user_id: str):
    payload = {
        "sub": user_id,
        "role" : "user",  #for simplicity, we assign 'user' role to all tokens. In production, you would determine this based on your user database.
        "exp": datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None