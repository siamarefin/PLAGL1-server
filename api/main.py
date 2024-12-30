from fastapi import FastAPI
# from sqlalchemy.orm import Session
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from jose import JWTError, jwt
# from datetime import datetime, timedelta, timezone
# from passlib.context import CryptContext
# from api.models.models import User
# from database import SessionLocal, engine
# from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# from database import get_db
# from core.security import hash_password, verify_password

from routers.auth_router import router as auth_router
from routers.operation_router import router as operation_router
from routers.file_router import router as file_router
from routers.micro_router import router as micro_router
from routers.annotation_router import router as annotation_router
from routers.venn_router import router as venn_router
from routers.heatmap_router import router as heatmap_router 

app = FastAPI()

origins = [
    "http://localhost:3000",  
    "https://yourfrontenddomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins from the list
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# add a / route

@app.get("/")
async def root():
    return {"message": "PLAGL1 Server is running..."}

app.include_router(auth_router)
app.include_router(operation_router)
app.include_router(file_router)
app.include_router(micro_router)
app.include_router(annotation_router)
app.include_router(venn_router)
app.include_router(heatmap_router)

# Authenticate the user


# Create access token








