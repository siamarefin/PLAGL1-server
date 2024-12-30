from fastapi import APIRouter, Depends, HTTPException, status
from models.schema import UserCreate, UserLogin
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from models.models import User
from database import get_db
from core.security import hash_password, verify_password, create_access_token, verify_token

router = APIRouter(prefix='/auth', tags=['auth'])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):

    try:
        hashed_password = hash_password(user.password)
        user = User(email=user.email, hashed_password=hashed_password, name=user.name)
        db.add(user)
        db.commit()
        db.refresh(user)    

        # return jwt_token

        token = create_access_token(data={ "user_id" : user.id, "name": user.name, "email" :user.email})
        return {"message": "User created successfully" ,"token" : token}
    
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Email already registered"
        )
    
    except Exception as e:

        print(e)

        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
    )
    

@router.post('/login', status_code=status.HTTP_202_ACCEPTED)
async def login_user(body: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    if not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )


    # return jwt_token
    token = create_access_token(data={ "user_id" : user.id, "name": user.name, "email" :user.email})
    return {"message" : "User logged in successfully", "token" : token}
    
    

@router.get("/allusers")
async def get_all_users(db: Session = Depends(get_db), user_info: dict = Depends(verify_token)):

    name = user_info.get("name")

    users = db.query(User).all()
    return {"logged in user": name,  "users" : users}

