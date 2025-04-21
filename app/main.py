from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from typing import List, Annotated
from pymongo.results import InsertOneResult,UpdateResult, InsertManyResult
from pymongo.errors import DuplicateKeyError
from datetime import datetime,timedelta
from bson.objectid import ObjectId

#importing from other files
from database import connect_to_mongo,close_mongo_connection,get_collection
from models1 import (
    UserCreate, UserPublic, UserInDB, UserUpdate, Token, PyObjectId,
)
# from .security import get_password_hash
from security import get_password_hash
from auth import (
    create_access_token,authenticate_user,get_current_active_user,ACCESS_TOKEN_EXPIRE_TIMES,get_current_user
)
# from .auth import (
    # create_access_token,authenticate_user,get_current_active_user,ACCESS_TOKEN_EXPIRE_TIMES,get_current_user
# )


# -- Lifespan Manager ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    print("INFO: Starting up Application\n")
    await connect_to_mongo()
    yield
    await close_mongo_connection()


app=FastAPI(
    lifespan=lifespan,
    title="Nature Cure - The God's blessing!",
    # version=""
)

# ---- Collection Name  ----
USERS_COLLECTION="users"

# -- API's for user's and their registraion, profile and Login
#registering api
@app.post("/register/",response_model=UserPublic,status_code=status.HTTP_201_CREATED)
async def register_user(user_data:UserCreate):
    """User registeration using API and validate existing user's."""
    users_collection=get_collection(USERS_COLLECTION)
    #checking if emal/username already exists!
    existing_user= await users_collection.find_one({
        "$or":[{"email":user_data.email},{"username":user_data.username}]
    })
    if existing_user:
        detail = "Email already registered" if existing_user["email"] == user_data.email else "Username already taken"
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    #hash the plain password coming as a request
    hashed_password=get_password_hash(user_data.password.get_secret_value())
    print(f"hashed-password={hashed_password}\n")

    user_doc=user_data.model_dump(exclude={"password"})
    user_doc["hashed_password"]=hashed_password
    user_doc["registered_at"]=datetime.now()
    try:
        insert_details : InsertOneResult= await users_collection.insert_one(user_doc)
        created_doc= await users_collection.find_one({"_id":insert_details.inserted_id})

        if created_doc:
            return UserPublic(**created_doc)
        else:
            return HTTPException(status_code=500,detail="Something went wrong!")
    except Exception as e:
        print(f"Error during user registration!,-> {e}\n")


#login api
@app.post("/token/",response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm,Depends()]): # Depends() injects directly form data
    """For authentication and on successful call, returns JWT token."""
    user= await authenticate_user(form_data.username, form_data.password)
    if not user:
        return HTTPException(status_code=500,detail="Incorrect username or password!",headers={"WWW-Authenticate":"Bearer"})
    access_token_expires=timedelta(minutes=ACCESS_TOKEN_EXPIRE_TIMES)
    access_token=create_access_token(
        data={"username":form_data.username},
        expires_delta=access_token_expires
    )
    return {"access_token":access_token,"token_type":"Bearer"}

#getting profile details
@app.get("/user/me",response_model=UserPublic, tags=["Users"])
async def get_current_users_details(current_user: Annotated[UserInDB,Depends(get_current_active_user)]):
    #this returns the details of the user and Hide hashed_password
    return current_user
    
#updating profile
@app.patch("/user/update/", response_model=UserPublic, tags=["Users"])
async def update_user_info(user_update_playload: UserUpdate ,current_user: Annotated[UserInDB,Depends(get_current_active_user)]):
    """Endpoint to update user profile."""
    users_collection=get_collection(USERS_COLLECTION)
    update_data=user_update_playload.model_dump(exclude_unset=True)  # for updating user's profile and only those fields which will be provide
    if not update_data:
        return HTTPException(status_code=500,detail="No data provided to update in profile!")
    new_email=update_data.get("email")
    
    print(f"update_data request:{update_data}\n NewEmail --->  {new_email}\n")
    if new_email and new_email!= current_user.email:
        # validate email in db 
        existing_email_user=await users_collection.find_one({"email": new_email,"_id":{"$ne":current_user.id}})
        if existing_email_user:
            return HTTPException(status_code=500,detail="User with email already exists!")
        print(f"step 2\n")
        try:
            # update_result : UpdateResult = users_collection.update_one({"_id":current_user.id},{"$set":update_data})
            filtering_document={"_id":current_user.id}
            update_document={"$set":update_data}
            update_result:   UpdateResult=users_collection.update_one(filtering_document,update_document)
            
            updated_user_doc= await users_collection.find_one({"_id":current_user.id})
            print(f"Updated profile:{updated_user_doc}\n")
            
            if not updated_user_doc:
                return HTTPException(status_code=500,detail="No user found in DB!\n")

            return UserPublic(**updated_user_doc)
        except Exception as e:
            print(f"Got some error - {e}")
            return HTTPException(status_code=500,detail="Something went wrong!")
    if not new_email:
        try:
            filtering_document={"_id":current_user.id}
            update_document={"$set":update_data}
            update_result:UpdateResult= users_collection.update_one(filtering_document,update_document)

            updated_user_doc=await users_collection.find_one({"_id":current_user.id})
            if not updated_user_doc:
                return HTTPException(status_code=500,detail="No user found!")
            return UserPublic(**updated_user_doc)
        except Exception as e:
            return HTTPException(status_code=500,detail=f"Error - {e}")
        




