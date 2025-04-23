from fastapi import FastAPI, HTTPException, status, Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
from typing import List, Annotated, Dict, Any
import logging
from pymongo.results import InsertOneResult,UpdateResult, InsertManyResult
from pymongo.errors import DuplicateKeyError
from datetime import datetime,timedelta
from bson.objectid import ObjectId
import uuid

#importing from other files
from database import connect_to_mongo,close_mongo_connection,get_collection
from models1 import (
    UserCreate, UserPublic, UserInDB, UserUpdate, Token, PyObjectId,Store,StoreCreate,TopicList, ProcessResponse, QueryRequest, QueryResponse, RetrievedChunk
)
# from .security import get_password_hash
from security import get_password_hash
from auth import (
    create_access_token,authenticate_user,get_current_active_user,ACCESS_TOKEN_EXPIRE_TIMES,get_current_user
)

from dotenv import load_dotenv
from pipeline import ( # Import initialization functions and core logic
    initialize_llm,
    initialize_embedding_model,
    initialize_agent_executor,
    initialize_text_splitter,
    setup_milvus_collection,
    run_pipeline_for_topic,
    COLLECTION_NAME, # Import constants if needed
    CONNECTION_ALIAS
)
from datetime import time
from pymilvus import Collection, connections, utility # Import necessary Milvus components
from langchain.agents import AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Or specific embedding class
from langchain_groq import ChatGroq # Or specific LLM class
# from .auth import (
    # create_access_token,authenticate_user,get_current_active_user,ACCESS_TOKEN_EXPIRE_TIMES,get_current_user
# )



# -- Lifespan Manager ----
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Handles startup and shutdown events."""
#     print("INFO: Starting up Application\n")
#     await connect_to_mongo()
#     yield
#     await close_mongo_connection()

# --- Load Environment Variables ---
load_dotenv()

# --- Logging Setup ---
# Configure logging level for more details if needed during development
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global State for Initialized Components ---
app_state: Dict[str, Any] = {}

# --- Modified Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events."""
    logging.info("INFO: Starting up Application")
    # Connect to MongoDB (existing)
    await connect_to_mongo()

    # Initialize Pipeline Components
    logging.info("INFO: Initializing Pipeline Components...")
    try:
        # Initialize LLM
        app_state["llm"] = initialize_llm()
        # Initialize Embedding Model
        app_state["embeddings_model"] = initialize_embedding_model()
        # Initialize Agent Executor (needs LLM)
        app_state["agent_executor"] = initialize_agent_executor(app_state["llm"])
        # Initialize Text Splitter
        app_state["text_splitter"] = initialize_text_splitter()
        # Setup Milvus Connection and Collection
        logging.info("INFO: Setting up Milvus connection and collection...")
        milvus_collection = setup_milvus_collection()
        app_state["milvus_collection"] = milvus_collection
        logging.info("INFO: Pipeline components initialized successfully.")

    except Exception as e:
        logging.critical(f"CRITICAL: Failed to initialize pipeline components during startup: {e}", exc_info=True)
        # Optionally raise the error to prevent startup if components are essential
        # raise RuntimeError("Failed to initialize critical pipeline components") from e
        app_state["pipeline_initialized"] = False # Flag initialization failure
    else:
        app_state["pipeline_initialized"] = True

    yield # Application runs here

    # --- Shutdown ---
    logging.info("INFO: Shutting down Application")
    # Disconnect Milvus
    try:
        logging.info("INFO: Disconnecting from Milvus...")
        if connections.has_connection(CONNECTION_ALIAS):
            connections.disconnect(CONNECTION_ALIAS)
            logging.info("INFO: Milvus connection closed.")
    except Exception as e:
        logging.warning(f"WARN: Error disconnecting from Milvus: {e}")
    # Close MongoDB connection (existing)
    await close_mongo_connection()
    logging.info("INFO: Application shutdown complete.")




# --- Initialize FastAPI App with Lifespan ---
app = FastAPI(
    lifespan=lifespan,
    title="Nature's Cure - Integrated AI Pipeline",
    version="1.1.0" # Updated version
)

# --- Helper Function to Check Pipeline State ---
def check_pipeline_ready():
    if not app_state.get("pipeline_initialized", False):
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail="Pipeline components are not ready. Please try again later."
        )
    # Check for specific components needed by the endpoint
    if not all(k in app_state for k in ["milvus_collection", "embeddings_model", "llm", "agent_executor", "text_splitter"]):
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail="One or more essential pipeline components failed to initialize."
        )

# === Existing User/Store Endpoints ===
# (@app.post("/register/"), @app.post("/token/"), etc. remain here)
# Make sure they work alongside the new lifespan logic.
# Add your existing user/store endpoints here...
# Example:
@app.get("/")
async def read_root():
    return {"message": "Welcome to Nature's Cure API with RAG capabilities!"}

# === New Pipeline Processing Endpoint ===
@app.post("/process-topics/", response_model=ProcessResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Data Pipeline"])
async def process_topics_endpoint(
    topic_list: TopicList,
    background_tasks: BackgroundTasks,
    # Optional: Add authentication dependency if needed
    # current_user: Annotated[UserInDB, Depends(get_current_active_user)]
):
    """
    Accepts a list of topics and processes them in the background
    using the Tavily->Milvus pipeline.
    """
    check_pipeline_ready() # Ensure components are loaded before proceeding

    milvus_collection = app_state["milvus_collection"]
    agent_executor_instance = app_state["agent_executor"]
    text_splitter_instance = app_state["text_splitter"]
    embeddings_model_instance = app_state["embeddings_model"]
    topics = topic_list.topics
    num_topics = len(topics)

    if num_topics == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No topics provided.")

    logging.info(f"Received request to process {num_topics} topics.")

    # Define the background task function
    # Pass initialized components explicitly to the background task
    def background_processing(
        topics_to_run: List[str],
        collection: Collection,
        agent_exec: AgentExecutor,
        splitter: RecursiveCharacterTextSplitter,
        embed_model: HuggingFaceEmbeddings # Use specific type hint
        ):
        logging.info(f"Background task started for {len(topics_to_run)} topics.")
        successful = 0
        failed = 0
        for i, topic in enumerate(topics_to_run):
            logging.info(f"Background processing topic {i+1}/{len(topics_to_run)}: '{topic}'")
            try:
                # Call the refactored function from pipeline.py
                success = run_pipeline_for_topic(
                    topic=topic,
                    milvus_collection=collection,
                    agent_executor_instance=agent_exec,
                    text_splitter_instance=splitter,
                    embeddings_model_instance=embed_model
                )
                if success:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Exception in background task for topic '{topic}': {e}", exc_info=True)
                failed += 1
        logging.info(f"Background task finished. Successful: {successful}, Failed: {failed}")

    # Add the processing job to background tasks
    background_tasks.add_task(
        background_processing,
        topics,
        milvus_collection,
        agent_executor_instance,
        text_splitter_instance,
        embeddings_model_instance
        )

    return ProcessResponse(
        message=f"Accepted {num_topics} topics for background processing.",
        topics_received=num_topics
    )


# === New RAG Query Endpoint ===
import time
@app.post("/query-rag/", response_model=QueryResponse, tags=["RAG Query"])
async def query_rag_endpoint(
    request: QueryRequest,
    # Optional: Add authentication dependency if needed
    # current_user: Annotated[UserInDB, Depends(get_current_active_user)]
):
    """
    Accepts a query, searches the Milvus vector store, and generates
    an answer using an LLM based on retrieved context.
    """
    check_pipeline_ready() # Ensure components are loaded

    query = request.query
    top_k = request.top_k
    milvus_collection: Collection = app_state["milvus_collection"]
    embedding_model: HuggingFaceEmbeddings = app_state["embeddings_model"] # Use specific type hint
    llm_model: ChatGroq = app_state["llm"] # Use specific type hint

    logging.info(f"Received RAG query: '{query}', top_k: {top_k}")

    try:
        # 1. Embed the query
        logging.info("Embedding query...")
        start_time = time.time()
        query_embedding = embedding_model.embed_query(query) # Use embed_query for single queries
        logging.info(f"Query embedding generated in {time.time() - start_time:.2f}s.")

        # 2. Search Milvus
        # Ensure search params match index params (metric_type) and are reasonable
        search_params = {
            "metric_type": "L2",  # Must match index metric type created in setup_milvus_collection
            "params": {"ef": 128},  # ef (search efficiency) - higher=more accurate/slower, tune this
        }
        logging.info(f"Searching Milvus collection '{COLLECTION_NAME}'...")
        start_time = time.time()
        results = milvus_collection.search(
            data=[query_embedding],
            anns_field="embedding", # Your vector field name
            param=search_params,
            limit=top_k,
            output_fields=["text_chunk", "source_document"] # Fields to retrieve from Milvus
        )
        logging.info(f"Milvus search completed in {time.time() - start_time:.2f}s. Found {len(results[0]) if results else 0} potential results.")

        # 3. Format Context and Prepare Response Data
        context = ""
        retrieved_chunks_for_response = []
        if results and results[0]:
            context_lines = []
            logging.info("Processing search hits...")
            for i, hit in enumerate(results[0]):
                # Ensure entity and methods exist before calling them
                if hasattr(hit, 'entity') and callable(getattr(hit.entity, 'get', None)):
                    # text = hit.entity.get('text_chunk', f'Error: text_chunk missing in hit {i}')
                    # source = hit.entity.get('source_document', 'N/A')
                    text = getattr(hit.entity, 'text_chunk', f'Error: text_chunk missing in hit {i}')
                    source = getattr(hit.entity, 'source_document', 'N/A')
                    score = hit.distance # L2 distance, lower is better
                    context_lines.append(text)
                    retrieved_chunks_for_response.append(RetrievedChunk(text=text, source=source, score=score))
                    logging.info(f"  Hit {i+1}: Score={score:.4f}, Source='{source}', Text='{text[:100]}...'")
                else:
                    logging.warning(f"  Hit {i+1} has unexpected structure: {hit}")

            context = "\n\n".join(context_lines)
            logging.info(f"Formatted context from {len(retrieved_chunks_for_response)} chunks.")
        else:
             logging.warning("No relevant context found in Milvus for this query.")
             # Return a specific message if no context is found
             return QueryResponse(
                 answer="I don't have relevant knoledge about that but will find about it soon.",
                 retrieved_chunks=[]
            )

        # 4. Construct Prompt for LLM
        rag_template = """You are an assistant providing information based on scientific context. Answer the following question based ONLY on the provided context. Be concise and directly answer the question. If the context does not contain the answer, state that clearly.

Context:
---
{context}
---

Question: {question}

Answer:"""
        prompt_text = rag_template.format(context=context, question=query)
        logging.info("Constructed prompt for LLM.")
        # logging.debug(f"LLM Prompt:\n{prompt_text}") # Uncomment for debugging

        # 5. Generate Answer with LLM
        logging.info("Generating answer with LLM...")
        start_time = time.time()
        # Use invoke for a single response with ChatGroq
        ai_response = llm_model.invoke(prompt_text)
        answer = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
        logging.info(f"LLM generation complete in {time.time() - start_time:.2f}s.")

        return QueryResponse(answer=answer, retrieved_chunks=retrieved_chunks_for_response)

    except Exception as e:
        logging.exception(f"Error during RAG query processing for query: '{query}'")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during query processing.")

# === Keep your existing store endpoints ===
# (@app.post('/store/register/'), @app.get("/all/stores/"), etc.)
# Make sure they don't conflict with the lifespan changes
# Add your existing store endpoints here...



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
        

#--- Collection Name ---#
STORES_COLLECTION='stores'

#api for registering as a store
from datetime import timezone
import logging
@app.post('/store/register/',response_model=Store,tags=["Stores management"])
async def registering_store(store_data:StoreCreate, current_user:Annotated[UserInDB,Depends(get_current_active_user)]):
    stores_collection=  get_collection(STORES_COLLECTION) #for getting collection of 
    # FastAPI/Pydantic handles request body validation before your function runs. If the request body is empty or doesn't conform to StoreCreate, FastAPI will automatically return a 422 Unprocessable Entity error. This check is unnecessary and the 500 status code is incorrect for client input errors.
    print(f" User id = {current_user.id}\n User's Name ={current_user.name}\n")
    existing_store=await stores_collection.find_one({"$or":[{"email":store_data.email},{"store_lat":store_data.store_lat,"store_lang":store_data.store_lang}]})
    if existing_store:
        raise HTTPException(
        status_code=409, # 409 Conflict is standard for duplicates
        detail="A store with this email or at this exact location already exists."
        )
    store_details= store_data.model_dump()
    owner_related_id=ObjectId(current_user.id)
    store_details["owner_id"]=owner_related_id
    store_details["created_at"] = datetime.now(timezone.utc) #will store as datetime
    store_details["is_active"] = True
    store_details["opening_time"]=store_details["opening_time"].strftime('%H:%M:%S')
    store_details["closing_time"]=store_details["closing_time"].strftime('%H:%M:%S')
    # print(f"playload = {store_details}\n")
    try:
        insert_result=await stores_collection.insert_one(store_details)
        inserted_id=store_details["_id"]
        store_document_data = await stores_collection.find_one({"$or":[{"_id": inserted_id}]})
        # if insert_result.inserted_id!= store_data.storeid:
            # return {"Error":f"Warning: MongoDB generated ID {insert_result.inserted_id} differs from model ID {store_data.storeid}"}
        # store_document_data=await stores_collection.find_one({
        #     "$or":[{"storeid":inserted_id},{"owner_id":current_user.id}]
        # })
        if not store_document_data:
            logging.error(f"Failed to retrieve newly inserted store with storeid: {inserted_id}")
            raise HTTPException(status_code=500, detail="Failed to retrieve store after creation.")
        return Store(**store_document_data)
    except Exception as e:
        # return HTTPException(status_code=500, detail=f"Error occured as {e}")
        # except Exception as e:
        # Log the actual error with traceback for server-side debugging
        logging.exception(f"Error during store registration for owner {current_user.id}")

        # RAISE the HTTPException instead of returning it
        raise HTTPException(
            status_code=500,
            # Don't expose raw error details to the client
            detail="An internal error occurred while registering the store."
        )
    

#--- Getting all stores which are active ----#
@app.get("/all/stores/",response_model=Store, tags=["Stores management"])
async def get_all_stores(current_user:Annotated[UserInDB,Depends(get_current_active_user)]):
    stores_collection=get_collection(STORES_COLLECTION)


    pass
    



    
    

    


