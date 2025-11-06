import os
import time
import logging # Use logging instead of print for better tracking
from dotenv import load_dotenv

# Langchain / Data Extraction Imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Milvus Imports
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Load Environment Variables
load_dotenv()
logging.info("Loaded environment variables.")

# --- Configuration (can be accessed via os.getenv directly where needed) ---
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")
# LLM_MODEL_NAME='llama3'
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", 768)) # Make sure this matches the embedding model
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
CONNECTION_ALIAS = os.getenv("CONNECTION_ALIAS", "default")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")

# --- Component Initialization Functions ---
LLAMA_URL=os.getenv('OLAMA_URL')
import httpx
async def initialize_llm():
    """Initializes and returns the ChatGroq LLM."""
    # if not LLAMA_URL:
        # raise ValueError("Inference server not found! Try running in OLAMA using command ollama pull llama3.")
    # Set environment variable for the library if needed
    os.environ["GROQ_API_KEY"] = GROQ_API_TOKEN
    try:
        llm = ChatGroq(temperature=0, model_name=LLM_MODEL_NAME, streaming=True)  #initialize LLM
        return llm
        # playload={
        #     "model":"llama3",
        #     "prompt":"What's your name?",
        #     "stream":False
        # }
        async with httpx.AsyncClient() as client:
            response= await client
        # response= await httpx.AsyncClient().post(LLAMA_URL,json=playload)
        # result = response.json()
        
        # logging.info(f"Initialized LLM: {LLM_MODEL_NAME}- status-{result.get('response')}")

        # return result.get('response')
    except Exception as e:
        logging.error(f"Error initializing LLM: {e}")
        raise




def initialize_embedding_model():
    """Initializes and returns the Sentence Transformer embedding model."""
    try:
        # 1. Load the model directly with sentence-transformers to check dimension
        logging.info(f"Loading SentenceTransformer: {EMBEDDING_MODEL_NAME} for dimension check...")
        st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        actual_dim = st_model.get_sentence_embedding_dimension()
        logging.info(f"Actual model dimension: {actual_dim}")

        # 2. Verify dimension against configuration
        if actual_dim != VECTOR_DIMENSION:
            logging.warning(
                f"Model dimension ({actual_dim}) != configured dimension ({VECTOR_DIMENSION}). "
                "Ensure VECTOR_DIMENSION in your config matches the model."
            )
            # Decide how critical this mismatch is for your application
            # raise ValueError("Embedding dimension mismatch")

        # 3. Initialize the LangChain wrapper (it will load the model again internally,
        #    or you might pass the loaded st_model if the wrapper supports it, check docs)
        logging.info(f"Initializing LangChain HuggingFaceEmbeddings wrapper...")
        lc_embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            # model_kwargs={'device': 'cuda'} # Optional: specify device
            # encode_kwargs={'normalize_embeddings': False} # Optional: encoding args
        )
        logging.info(f"Initialized Embedding Model Wrapper: {EMBEDDING_MODEL_NAME}")
        return lc_embedder

    except Exception as e:
        logging.error(f"Error initializing Embedding Model: {e}")
        raise

# Example Usage:
try:
    embedding_model = initialize_embedding_model()
    # Now use embedding_model.embed_query(...) or embedding_model.embed_documents(...)
except Exception as e:
    logging.error(f"Failed to get embedding model: {e}")
def initialize_agent_executor(llm):
    """Initializes and returns the Langchain Agent Executor."""
    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")
    tavily_tool = TavilySearchResults(max_results=5)
    tools = [tavily_tool]
    logging.info("Initialized Tavily Search Tool.")

    # Recreate prompt template (ensure it's exactly as needed)
    prompt_template_str = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (provide the raw information gathered)

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""
    prompt = PromptTemplate.from_template(prompt_template_str)
    try:
        agent = create_react_agent(llm, tools, prompt)
        # Set verbose=False for less console noise when run via API
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
        logging.info("Initialized Langchain Agent Executor.")
        return agent_executor
    except Exception as e:
        logging.error(f"Error initializing Agent Executor: {e}")
        raise

def initialize_text_splitter():
    """Initializes and returns the RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    logging.info(f"Initialized Text Splitter (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")
    return splitter

# --- Milvus Setup Function (Keep as is, or slightly adapt logging) ---
def setup_milvus_collection() -> Collection:
    """Connects to Milvus and ensures the collection exists and is loaded."""
    try:
        if not connections.has_connection(CONNECTION_ALIAS):
             connections.connect(
                alias=CONNECTION_ALIAS,
                host=MILVUS_HOST,
                port=MILVUS_PORT
            )
        logging.info(f"Successfully connected/verified connection to Milvus instance at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        raise ConnectionError(f"Could not connect to Milvus: {e}") from e

    # Define Schema (Ensure VECTOR_DIMENSION is correct)
    field_id = FieldSchema(name='chunk_id', dtype=DataType.INT64, is_primary=True, auto_id=True)
    field_vector = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
    field_text_chunk = FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=65535) # Increased size
    field_source_doc = FieldSchema(name="source_document", dtype=DataType.VARCHAR, max_length=1024)
    schema = CollectionSchema(
      fields=[field_id, field_vector, field_text_chunk, field_source_doc],
      description="Collection for RAG document chunks",
      enable_dynamic_field=False
    )

    # Get or Create Collection
    if utility.has_collection(COLLECTION_NAME, using=CONNECTION_ALIAS):
        logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
        collection = Collection(name=COLLECTION_NAME, using=CONNECTION_ALIAS)
    else:
        logging.info(f"Creating collection '{COLLECTION_NAME}'...")
        collection = Collection(
            name=COLLECTION_NAME,
            schema=schema,
            using=CONNECTION_ALIAS,
            consistency_level="Bounded" # Good default
        )
        logging.info("Collection created successfully.")

        # --- Create Index ---
        INDEX_FIELD_NAME = "embedding"
        index_params = {
          "metric_type": "L2", # Or "IP" - Should match embedding model's optimal metric
          "index_type": "HNSW",
          "params": {"M": 16, "efConstruction": 256}
        }
        logging.info(f"Creating index for field '{INDEX_FIELD_NAME}'...")
        try:
            collection.create_index(
              field_name=INDEX_FIELD_NAME,
              index_params=index_params,
              index_name=f"{INDEX_FIELD_NAME}_idx"
            )
            logging.info("Index creation initiated. Waiting for completion...")
            utility.wait_for_index_building_complete(COLLECTION_NAME, using=CONNECTION_ALIAS)
            logging.info("Index building complete.")
        except Exception as e:
            logging.error(f"Error creating index: {e}")
            # Decide if you want to raise error or continue without index
            # raise

    # Load collection into memory for searching
    # Check if loaded - avoid reloading if already loaded
    load_state = utility.load_state(COLLECTION_NAME, using=CONNECTION_ALIAS)
    if load_state != "Loaded":
        logging.info(f"Loading collection '{COLLECTION_NAME}' into memory...")
        collection.load()
        utility.wait_for_loading_complete(COLLECTION_NAME, using=CONNECTION_ALIAS)
        logging.info("Collection loaded successfully.")
    else:
         logging.info(f"Collection '{COLLECTION_NAME}' already loaded.")

    return collection

# --- Modified Pipeline Function (Accepts initialized components) ---
def run_pipeline_for_topic(
    topic: str,
    milvus_collection: Collection,
    agent_executor_instance: AgentExecutor, # Pass instance
    text_splitter_instance: RecursiveCharacterTextSplitter, # Pass instance
    embeddings_model_instance: HuggingFaceEmbeddings # Pass instance
    ) -> bool:
    """Executes the data extraction, processing, embedding, and storage for a single topic."""
    logging.info(f"\n--- Starting Pipeline for Topic: '{topic}' ---")

    # 1. Extract Data using Agent
    logging.info(f"\n1. Researching topic: '{topic}'...")
    try:
        # Use the passed agent_executor instance
        search_results = agent_executor_instance.invoke({"input": topic})
        collected_data = search_results.get('output')
        print(f"{collected_data}\n")
        if not collected_data or not isinstance(collected_data, str):
            logging.warning("   Agent did not return valid string data ('output' field). Skipping topic.")
            return False
        logging.info("   Data extraction complete.")
       
    except Exception as e:
        logging.error(f"   Error during agent execution for topic '{topic}': {e}")
        return False

    # 2. Process Data (Split into Chunks)
    logging.info("\n2. Splitting data into chunks...")
    try:
        # Use the passed text_splitter instance
        chunks = text_splitter_instance.split_text(collected_data)
        logging.info(f"   Split into {len(chunks)} chunks.")
        if not chunks:
            logging.warning("   No chunks were created. Skipping topic.")
            return False
    except Exception as e:
        logging.error(f"   Error during text splitting for topic '{topic}': {e}")
        return False

    # 3. Generate Embeddings
    logging.info(f"\n3. Generating embeddings using '{EMBEDDING_MODEL_NAME}'...")
    try:
        start_time = time.time()
        # Use the passed embeddings_model instance
        chunk_embeddings = embeddings_model_instance.embed_documents(chunks)
        end_time = time.time()
        logging.info(f"   Generated {len(chunk_embeddings)} embeddings in {end_time - start_time:.2f} seconds.")

        if len(chunk_embeddings) != len(chunks):
             logging.error(f"   Error: Mismatch between chunks ({len(chunks)}) and embeddings ({len(chunk_embeddings)}). Skipping topic.")
             return False
    except Exception as e:
        logging.error(f"   Error generating embeddings for topic '{topic}': {e}")
        return False

    # 4. Prepare Data for Milvus
    logging.info("\n4. Preparing data for Milvus insertion...")
    source_documents = [topic] * len(chunks) # Use the current topic as the source identifier
    # Ensure the order matches the schema defined in setup_milvus_collection
    # Schema: [field_id, field_vector, field_text_chunk, field_source_doc]
    # We provide data for: field_vector, field_text_chunk, field_source_doc (id is auto)
    data_to_insert = [chunk_embeddings, chunks, source_documents]
    logging.info(f"   Prepared {len(chunks)} entities for insertion.")

    # 5. Store in Milvus
    logging.info("\n5. Storing data in Milvus...")
    try:
        # Use the passed milvus_collection instance
        insert_result = milvus_collection.insert(data_to_insert)
        logging.info(f"   Insertion result: {insert_result}")
        logging.info(f"   Successfully inserted {insert_result.insert_count} entities.")

        # Optional: Flush data immediately
        start_flush = time.time()
        milvus_collection.flush()
        end_flush = time.time()
        logging.info(f"   Data flushed in {end_flush - start_flush:.2f} seconds.")

    except Exception as e:
        logging.error(f"   Error inserting data into Milvus for topic '{topic}': {e}")
        return False

    logging.info(f"\n--- Pipeline Finished Successfully for Topic: '{topic}' ---")
    return True

# --- Remove or comment out the old execution block ---
# if __name__ == "__main__":
#     # ... (old code for standalone execution) ...
#     pass