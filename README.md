# Project Title (Assumed: FastAPI and Milvus RAG Project)

## Description

This project appears to be a backend application built with FastAPI, designed to handle user authentication, data storage (MongoDB), and a Retrieval Augmented Generation (RAG) system using Milvus as a vector database. The RAG pipeline seems focused on extracting information about herbal medicines and their scientific evidence. It also includes components for managing e-commerce like features such as stores and products. The `GNN.ipynb` notebook suggests an exploration or experimentation phase related to Graph Neural Networks, potentially for analyzing interactions between biomolecules, proteins, pathways, and diseases, though it encounters some dependency issues.

## File Descriptions

Below is a brief description of each file in the project:

* **`main.py`**:
    * The main FastAPI application file.
    * Handles API routing, request/response validation, and integrates various modules.
    * Includes endpoints for user authentication (token generation), user management (creation, fetching), store management, and product management (commented out).
    * Integrates the RAG pipeline, allowing topics to be processed and queries to be made against the Milvus vector database.
    * Contains an endpoint (`/ayur-doct`) to interact with an Ollama model named "ayur-doctor".
    * Manages application lifecycle events like connecting to and closing MongoDB and Milvus connections.

* **`auth.py`**:
    * Manages authentication logic.
    * Includes functions for creating and verifying JWT access tokens.
    * Provides an OAuth2PasswordBearer scheme for token-based authentication.
    * Contains functions to authenticate users against the database and to get the current active user from a token.
    * Relies on `SECRET_KEY`, `ALGORITHM`, and `ACCESS_TOKEN_EXPIRE_MINUTES` from environment variables.

* **`data_extractor.py`**:
    * Seems to be a script for setting up and running a Langchain agent to gather information.
    * Uses `TavilySearchResults` as a tool for the agent.
    * Configures a `ChatGroq` LLM (e.g., "llama3-8b-8192").
    * The agent is designed to answer questions and has a prompt template for structured thought, action, and observation steps.
    * It processes a predefined topic: "Different herbal medicines for different diseases having authenticated scientific evidence."
    * The collected data is then intended for processing (splitting) and potentially embedding for a vector database.

* **`database.py`**:
    * Handles MongoDB database connections.
    * Uses `motor` (async MongoDB driver).
    * Reads `MONGODB_URL` and `DATABASE_NAME` from environment variables.
    * Provides functions to connect to MongoDB, close the connection, and get a database collection instance.
    * Includes `certifi` for TLS/SSL connections, often needed for MongoDB Atlas.

* **`embedding.py`**:
    * Manages the generation of text embeddings.
    * Uses the `sentence-transformers` library, with a default model like "all-MiniLM-L6-v2" (configurable via `EMBEDDING_MODEL_NAME` environment variable).
    * Loads the embedding model and provides its dimension.
    * Includes a function `get_embedding(text: str)` to generate a vector for a given text.
    * Contains commented-out placeholders for using OpenAI embeddings.

* **`GNN.ipynb`**:
    * A Jupyter Notebook for exploring Graph Neural Networks.
    * Initial comments suggest nodes representing biomolecules, proteins, pathways, and diseases, with edges as interactions (e.g., activates, inhibits).
    * Includes cells for installing dependencies like `rdkit-pypi`, `requests`, `pandas`, `openpyxl`, `tqdm`.
    * The notebook shows errors during the installation of `rdkit-pypi` with Python 3.12, indicating a missing compatible wheel. It also shows issues with NumPy version compatibility when trying to import `rdkit.Chem`.
    * Further cells seem to involve fetching compound data (e.g., using PubChem PUG REST API for CIDs like 969516, 5280343, 445154, 5280445, 5280863, 65064, 370, 9064) and processing it, likely for GNN input. The output shows keys from fetched data like 'AID', 'CID', 'Activity Outcome', etc.

* **`milvusdb_config.py`**:
    * Configures and initializes a connection to a Milvus vector database.
    * Defines connection parameters (`MILVUS_HOST`, `MILVUS_PORT`, `CONNECTION_ALIAS`).
    * Defines the schema for a Milvus collection, including fields for `chunk_id` (primary key, auto-generated), `embedding` (float vector), `text_chunk` (varchar), and `source_document` (varchar). The `VECTOR_DIMENSION` is set to 768.
    * Checks if the collection exists and creates it if not.
    * Creates an index (e.g., `IVF_FLAT` or HNSW if specified, here it defaults to `AUTOINDEX` or checks for a pre-existing one) on the embedding field for efficient similarity search.
    * Loads the collection into memory.
    * Includes example code for preparing and inserting data into the collection (text chunks, source documents, and their embeddings).

* **`models1.py`**:
    * Defines Pydantic models for data validation, serialization, and API request/response structures.
    * Includes models for:
        * `UserCreate`, `UserPublic`, `UserInDB`, `UserUpdate`: User-related data and authentication.
        * `Token`, `TokenData`: JWT token structure.
        * `Store`, `StoreCreate`: E-commerce store information.
        * `Products`, `ProductsCreate`: Product information.
        * `TopicList`, `ProcessResponse`: For handling lists of topics to be processed by the RAG system.
        * `QueryRequest`, `QueryResponse`, `RetrievedChunk`: For querying the RAG system and returning results.
    * Defines custom types like `PyObjectId` for MongoDB ObjectId handling, `PasswordStr`, `UsernameStr`, and enums like `UserType`.

* **`pipeline.py`**:
    * Defines the core RAG pipeline logic.
    * Loads various configurations from environment variables (LLM model, embedding model, API keys, chunking parameters, Milvus settings).
    * Provides functions to initialize:
        * LLM (e.g., `ChatGroq`).
        * Embedding model (e.g., `SentenceTransformer`).
        * Langchain agent executor with tools like `TavilySearchResults`.
        * Text splitter (`RecursiveCharacterTextSplitter`).
    * Includes `setup_milvus_collection` to connect to Milvus and ensure the collection with the correct schema and index exists.
    * `run_pipeline_for_topic` function orchestrates the process:
        1.  Data Extraction: Uses the agent to research a given topic.
        2.  Text Splitting: Splits the extracted text into manageable chunks.
        3.  Embedding Generation: Creates vector embeddings for each chunk.
        4.  Data Preparation: Structures the data for Milvus insertion.
        5.  Storage: Inserts the data (embeddings, chunks, source documents) into the specified Milvus collection.
    * Uses extensive logging for tracking the pipeline's progress and any errors.

* **`security.py`**:
    * Provides utility functions for password security.
    * Uses `passlib` for password hashing.
    * `verify_password(plain_password, hashed_password)`: Verifies a plain password against its hashed version.
    * `get_password_hash(password)`: Generates a bcrypt hash for a given password.

## Setup and Installation

1.  **Environment Variables**:
    Create a `.env` file in the root directory and populate it with the necessary credentials and configurations. Based on the files, you'll likely need:
    * `MONGODB_URL`: MongoDB connection string.
    * `DATABASE_NAME`: Name of the MongoDB database.
    * `SECRET_KEY`: A secret key for JWT token generation.
    * `ALGORITHM`: JWT algorithm (e.g., "HS256").
    * `ACCESS_TOKEN_EXPIRE_MINUTES`: Expiration time for access tokens.
    * `EMBEDDING_MODEL_NAME`: Name of the sentence transformer model (e.g., "all-MiniLM-L6-v2" or "sentence-transformers/all-mpnet-base-v2").
    * `LLM_MODEL_NAME`: (If applicable for `pipeline.py` directly, though `data_extractor.py` and `main.py` might have their own LLM configs)
    * `TAVILY_API_KEY`: API key for Tavily search.
    * `GROQ_API_TOKEN`: API key for Groq.
    * `CHUNK_SIZE`: Size of text chunks for the RAG pipeline.
    * `CHUNK_OVERLAP`: Overlap between text chunks.
    * `MILVUS_HOST`: Host for the Milvus server (default: "localhost").
    * `MILVUS_PORT`: Port for the Milvus server (default: "19530").
    * `COLLECTION_NAME`: Name for the Milvus collection (e.g., "herbal_evidence_rag").
    * `INDEX_FIELD_NAME`: Field name for indexing in Milvus (usually "embedding").
    * `VECTOR_DIMENSION`: Dimension of the embeddings (e.g., 384 for "all-MiniLM-L6-v2", 768 for "all-mpnet-base-v2" or if specified in `milvusdb_config.py`).
    * `INDEX_TYPE`: Milvus index type (e.g., "IVF_FLAT", "HNSW").
    * `METRIC_TYPE`: Milvus metric type for similarity search (e.g., "L2", "IP").

2.  **Python Dependencies**:
    Install the required Python packages. You'll likely need a `requirements.txt` file, but based on the imports, key dependencies include:
    * `fastapi`
    * `uvicorn` (for running FastAPI)
    * `pydantic` (and `pydantic-settings`)
    * `python-jose[cryptography]` (for JWT)
    * `passlib[bcrypt]` (for password hashing)
    * `motor` (async MongoDB driver)
    * `pymilvus` (Milvus client)
    * `langchain`
    * `langchain-core`
    * `langchain-community`
    * `langchain-groq`
    * `langchain-huggingface`
    * `sentence-transformers`
    * `tavily-python` (for Tavily search tool)
    * `python-dotenv`
    * `httpx`
    * `certifi`
    * `ollama` (if using local Ollama models directly with `langchain_community.llms`)
    * For `GNN.ipynb`: `requests`, `pandas`, `openpyxl`, `tqdm`, and potentially `rdkit-pypi` (though it has installation issues noted in the notebook for Python 3.12).

    ```bash
    pip install fastapi uvicorn pydantic pydantic-settings python-jose[cryptography] passlib[bcrypt] motor pymilvus langchain langchain-core langchain-community langchain-groq langchain-huggingface sentence-transformers tavily-python python-dotenv httpx certifi ollama requests pandas openpyxl tqdm
    ```
    *(Note: This is a comprehensive list based on imports; a `requirements.txt` would be more precise.)*

3.  **Setup External Services**:
    * **MongoDB**: Ensure you have a running MongoDB instance (local or Atlas) and the connection URL is set in `.env`.
    * **Milvus**: Ensure you have a running Milvus instance and its host/port are configured. The `milvusdb_config.py` script can be used to set up the initial connection and collection schema if run independently, or this logic might be incorporated into the main application startup.
    * **Ollama** (Optional): If using the `/ayur-doct` endpoint, ensure Ollama is running and the "ayur-doctor" model is available.

## Running the Application

1.  **Start FastAPI Server**:
    ```bash
    uvicorn main:app --reload
    ```
    The application will typically be available at `http://127.0.0.1:8000`.

2.  **API Endpoints**:
    Refer to `main.py` for available API endpoints. Key functionalities include:
    * `/token`: User login and token generation.
    * User registration and management.
    * Store management.
    * `/process-topics/`: To initiate the RAG pipeline for a list of topics.
    * `/query-rag/`: To ask questions to the RAG system.
    * `/ayur-doct`: To interact with the local "ayur-doctor" Ollama model.

## RAG Pipeline (`pipeline.py` and `data_extractor.py`)

* The system is designed to:
    1.  Take a research topic.
    2.  Use a Langchain agent (`TavilySearchResults`) to gather information from the web.
    3.  Split the gathered text into chunks.
    4.  Generate embeddings for these chunks using a SentenceTransformer model.
    5.  Store these chunks and their embeddings in a Milvus collection for similarity search.
* The `main.py` application provides endpoints to trigger this pipeline for new topics and to query the indexed data.

## GNN Exploration (`GNN.ipynb`)

* This notebook is for experimenting with Graph Neural Networks.
* It attempts to use `rdkit` for chemical informatics, but faces installation/compatibility issues with Python 3.12 and NumPy 2.0.
    * Users might need to create a specific Python environment (e.g., Python 3.9, 3.10, or 3.11 as suggested by `rdkit-pypi` error messages) and manage NumPy versions (e.g., `numpy<2`) to run this notebook successfully.
* It fetches data from PubChem for specific compound IDs.

## Key Modules & Functionality

* **Authentication (`auth.py`, `security.py`):** JWT-based authentication with password hashing.
* **Database (`database.py`):** Async MongoDB interaction for storing application data (users, stores, etc.).
* **Vector Embeddings (`embedding.py`):** Generation of text embeddings using sentence transformers.
* **Vector Database (`milvusdb_config.py`, `pipeline.py`):** Milvus setup, schema definition, and data ingestion for RAG.
* **Data Models (`models1.py`):** Pydantic models for API data structures and validation.
* **RAG Core (`pipeline.py`, `data_extractor.py`):** Orchestrates data extraction, processing, embedding, and storage for the RAG system.
* **API (`main.py`):** FastAPI application providing various endpoints.

## Notes and Considerations

* Ensure all environment variables are correctly set up before running the application.
* The Milvus collection and schema need to be properly initialized. The `pipeline.py` or a dedicated script like `milvusdb_config.py` (if run separately) handles this.
* The `GNN.ipynb` notebook has specific dependency challenges that need to be addressed if you intend to use it.
* The product-related endpoints in `main.py` are commented out and would need to be completed if product management is a required feature.
