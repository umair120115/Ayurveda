# app/models.py
from pydantic import BaseModel, Field, EmailStr, ConfigDict, SecretStr, StringConstraints,field_validator,ValidationError,WithJsonSchema
from typing import Optional, List, Annotated
from bson import ObjectId
from datetime import datetime,timezone
import enum
from datetime import date ,datetime,time,timezone
import phonenumbers
import uuid


# --- ObjectId Helper ---
# (Your PyObjectId class from the file is correct)
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls): yield cls.validate
    @classmethod
    def validate(cls, v, _):
        if not ObjectId.is_valid(v): raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.json_or_python_schema(
            python_schema=core_schema.with_info_plain_validator_function(cls.validate),
            json_schema=core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda x: str(x)),)

# --- Custom Types from your file ---
PasswordStr = Annotated[SecretStr, StringConstraints(min_length=8)]
UsernameStr = Annotated[str, StringConstraints(min_length=3, max_length=50)]

class UserType(str, enum.Enum):
    STUDENT = 'student'
    PROFESSIONAL = 'professional'
    RESEARCHER = 'researcher'
    BUSSINESS = 'bussiness' # Typo? Consider 'BUSINESS'

# --- Standard User Models (Adapted from your UserModel) ---

class UserBase(BaseModel):
    # Fields common to user creation and representation
    username: UsernameStr
    email: EmailStr    #user's email
    name: str = Field(description='Name of the user.') #user's name
    phone: Optional[int] = None  #uswr's phone number
    user_type: UserType #type of user
    is_admin: bool = False # Default admin=False


# This model basically inherit UserBase and then adding password field. This model will be used for user registration
class UserCreate(UserBase):
    password: PasswordStr # Expects plain password input (as SecretStr)


# Model reflecting the structure stored in MongoDB
class UserInDB(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    hashed_password: str # Hashed password stored here
    registered_at: datetime = Field(default_factory=datetime.now)
    

    model_config = ConfigDict(
        populate_by_name=True, # Allows reading MongoDB's '_id' field
        arbitrary_types_allowed=True, # Allows custom types like ObjectId
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()} # How to serialize to JSON
    )
#Model which wil return the details 
class UserPublic(BaseModel):
    # Model defining data safe to return in API responses (no password hash)
    id: PyObjectId = Field(alias="_id")
    username: UsernameStr
    email: EmailStr
    name: str
    phone: Optional[int] = None
    user_type: UserType
    registered_at: datetime
    is_admin: bool
    

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()}
    )

class UserUpdate(BaseModel):
    # Model defining fields allowed for updating via PATCH /users/me/
    
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    phone: Optional[int] = None
    user_type: Optional[UserType] = None
    


    # Exclude sensitive fields like password, is_admin, username from profile update cannot update username!

    model_config = ConfigDict(
        extra='forbid' # Prevent other fields from being sent in update payload kind of validation
    )

# --- Token Models (Needed for Authentication) ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel): # Internal model for JWT payload data
    username: Optional[str] = None

# --- Document/Embedding Models (Keep if needed for Vector Search) ---
# You might need models like DocumentChunkCreate, DocumentChunkInDB, SearchQuery here as well
# if you implement the vector search endpoints in the same application.
# Define them based on previous examples if required.
import uuid

class StoreType(str,enum.Enum):
    """ Different types of stores or types of bussiness partners we are going to link."""
    RESTUARANT='restuarant'
    DHABA='dhaba'
    STORE='store'
    SWEETS_STORE='sweets-store'
    VEGETABLE_STORE='vegetable-store'
    DAIRY='dairy'
    CLOTHES_STORE='clothes-store'
    GROCERY_STORE='grocery-store'
class Store(BaseModel):
    storeid: uuid.UUID=Field(default_factory=uuid.uuid4, ) #creating unique storeid and it is required
    name: str
    email: EmailStr
    store_type: StoreType = Field(default=StoreType.RESTUARANT,description="Type of bussiness they are running and type of their product they are selling.") #type of store it is
    address: str
    store_lat: float
    store_lang: float
    postalcode: str
    owner_id: PyObjectId # Correctly typed required field  --> need to be confirmed!
    created_at: datetime=Field(default_factory=lambda : datetime.now(timezone.utc))  #now it will match with the type hint 
    opening_time: time 
    closing_time: time 
    is_active: bool=True
    store_description: str
    contact: str

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()}
    )

# PyObjectId= Annotated[ObjectId,WithJsonSchema({'type': 'string', 'example': '662508e9318b9a3e4f7b4e83'})]  #This tells that this field will be used as _id like primary key
class StoreCreate(BaseModel):
    name: str = Field(description="Name of the store/restuarant.")
    email : EmailStr=Field(description="Unique email for store.")
    store_type: StoreType
    address: str = Field(description="Address of the store.")
    store_lat: float = Field(description="Latitude of the store.")
    store_lang: float = Field(description="Longitude of store.")
    postalcode: str = Field(description="Postal code of the store.")
    opening_time: time = Field(description="Opening time...")
    closing_time: time = Field(description="Closing time...")
    store_description: str = Field(description="Brief description...")
    contact: str = Field(description="Contact phone number...")

    @field_validator('contact')
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """Validate the phone number using the phonenumbers library."""
        if not v:
            raise ValueError('Phone number cannot be empty')
        try:
            # Attempt to parse the number.
            # You might want to specify a default region (e.g., 'IN' for India based on your location)
            # This helps parse numbers that aren't in full E.164 format (e.g., local numbers)
            # If you expect numbers from anywhere, you might omit region or handle it dynamically.
            parsed_number = phonenumbers.parse(v, "IN") # Using "IN" for India as default region

            # Check if the number is possible and valid
            if not phonenumbers.is_possible_number(parsed_number):
                 raise ValueError(f"'{v}' is not a possible phone number format.")
            if not phonenumbers.is_valid_number(parsed_number):
                raise ValueError(f"'{v}' is not a valid phone number.")

            # Optionally, format to a standard format like E.164 before storing
            # E.164 format is '+[country code][subscriber number]' e.g., +919876543210
            formatted_number = phonenumbers.format_number(
                parsed_number,
                phonenumbers.PhoneNumberFormat.E164
            )
            return formatted_number

        except phonenumbers.NumberParseException as e:
            # Re-raise as ValueError for Pydantic
            raise ValueError(f"Failed to parse phone number '{v}': {e}") from e
        except Exception as e:
             # Catch any other unexpected errors during validation
            raise ValueError(f"An error occurred during phone number validation for '{v}': {e}") from e






    model_config = ConfigDict(
        populate_by_name=True, # Allows reading MongoDB's '_id' field
        arbitrary_types_allowed=True, # Allows custom types like ObjectId
        json_encoders={ObjectId: str, datetime: lambda dt: dt.isoformat()} # How to serialize to JSON
    )

class Products(BaseModel):
    productid : uuid.UUID = Field(default_factory=uuid.uuid4)
    product_name: str
    store: PyObjectId
    price: float
    discounted_price: float
    description: str
    stock: int
    image_url:str
    
    pass





#---  RAG and Centralized DB ---#

class TopicList(BaseModel):
    topics: List[str] = Field(description="A list of research topics to process.")
    # Optional: Add a field to specify the source, if needed
    # source_identifier: Optional[str] = Field(default=None, description="Identifier for the source of these topics")

class ProcessResponse(BaseModel):
    message: str
    topics_received: int

class QueryRequest(BaseModel):
    query: str = Field(description="The user's question to ask the RAG system.")
    top_k: int = Field(default=3, description="Number of relevant chunks to retrieve.")

class RetrievedChunk(BaseModel):
    text: str
    source: Optional[str] = None
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: Optional[List[RetrievedChunk]] = None