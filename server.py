from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models
class Attachment(BaseModel):
    filename: str
    data: str  # base64 encoded image data
    content_type: str

class ActivityConvertRequest(BaseModel):
    description: str
    attachments: Optional[List[Attachment]] = []

class ConvertedLog(BaseModel):
    title: str
    learning_area: str
    nesa_outcomes: List[str]
    skills_developed: List[str]
    reflection: str

class ActivityLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    learning_area: str
    nesa_outcomes: List[str]
    skills_developed: List[str]
    reflection: str
    attachments: Optional[List[Attachment]] = []
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_description: str

class ActivityLogCreate(BaseModel):
    title: str
    learning_area: str
    nesa_outcomes: List[str]
    skills_developed: List[str]
    reflection: str
    attachments: Optional[List[Attachment]] = []
    original_description: str

# Initialize LLM Chat
async def convert_activity_to_log(description: str) -> ConvertedLog:
    """Convert informal activity description to professional NESA-compliant log"""
    api_key = os.environ.get('EMERGENT_LLM_KEY')
    
    system_message = """You are an expert education documentation assistant specializing in NSW Education Standards Authority (NESA) requirements. 
Your task is to convert informal homeschool activity descriptions into professional, NESA-compliant learning logs.

You must return a JSON object with this exact structure:
{
    "title": "Brief subject-focused title (e.g., 'Mathematics, Science')",
    "learning_area": "Detailed description of what was learned",
    "nesa_outcomes": ["Outcome code 1", "Outcome code 2", "Outcome code 3"],
    "skills_developed": ["Skill 1", "Skill 2", "Skill 3"],
    "reflection": "A professional paragraph reflecting on the student's learning and progress"
}

NESA Outcome codes should follow the format: SUBJECT-STAGE (e.g., MA4-5NA for Mathematics Stage 4, EN4-2A for English Stage 4, HT4-6 for History Stage 4, SC4-8WS for Science Stage 4).

Skills should be specific, observable capabilities demonstrated during the activity.

The reflection should be professional, specific, and highlight the student's understanding and progress."""

    try:
        chat = LlmChat(
            api_key=api_key,
            session_id=f"convert-{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-5.2")
        
        user_message = UserMessage(
            text=f"Convert this student activity into a professional NESA-compliant learning log. Return ONLY valid JSON, no other text:\n\n{description}"
        )
        
        response = await chat.send_message(user_message)
        
        # Parse the JSON response
        import json
        # Clean up response if it has markdown code blocks
        response_text = response.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        log_data = json.loads(response_text)
        return ConvertedLog(**log_data)
        
    except Exception as e:
        logger.error(f"Error converting activity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error converting activity: {str(e)}")

# Routes
@api_router.get("/")
async def root():
    return {"message": "NESA Homeschool Documentation API"}

@api_router.post("/convert", response_model=ConvertedLog)
async def convert_activity(request: ActivityConvertRequest):
    """Convert informal activity description to professional log"""
    converted = await convert_activity_to_log(request.description)
    return converted

@api_router.post("/logs", response_model=ActivityLog)
async def save_log(input: ActivityLogCreate):
    """Save a converted log"""
    log_obj = ActivityLog(**input.model_dump())
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = log_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    await db.activity_logs.insert_one(doc)
    return log_obj

@api_router.get("/logs", response_model=List[ActivityLog])
async def get_logs():
    """Get all saved logs"""
    logs = await db.activity_logs.find({}, {"_id": 0}).sort("timestamp", -1).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for log in logs:
        if isinstance(log['timestamp'], str):
            log['timestamp'] = datetime.fromisoformat(log['timestamp'])
    
    return logs

@api_router.delete("/logs/{log_id}")
async def delete_log(log_id: str):
    """Delete a log"""
    result = await db.activity_logs.delete_one({"id": log_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Log not found")
    
    return {"message": "Log deleted successfully"}

@api_router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and return base64 encoded data"""
    try:
        contents = await file.read()
        base64_data = base64.b64encode(contents).decode('utf-8')
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "data": base64_data
        }
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
