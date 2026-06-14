import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import os

from app.config import settings, SUPPORTED_LLM_MODELS, SUPPORTED_ASR_MODELS
from app.services.transcription import transcription_service
from app.services.generator import note_generator_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical AI Assistant API",
    description="REST API to transcribe consultations and generate structured clinical notes",
    version="1.0.0"
)

# Enable CORS for external clinical system integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure app directories exist
os.makedirs("app/static/css", exist_ok=True)
os.makedirs("app/static/js", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Request Models
class NoteGenerationRequest(BaseModel):
    transcript: str
    model_key: Optional[str] = "qwen"
    system_prompt: Optional[str] = None
    hf_token: Optional[str] = None
    mode: Optional[str] = "structured"
    custom_prompt: Optional[str] = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the frontend dashboard SPA."""
    template_path = os.path.join("app", "templates", "index.html")
    if os.path.exists(template_path):
        return FileResponse(template_path)
    return HTMLResponse("<h2>Frontend template index.html not found yet.</h2>")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Handles browser favicon requests gracefully."""
    return Response(status_code=204)

# API Route to get models
@app.get("/api/settings/models")
async def get_models():
    """Returns the list of supported LLM and ASR models."""
    return {
        "llm_models": SUPPORTED_LLM_MODELS,
        "asr_models": SUPPORTED_ASR_MODELS,
        "default_llm": settings.DEFAULT_LLM_KEY,
        "default_asr": settings.DEFAULT_ASR_KEY,
        "hf_token_configured": bool(settings.HF_TOKEN.strip())
    }

# API Route to get specialty templates
@app.get("/api/settings/specialties")
async def get_specialties():
    """Returns the list of supported clinical specialties and their prompts."""
    from app.config import SPECIALTY_TEMPLATES
    return SPECIALTY_TEMPLATES

# API Route to transcribe audio file
@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model_key: str = Form("whisper-large"),
    hf_token: Optional[str] = Form(None)
):
    """
    Transcribes uploaded audio files in memory.
    Supports MP3, WAV, M4A, MP4, FLAC, WEBM, AAC, OGG.
    No permanent storage is used.
    """
    filename = file.filename or ""
    content_type = file.content_type or ""
    ext = os.path.splitext(filename)[1].lower()
    
    logger.info(f"[Backend] Received transcription request. Filename: '{filename}', content_type: '{content_type}', extension: '{ext}', model: '{model_key}'")
    
    # Check if format is supported
    supported_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".webm", ".ogg", ".aac"}
    
    if not (content_type.startswith("audio/") or ext in supported_extensions):
        logger.warning(f"[Backend] File format rejected: ext='{ext}', content_type='{content_type}'")
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file type is not a supported audio format. Supported formats: MP3, WAV, M4A, MP4, FLAC, WEBM, AAC, OGG."
        )
        
    try:
        # Read bytes in memory
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
            
        transcript = transcription_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            model_key=model_key,
            hf_token=hf_token,
            content_type=content_type,
            filename=filename
        )
        return {"transcript": transcript}
    except ValueError as e:
        logger.error(f"[Backend] ValueError during transcription: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("An error occurred during audio transcription:")
        err_msg = str(e)
        if "Too Many Requests" in err_msg or "429" in err_msg or "rate limit" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: Too many requests. Please wait a minute and try again, or configure a custom Hugging Face Access Token in the Settings."
            )
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {err_msg}")

# API Route to generate structured clinical note
@app.post("/api/generate-note")
async def generate_note(request: NoteGenerationRequest):
    """
    Converts a doctor-patient conversation transcript into a structured clinical note, custom formatted text, or raw transcript.
    """
    transcript = request.transcript.strip()
    if not transcript:
        raise HTTPException(
            status_code=400,
            detail="Transcript content cannot be empty."
        )
        
    try:
        note_data = note_generator_service.generate_note(
            transcript=transcript,
            model_key=request.model_key or "qwen",
            system_prompt=request.system_prompt,
            hf_token=request.hf_token,
            mode=request.mode or "structured",
            custom_prompt=request.custom_prompt
        )
        
        mode = request.mode or "structured"
        if mode == "transcript":
            return {
                "mode": "transcript",
                "transcript": transcript
            }
        elif mode == "custom":
            return {
                "mode": "custom",
                "output": note_data.get("custom_output", note_data.get("raw_note", ""))
            }
        else: # structured
            return {
                "mode": "structured",
                "data": {
                    "chief_complaint": note_data.get("chief_complaint", ""),
                    "hpi": note_data.get("hpi", ""),
                    "assessment": note_data.get("assessment", ""),
                    "plan": note_data.get("plan", ""),
                    "prescription": note_data.get("prescription", ""),
                    "recommended_tests": note_data.get("recommended_tests", ""),
                    "follow_up": note_data.get("follow_up", ""),
                    "raw_note": note_data.get("raw_note", ""),
                    "model_used": note_data.get("model_used", "")
                }
            }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("An error occurred during clinical note generation:")
        err_msg = str(e)
        if "getaddrinfo failed" in err_msg or "NameResolutionError" in err_msg or "Connection failed" in err_msg:
            detail = "Unable to connect to AI service."
            raise HTTPException(status_code=500, detail=detail)
        elif "Too Many Requests" in err_msg or "429" in err_msg or "rate limit" in err_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: Too many requests. Please wait a minute and try again, or configure a custom Hugging Face Access Token in the Settings."
            )
        elif "'dict' object has no attribute 'strip'" in err_msg or "Invalid model response" in err_msg or "not supported as a serverless chat model" in err_msg:
            detail = "Invalid model response."
            raise HTTPException(status_code=500, detail=detail)
        elif "Custom prompt cannot be empty" in err_msg or "Instruction Empty" in err_msg:
            detail = "Custom prompt cannot be empty."
            raise HTTPException(status_code=400, detail=detail)
        else:
            detail = "Failed to generate the clinical note. Please try again."
            raise HTTPException(status_code=500, detail=detail)

