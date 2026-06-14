import os
import logging
import mimetypes
from typing import Optional
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from app.config import settings, SUPPORTED_ASR_MODELS

logger = logging.getLogger(__name__)

def get_audio_mime_type(content_type: Optional[str] = None, filename: Optional[str] = None) -> str:
    """
    Detects and returns the correct audio MIME type.
    Defaults to 'audio/wav' if undetermined.
    """
    if content_type and content_type.startswith("audio/"):
        return content_type
        
    if filename:
        guessed_type, _ = mimetypes.guess_type(filename)
        if guessed_type and guessed_type.startswith("audio/"):
            return guessed_type
            
    # Fallback default
    return "audio/wav"

class TranscriptionService:
    def __init__(self):
        pass

    def transcribe_audio_bytes(
        self, 
        audio_bytes: bytes, 
        model_key: str = "whisper-large", 
        hf_token: Optional[str] = None,
        content_type: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Transcribes audio bytes using the Hugging Face serverless Inference API.
        Does not store the file on disk.
        """
        token = hf_token or settings.HF_TOKEN
        is_demo = not token or token.strip().lower() in ("", "demo", "test", "mock", "none", "hf_demo")
        
        if is_demo:
            logger.info("Demo mode active for transcription. Returning mock consultation text.")
            return (
                "Doctor: Good afternoon, please tell me what's been going on.\n"
                "Patient: Hello doctor, I've had a really bad headache for the past 3 days, especially behind my left eye. It throbs and feels very intense. I also feel nauseated if I look at bright lights.\n"
                "Doctor: I see. Did anything trigger it, and have you taken any medications?\n"
                "Patient: No specific trigger, maybe stress. I took some ibuprofen but it didn't help much. I had similar headaches years ago but they weren't this frequent.\n"
                "Doctor: It sounds like a severe migraine. I will prescribe Sumatriptan 50mg, to be taken immediately at the first sign of a migraine. You can repeat it after 2 hours if needed, but do not exceed 100mg in 24 hours. I also want you to rest in a dark, quiet room when it happens. Let's order a routine lipid panel and follow up in 2 weeks. If you experience sudden vision changes or neck stiffness, please go to the ER.\n"
                "Patient: Thank you, doctor. I'll get the prescription and follow your advice."
            )

        model_id = SUPPORTED_ASR_MODELS.get(model_key, SUPPORTED_ASR_MODELS["whisper-large"])["id"]
        
        # Detect and log correct audio MIME Type
        mime_type = get_audio_mime_type(content_type, filename)
        logger.info(f"[Backend] Processing ASR bytes: filename='{filename}', mime_type='{mime_type}', model='{model_id}'")
        
        # Hugging Face serverless router rejects some non-standard audio MIME types (like audio/mp4, audio/m4a, audio/aac).
        # We map these to application/octet-stream to allow Whisper's backend to decode them from container headers.
        hf_mime_type = mime_type
        if mime_type in ("audio/mp4", "audio/m4a", "audio/aac", "audio/x-caf", "audio/caf", "audio/x-m4a"):
            hf_mime_type = "application/octet-stream"
            logger.info(f"[Backend] Mapping MIME type '{mime_type}' to '{hf_mime_type}' for Hugging Face compatibility.")

        try:
            # Set the Content-Type header in the constructor to avoid None type rejection
            client = InferenceClient(
                token=token,
                base_url="https://router.huggingface.co/hf-inference",
                headers={"Content-Type": hf_mime_type}
            )
            # Build full URL to bypass api-inference.huggingface.co DNS resolution error
            model_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
            logger.info(f"[Backend] Connecting to: {model_url}")
            response = client.automatic_speech_recognition(audio_bytes, model=model_url)
            
            # The ASR response is usually a dictionary with {"text": "..."}
            if isinstance(response, dict) and "text" in response:
                return response["text"]
            elif isinstance(response, str):
                return response
            else:
                return str(response)
                
        except HfHubHTTPError as e:
            logger.error(f"Hugging Face Inference API error: {e}")
            if e.response is not None:
                logger.error(f"[Backend] HF Response Status Code: {e.response.status_code}")
                logger.error(f"[Backend] HF Response Headers: {dict(e.response.headers)}")
                logger.error(f"[Backend] HF Response Text: {e.response.text}")
                raise RuntimeError(f"Hugging Face API error (status {e.response.status_code}): {e.response.text}")
            raise RuntimeError(f"Hugging Face Hub error: {e.server_message or str(e)}")
        except Exception as e:
            logger.error(f"Unexpected transcription error: {e}")
            if "getaddrinfo failed" in str(e) or "NameResolutionError" in str(e):
                raise RuntimeError(
                    "Connection failed: Unable to resolve Hugging Face server. "
                    "Please verify your internet connection or DNS settings."
                )
            raise RuntimeError(f"Transcription failed: {str(e)}")

transcription_service = TranscriptionService()
