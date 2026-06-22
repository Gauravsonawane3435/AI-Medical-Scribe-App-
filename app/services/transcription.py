import os
import logging
import mimetypes
import time
import subprocess
import tempfile
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

def convert_audio_to_wav_16k(audio_bytes: bytes) -> bytes:
    """
    Converts input audio bytes (e.g. webm/opus) to 16kHz mono PCM WAV format.
    Requires the ffmpeg executable on the system path.
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp_in:
        temp_in.write(audio_bytes)
        temp_in_name = temp_in.name
        
    temp_out_name = temp_in_name + ".wav"
    
    try:
        cmd = ["ffmpeg", "-y", "-i", temp_in_name, "-ar", "16000", "-ac", "1", "-f", "wav", temp_out_name]
        logger.info(f"[AudioConverter] Running command: {' '.join(cmd)}")
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=startupinfo,
            check=True
        )
        
        with open(temp_out_name, "rb") as f:
            wav_bytes = f.read()
            
        logger.info(f"[AudioConverter] Successfully converted audio to WAV. Size: {len(wav_bytes)} bytes")
        return wav_bytes
        
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "Unknown error"
        logger.error(f"[AudioConverter] ffmpeg failed with code {e.returncode}. stderr: {stderr_msg}")
        raise RuntimeError(f"ffmpeg conversion failed: {stderr_msg}")
    except FileNotFoundError:
        logger.error("[AudioConverter] ffmpeg executable not found on the system path.")
        raise RuntimeError("ffmpeg executable not found on the system path. Please ensure ffmpeg is installed.")
    finally:
        if os.path.exists(temp_in_name):
            try: os.remove(temp_in_name)
            except: pass
        if os.path.exists(temp_out_name):
            try: os.remove(temp_out_name)
            except: pass

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
        logger.info(f"[Backend] Processing ASR bytes: filename='{filename}', mime_type='{mime_type}', size={len(audio_bytes)} bytes, model='{model_id}'")
        
        # Check if the incoming file needs conversion to standard WAV (16kHz mono) for Hugging Face compatibility.
        # Native WAV and MP3 are supported directly. WebM, M4A, OGG, AAC, MP4, and CAF are converted to WAV if ffmpeg is available.
        needs_conversion = any(ext in mime_type.lower() for ext in ["webm", "m4a", "ogg", "aac", "mp4", "caf"]) or \
                           (filename and filename.lower().endswith((".webm", ".weba", ".m4a", ".ogg", ".aac", ".mp4", ".caf")))
        
        if needs_conversion:
            logger.info(f"[Backend] Audio format '{mime_type}' needs conversion. Attempting to convert to WAV (16kHz mono)...")
            try:
                converted_bytes = convert_audio_to_wav_16k(audio_bytes)
                audio_bytes = converted_bytes
                mime_type = "audio/wav"
                filename = "recording.wav"
                logger.info("[Backend] Conversion successful. Updated parameters to WAV.")
            except Exception as e:
                logger.warning(f"[Backend] Audio conversion to WAV failed: {e}. Falling back to raw audio upload with original MIME type.")
        
        # Guess/Map the best supported MIME type for Hugging Face compatibility.
        # Hugging Face supports audio/webm, audio/webm;codecs=opus, audio/ogg, audio/wav, and audio/m4a.
        # We only map truly unsupported types (like mp4, aac, caf) to m4a/wav, and guess if octet-stream is sent.
        hf_mime_type = mime_type
        if hf_mime_type == "application/octet-stream":
            if filename and filename.lower().endswith(".webm"):
                hf_mime_type = "audio/webm"
            elif filename and filename.lower().endswith((".m4a", ".mp4")):
                hf_mime_type = "audio/m4a"
            elif filename and filename.lower().endswith(".ogg"):
                hf_mime_type = "audio/ogg"
            else:
                hf_mime_type = "audio/wav"
                
        if hf_mime_type in ("audio/mp4", "audio/aac", "audio/caf", "audio/x-caf"):
            hf_mime_type = "audio/m4a"
            logger.info(f"[Backend] Mapping MIME type '{mime_type}' to '{hf_mime_type}' for Hugging Face routing compatibility.")
        else:
            logger.info(f"[Backend] Using MIME type '{hf_mime_type}' directly for Hugging Face routing.")

        try:
            # Set the Content-Type header and client timeout (60s) to support longer audio recordings
            client = InferenceClient(
                token=token,
                base_url="https://router.huggingface.co/hf-inference",
                headers={"Content-Type": hf_mime_type},
                timeout=60.0
            )
            # Build full URL to bypass api-inference.huggingface.co DNS resolution error
            model_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
            logger.info(f"[Backend] Connecting to: {model_url}")
            
            # Retry loop with exponential backoff for rate limits, model loading, and transient network errors
            max_retries = 3
            backoff_factor = 2
            base_delay = 1.0  # seconds
            
            response = None
            for attempt in range(max_retries + 1):
                try:
                    response = client.automatic_speech_recognition(audio_bytes, model=model_url)
                    break # Success!
                except (HfHubHTTPError, Exception) as e:
                    is_retryable = True
                    # Check if it is a permanent auth/validation issue (400, 401, 403)
                    if isinstance(e, HfHubHTTPError) and e.response is not None:
                        if e.response.status_code in (400, 401, 403):
                            is_retryable = False
                    
                    if isinstance(e, ValueError):
                        is_retryable = False
                        
                    if is_retryable and attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(f"[Backend] Retryable error hit during transcription: {e}. Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise e
            
            # The ASR response is usually a dictionary with {"text": "..."}
            transcript = ""
            if isinstance(response, dict) and "text" in response:
                transcript = response["text"]
            elif isinstance(response, str):
                transcript = response
            else:
                transcript = str(response)
                
            logger.info(f"[Backend] Transcription successful. Result preview: '{transcript[:100]}...'")
            return transcript
                
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
