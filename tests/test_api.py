import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set up test environment
import os
os.environ["HF_TOKEN"] = "test_token"

from app.main import app

client = TestClient(app)

def test_get_models():
    """Test the models configuration retrieval endpoint."""
    response = client.get("/api/settings/models")
    assert response.status_code == 200
    data = response.json()
    assert "llm_models" in data
    assert "asr_models" in data
    assert data["default_llm"] == "qwen"
    assert data["default_asr"] == "whisper-large"
    assert "qwen" in data["llm_models"]

@patch("app.services.transcription.transcription_service.transcribe_audio_bytes")
def test_transcribe_audio_success(mock_transcribe):
    """Test successful audio upload transcription with mocked ASR service."""
    mock_transcribe.return_value = "Hello doctor, I have a headache."
    
    # Send a mock file
    file_content = b"fake audio bytes"
    files = {"file": ("test.wav", file_content, "audio/wav")}
    data = {"model_key": "whisper-large", "hf_token": "test_token"}
    
    response = client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 200
    assert response.json() == {"transcript": "Hello doctor, I have a headache."}
    mock_transcribe.assert_called_once_with(
        audio_bytes=file_content,
        model_key="whisper-large",
        hf_token="test_token",
        content_type="audio/wav",
        filename="test.wav"
    )

def test_transcribe_invalid_file_type():
    """Test that non-audio files are rejected with appropriate error."""
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    data = {"model_key": "whisper-large"}
    response = client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 400
    assert "not a supported audio format" in response.json()["detail"]

@patch("app.services.generator.note_generator_service.generate_note")
def test_generate_note_success(mock_generate):
    """Test successful clinical note generation with mocked generator service."""
    mock_generate.return_value = {
        "raw_note": "Chief Complaint:\nHeadache.\n\nHPI:\nPatient has headache.\n\nAssessment:\nMigraine.\n\nPlan:\nRest.",
        "model_used": "Qwen/Qwen2.5-72B-Instruct",
        "chief_complaint": "Headache.",
        "hpi": "Patient has headache.",
        "assessment": "Migraine.",
        "plan": "Rest.",
        "prescription": "",
        "recommended_tests": "",
        "follow_up": ""
    }
    
    request_data = {
        "transcript": "Hello doctor, I have a headache. I think it is migraine.",
        "model_key": "qwen",
        "system_prompt": "Custom system prompt",
        "hf_token": "test_token"
    }
    
    response = client.post("/api/generate-note", json=request_data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["mode"] == "structured"
    note_data = res_json["data"]
    assert note_data["chief_complaint"] == "Headache."
    assert note_data["hpi"] == "Patient has headache."
    assert note_data["assessment"] == "Migraine."
    assert note_data["plan"] == "Rest."
    assert note_data["prescription"] == ""
    assert note_data["model_used"] == "Qwen/Qwen2.5-72B-Instruct"
    
    mock_generate.assert_called_once_with(
        transcript=request_data["transcript"],
        model_key="qwen",
        system_prompt="Custom system prompt",
        hf_token="test_token",
        mode="structured",
        custom_prompt=None
    )

def test_generate_note_empty_transcript():
    """Test note generation with empty transcript fails validation."""
    request_data = {
        "transcript": "   ",
        "model_key": "qwen"
    }
    response = client.post("/api/generate-note", json=request_data)
    assert response.status_code == 422 or response.status_code == 400

@patch("app.services.generator.note_generator_service.generate_note")
def test_generate_note_rate_limit_429(mock_generate):
    """Verify that a 429 Too Many Requests error from Hugging Face is mapped to HTTP 429."""
    mock_generate.side_effect = RuntimeError("Hugging Face Hub error: 429 Client Error: Too Many Requests for url")
    
    request_data = {
        "transcript": "Hello doctor.",
        "model_key": "qwen"
    }
    response = client.post("/api/generate-note", json=request_data)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]

@patch("app.services.transcription.transcription_service.transcribe_audio_bytes")
def test_transcribe_rate_limit_429(mock_transcribe):
    """Verify that a 429 Rate Limit error during transcription is mapped to HTTP 429."""
    mock_transcribe.side_effect = RuntimeError("Hugging Face Hub error: 429 Client Error: Too Many Requests for url")
    
    file_content = b"fake audio bytes"
    files = {"file": ("test.wav", file_content, "audio/wav")}
    data = {"model_key": "whisper-large"}
    
    response = client.post("/api/transcribe", files=files, data=data)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]

@patch("app.services.transcription.convert_audio_to_wav_16k")
@patch("app.services.transcription.InferenceClient")
def test_webm_transcribe_fallback(mock_inference_client, mock_convert_wav):
    """Verify that when ffmpeg fails during WebM transcription, we fallback to application/octet-stream."""
    # Mock ffmpeg failure
    mock_convert_wav.side_effect = FileNotFoundError("ffmpeg not found on the system path.")
    
    # Mock InferenceClient and response
    mock_client_instance = MagicMock()
    mock_client_instance.automatic_speech_recognition.return_value = "Mocked WebM output transcript"
    mock_inference_client.return_value = mock_client_instance
    
    from app.services.transcription import transcription_service
    
    # Run the transcription service directly
    res = transcription_service.transcribe_audio_bytes(
        audio_bytes=b"fake webm bytes",
        model_key="whisper-large",
        hf_token="test_hf_token",
        content_type="audio/webm",
        filename="recording.webm"
    )
    
    # Verify the output transcript matches
    assert res == "Mocked WebM output transcript"
    
    # Verify client was constructed with Content-Type application/octet-stream
    mock_inference_client.assert_called_once()
    kwargs = mock_inference_client.call_args[1]
    assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
