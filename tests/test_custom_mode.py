import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Ensure test token is present
os.environ["HF_TOKEN"] = "test_token"

from app.main import app
from app.services.generator import note_generator_service

client = TestClient(app)

@patch("app.services.generator.note_generator_service.generate_note")
def test_custom_mode_api_success(mock_generate):
    """Test API generation endpoint handles the custom mode parameters correctly."""
    mock_generate.return_value = {
        "raw_note": "Here is the summary of consultation.",
        "model_used": "Qwen/Qwen2.5-72B-Instruct",
        "custom_output": "Here is the summary of consultation.",
        "mode": "custom"
    }

    request_data = {
        "transcript": "Patient came in complaining of migraine.",
        "model_key": "qwen",
        "mode": "custom",
        "custom_prompt": "Summarize this consultation",
        "hf_token": "test_token"
    }

    response = client.post("/api/generate-note", json=request_data)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["mode"] == "custom"
    assert res_json["output"] == "Here is the summary of consultation."
    assert "chief_complaint" not in res_json

    mock_generate.assert_called_once_with(
        transcript=request_data["transcript"],
        model_key="qwen",
        system_prompt=None,
        hf_token="test_token",
        mode="custom",
        custom_prompt="Summarize this consultation"
    )

def test_custom_mode_demo_mode_referral():
    """Verify that in Demo Mode, a request with 'referral' in custom_prompt returns the mock referral template."""
    transcript = "Doctor-patient talk about knee pain."
    # With a token like "demo", demo mode is active
    result = note_generator_service.generate_note(
        transcript=transcript,
        model_key="qwen",
        hf_token="demo",
        mode="custom",
        custom_prompt="Create a referral letter to orthopedics"
    )
    assert result["mode"] == "custom"
    assert "REFERRAL LETTER" in result["custom_output"]
    assert "knee pain" in result["custom_output"]

def test_custom_mode_demo_mode_general():
    """Verify that in Demo Mode, a general custom prompt request returns the custom placeholder template."""
    transcript = "Doctor-patient talk about blood test results."
    result = note_generator_service.generate_note(
        transcript=transcript,
        model_key="qwen",
        hf_token="demo",
        mode="custom",
        custom_prompt="Extract blood tests and results"
    )
    assert result["mode"] == "custom"
    assert "Custom Prompt Output (Demo Mode)" in result["custom_output"]
    assert "Extract blood tests" in result["custom_output"]

@patch("app.services.generator.InferenceClient")
def test_custom_mode_llm_client_call(mock_client_cls):
    """Verify that the generator builds custom messages correctly and calls InferenceClient in Custom mode."""
    # Mock inference client and return values
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Custom model output text about headache"
    mock_client.chat_completion.return_value = mock_response

    transcript = "Doctor: Hello, we discussed your headaches."
    
    result = note_generator_service.generate_note(
        transcript=transcript,
        model_key="qwen",
        hf_token="valid_token_123",  # triggers actual InferenceClient call
        mode="custom",
        custom_prompt="Summarize headaches"
    )
    
    assert result["mode"] == "custom"
    assert result["custom_output"] == "Custom model output text about headache"
    
    # Check that system and user messages were structured correctly
    mock_client.chat_completion.assert_called_once()
    call_kwargs = mock_client.chat_completion.call_args[1]
    
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    
    # System message
    assert messages[0]["role"] == "system"
    assert "professional medical AI assistant" in messages[0]["content"]
    assert "Follow the doctor's instruction exactly" in messages[0]["content"]
    
    # User message
    assert messages[1]["role"] == "user"
    assert "Doctor Instruction:\nSummarize headaches" in messages[1]["content"]
    assert f"Conversation:\n{transcript}" in messages[1]["content"]

def test_custom_mode_sanitization():
    """Verify that the validation layer sanitizes hallucinated details in the custom mode free text output."""
    transcript = "Doctor: I will prescribe Amoxicillin."
    
    # If the custom mode LLM output invents vital signs, dosages, or diagnoses not in transcript:
    hallucinated_text = (
        "Here is the summary:\n"
        "- Prescribed Amoxicillin 500mg daily.\n"
        "- Patient has blood pressure of 140/90.\n"
        "- Diagnosed with Cholelithiasis."
    )
    
    sanitized = note_generator_service._sanitize_general_section(hallucinated_text, transcript)
    
    # 500mg was not in the transcript, nor was 140/90, nor Cholelithiasis.
    # Check that they were replaced with "Not specified" (or filtered).
    assert "500mg" not in sanitized
    assert "140/90" not in sanitized
    assert "Cholelithiasis" not in sanitized
    
    # Amoxicillin was in transcript, so it should be preserved.
    assert "Amoxicillin" in sanitized

def test_raw_transcript_mode():
    """Verify that raw transcript mode returns only the speech-to-text output."""
    transcript = "This is a raw consultation conversation transcript."
    payload = {
        "transcript": transcript,
        "mode": "transcript"
    }
    response = client.post("/api/generate-note", json=payload)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["mode"] == "transcript"
    assert res_json["transcript"] == transcript

def test_empty_custom_prompt_validation():
    """Verify that empty custom prompt in custom mode triggers the correct error response."""
    payload = {
        "transcript": "Hello doctor.",
        "mode": "custom",
        "custom_prompt": "   "
    }
    response = client.post("/api/generate-note", json=payload)
    assert response.status_code == 400 or response.status_code == 500
    res_json = response.json()
    assert res_json["detail"] == "Custom prompt cannot be empty."
