import pytest
from unittest.mock import patch, MagicMock
from app.services.transcript_validator import transcript_validator_service

def test_rule_based_cleanup_repetitions():
    """Verify that repeated Whisper silence-generated loops are cleaned correctly."""
    raw = "Hello doctor. Thank you. Thank you. Thank you. Patient has pain."
    cleaned = transcript_validator_service.rule_based_cleanup(raw)
    assert cleaned == "Hello doctor. Thank you. Patient has pain."

def test_rule_based_cleanup_homophones():
    """Verify that common STT/ASR clinical misspellings and homophones are mapped correctly."""
    raw = "Start the patient on sir tech and paracetol."
    cleaned = transcript_validator_service.rule_based_cleanup(raw)
    assert "Zyrtec" in cleaned
    assert "paracetamol" in cleaned

def test_rule_based_cleanup_duplicate_consecutive_words():
    """Verify that consecutive duplicated words are stripped."""
    raw = "The the patient has pain in in the back."
    cleaned = transcript_validator_service.rule_based_cleanup(raw)
    assert cleaned == "The patient has pain in the back."

@patch("app.services.transcript_validator.InferenceClient")
def test_validate_and_clean_api_success(mock_inference_client):
    """Verify that when HF API token is provided, it calls InferenceClient and returns corrected output."""
    mock_client = MagicMock()
    mock_inference_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Corrected clinical transcript details"
    mock_client.chat_completion.return_value = mock_response
    
    res = transcript_validator_service.validate_and_clean(
        raw_transcript="sir tech and paracetol",
        model_key="qwen",
        hf_token="real_token_123"
    )
    assert res == "Corrected clinical transcript details"
    mock_client.chat_completion.assert_called_once()

def test_validate_and_clean_demo_mode():
    """Verify that in demo mode, validator bypasses LLM and returns rule-cleaned output."""
    res = transcript_validator_service.validate_and_clean(
        raw_transcript="sir tech and paracetol",
        model_key="qwen",
        hf_token="demo"
    )
    assert "Zyrtec" in res
    assert "paracetamol" in res
