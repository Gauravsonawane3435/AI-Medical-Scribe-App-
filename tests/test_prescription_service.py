import pytest
from unittest.mock import patch, MagicMock
from app.services.prescription_service import prescription_service

def test_parse_note_prescription_local_single():
    """Verify that local parsing extracts single medication formatting correctly."""
    prescription_text = (
        "Medication:\n"
        "* Amoxicillin\n"
        "Dosage:\n"
        "500mg\n"
        "Frequency:\n"
        "Twice Daily\n"
        "Duration:\n"
        "7 Days"
    )
    result = prescription_service.parse_note_prescription_local(prescription_text)
    assert "medications" in result
    assert len(result["medications"]) == 1
    med = result["medications"][0]
    assert med["name"] == "Amoxicillin"
    assert med["dosage"] == "500mg"
    assert med["frequency"] == "Twice Daily"
    assert med["duration"] == "7 Days"

def test_parse_note_prescription_local_multiple():
    """Verify that local parsing extracts multiple medication blocks correctly."""
    prescription_text = (
        "Medication:\n"
        "* Amoxicillin\n"
        "Dosage:\n"
        "500mg\n"
        "Frequency:\n"
        "Twice Daily\n"
        "Duration:\n"
        "7 Days\n\n"
        "Medication:\n"
        "* Ibuprofen\n"
        "Dosage:\n"
        "400mg\n"
        "Frequency:\n"
        "Three times daily\n"
        "Duration:\n"
        "5 Days"
    )
    result = prescription_service.parse_note_prescription_local(prescription_text)
    assert "medications" in result
    assert len(result["medications"]) == 2
    assert result["medications"][0]["name"] == "Amoxicillin"
    assert result["medications"][1]["name"] == "Ibuprofen"
    assert result["medications"][1]["dosage"] == "400mg"

def test_parse_note_prescription_local_parallel():
    """Verify that local parsing extracts parallel lists correctly (when details are bulk-specified)."""
    prescription_text = (
        "* Amoxicillin\n"
        "* Paracetamol\n\n"
        "Dosage:\n"
        "* Not specified\n\n"
        "Frequency:\n"
        "* Not specified\n\n"
        "Duration:\n"
        "* Not specified"
    )
    result = prescription_service.parse_note_prescription_local(prescription_text)
    assert "medications" in result
    assert len(result["medications"]) == 2
    assert result["medications"][0]["name"] == "Amoxicillin"
    assert result["medications"][1]["name"] == "Paracetamol"
    assert result["medications"][0]["dosage"] == "Not specified"

@pytest.mark.anyio
@patch("app.services.prescription_service.InferenceClient")
async def test_extract_prescription_json_hf_api(mock_inference_client):
    """Verify that HF completions are invoked when a valid token is provided and Sonar is not configured."""
    mock_client = MagicMock()
    mock_inference_client.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = (
        "{\n"
        "  \"medications\": [\n"
        "    {\n"
        "      \"name\": \"Amoxicillin\",\n"
        "      \"dosage\": \"500mg\",\n"
        "      \"frequency\": \"Twice Daily\",\n"
        "      \"duration\": \"7 Days\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    mock_client.chat_completion.return_value = mock_response

    note_data = {
        "prescription": "Amoxicillin 500mg twice daily for 7 days"
    }

    result = await prescription_service.extract_prescription_json(
        transcript="Prescribe Amoxicillin.",
        note_data=note_data,
        model_key="qwen",
        hf_token="real_token_123"
    )

    assert "medications" in result
    assert len(result["medications"]) == 1
    assert result["medications"][0]["name"] == "Amoxicillin"
    mock_client.chat_completion.assert_called_once()
