import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.generator import note_generator_service

client = TestClient(app)

def test_parse_note_sections_dynamic_and_synonyms():
    """Verify that non-standard fields are parsed dynamically and synonyms are mapped."""
    raw_note = (
        "Chief Complaint:\nSevere migraine headache.\n\n"
        "Findings:\nPatient has bilateral headaches and throbbing pain. No aura.\n\n"
        "Treatment:\nTake rest in dark room and stay hydrated.\n\n"
        "Past Surgeries:\nCholecystectomy in 2020.\n\n"
        "Allergies:\nSulfa drugs."
    )
    
    sections = note_generator_service._parse_note_sections(raw_note)
    
    # Check standard fields are there (initialized or mapped)
    assert sections.get("chief_complaint") == "Severe migraine headache."
    
    # Check synonym mapping
    # Findings should map to assessment
    assert sections.get("assessment") == "Patient has bilateral headaches and throbbing pain. No aura."
    # Treatment should map to plan
    assert sections.get("plan") == "Take rest in dark room and stay hydrated."
    
    # Check dynamic/custom fields are parsed correctly
    assert sections.get("past_surgeries") == "Cholecystectomy in 2020."
    assert sections.get("allergies") == "Sulfa drugs."
    
    # Check unused standard fields are empty string
    assert sections.get("hpi") == ""
    assert sections.get("prescription") == ""
    assert sections.get("recommended_tests") == ""
    assert sections.get("follow_up") == ""

def test_dynamic_sanitization():
    """Verify that custom fields are dynamically sanitized against the transcript."""
    transcript = "Patient has a history of cholecystectomy in 2020. She has allergies to sulfa drugs."
    
    # Case 1: Custom fields contain grounded information
    sections = {
        "past_surgeries": "Cholecystectomy in 2020.",
        "allergies": "Sulfa drugs.",
        "prescription": ""
    }
    
    sanitized = note_generator_service._validate_and_sanitize_note(sections, transcript)
    assert sanitized["past_surgeries"] == "Cholecystectomy in 2020."
    assert sanitized["allergies"] == "Sulfa drugs."

    # Case 2: Custom fields contain ungrounded information (should be stripped/modified)
    sections_ungrounded = {
        "past_surgeries": "Cholecystectomy in 2020 and Appendectomy in 2018.", # Appendectomy 2018 not in transcript
        "allergies": "Sulfa drugs and Penicillin allergy.",                       # Penicillin not in transcript
        "prescription": ""
    }
    
    sanitized_ungrounded = note_generator_service._validate_and_sanitize_note(sections_ungrounded, transcript)
    # Appendectomy 2018 has a number/medical keyword and was not in the transcript, so it should be stripped
    assert "Appendectomy" not in sanitized_ungrounded["past_surgeries"]
    # Penicillin should be stripped
    assert "Penicillin" not in sanitized_ungrounded["allergies"]

@patch("app.services.generator.note_generator_service.generate_note")
def test_api_dynamic_fields_response(mock_generate):
    """Verify that the API returns the 7 standard keys and dynamically includes custom keys."""
    mock_generate.return_value = {
        "raw_note": "Mocked raw note",
        "model_used": "mocked-model",
        "chief_complaint": "Severe migraine.",
        "hpi": "",
        "assessment": "Migraine headache.",
        "plan": "Rest.",
        "prescription": "",
        "recommended_tests": "",
        "follow_up": "",
        "past_surgeries": "Cholecystectomy in 2020.",
        "allergies": "Sulfa drugs.",
        "mode": "structured"
    }
    
    request_data = {
        "transcript": "Patient has a history of cholecystectomy in 2020. She has allergies to sulfa drugs. Complains of severe migraine.",
        "model_key": "qwen"
    }
    
    response = client.post("/api/generate-note", json=request_data)
    assert response.status_code == 200
    res_json = response.json()
    
    assert res_json["mode"] == "structured"
    clinical_note = res_json["clinical_note"]
    
    # 7 standard keys must always be returned
    for key in ["chief_complaint", "hpi", "assessment", "plan", "prescription", "recommended_tests", "follow_up"]:
        assert key in clinical_note
        
    # Metadata keys
    assert clinical_note["raw_note"] == "Mocked raw note"
    assert clinical_note["model_used"] == "mocked-model"
    
    # Custom keys must be dynamically included
    assert clinical_note["past_surgeries"] == "Cholecystectomy in 2020."
    assert clinical_note["allergies"] == "Sulfa drugs."
