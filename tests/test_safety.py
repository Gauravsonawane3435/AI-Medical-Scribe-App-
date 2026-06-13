import pytest
from app.services.generator import note_generator_service

def test_medication_without_dosage_multiple():
    """Verify that multiple medications prescribed without details are formatted in the bullet block structure."""
    transcript = "Doctor: I will prescribe Amoxicillin and Paracetamol."
    parsed = {
        "prescription": "Amoxicillin 500mg three times daily for 7 days\nParacetamol 500mg every 6 hours"
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    
    expected_output = (
        "* Amoxicillin\n"
        "* Paracetamol\n\n"
        "Dosage:\n"
        "* Not specified\n\n"
        "Frequency:\n"
        "* Not specified\n\n"
        "Duration:\n"
        "* Not specified"
    )
    assert sanitized["prescription"] == expected_output

def test_medication_without_dosage_single():
    """Verify that a single medication prescribed without details is formatted in the single medication block structure."""
    transcript = "Doctor: I will prescribe Amoxicillin."
    parsed = {
        "prescription": "Amoxicillin 500mg three times daily for 7 days"
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    
    expected_output = (
        "Medication:\n"
        "* Amoxicillin\n"
        "Dosage:\n"
        "Not specified\n"
        "Frequency:\n"
        "Not specified\n"
        "Duration:\n"
        "Not specified"
    )
    assert sanitized["prescription"] == expected_output

def test_partial_prescriptions():
    """Verify that medication details mentioned partially are preserved, and missing components are marked 'Not specified'."""
    transcript = "Doctor: Let's start you on Ibuprofen 400mg."
    parsed = {
        "prescription": "Ibuprofen 400mg twice daily for 5 days"
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    
    expected_output = (
        "Medication:\n"
        "* Ibuprofen\n"
        "Dosage:\n"
        "400mg\n"
        "Frequency:\n"
        "Not specified\n"
        "Duration:\n"
        "Not specified"
    )
    assert sanitized["prescription"] == expected_output

def test_medication_absent_in_transcript():
    """Verify that medications not present in the transcript are completely filtered out."""
    transcript = "Doctor: Take some Paracetamol."
    parsed = {
        "prescription": "Amoxicillin 500mg\nParacetamol"
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    
    assert "Amoxicillin" not in sanitized["prescription"]
    assert "Paracetamol" in sanitized["prescription"]

def test_missing_allergies_guard():
    """Verify that missing allergy details are removed and sanitized to 'Not specified'."""
    transcript = "Patient has a slight fever and cough."
    parsed = {
        "hpi": "Patient has a fever and cough. No known drug allergies.",
        "assessment": "Fever and cough."
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    assert "allergies" not in sanitized["hpi"].lower()
    assert "allergy" not in sanitized["hpi"].lower()

def test_missing_past_medical_history_guard():
    """Verify that past medical history is sanitized to 'Not specified' if not in transcript."""
    transcript = "Doctor: What brings you in today? Patient: Just this sore throat."
    parsed = {
        "hpi": "Patient presents with sore throat. Past medical history includes hypertension and diabetes."
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    assert "hypertension" not in sanitized["hpi"]
    assert "diabetes" not in sanitized["hpi"]

def test_missing_laboratory_results_guard():
    """Verify that laboratory values and tests not mentioned in transcript are sanitized to 'Not specified'."""
    transcript = "Doctor: We will check your blood pressure."
    parsed = {
        "recommended_tests": "Please order a CBC, BMP, and EKG."
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    assert "CBC" not in sanitized["recommended_tests"]
    assert "BMP" not in sanitized["recommended_tests"]
    assert "EKG" not in sanitized["recommended_tests"]
    assert sanitized["recommended_tests"] == "Not specified"

def test_strict_medical_extraction_and_grounding():
    """Verify that diagnoses and follow-ups are preserved when mentioned, and not converted to speculations."""
    transcript = (
        "Doctor: I think this is an upper respiratory tract infection. "
        "Doctor: Return if symptoms worsen."
    )
    parsed = {
        "assessment": "Upper respiratory tract infection",
        "follow_up": "Return if symptoms worsen"
    }
    sanitized = note_generator_service._validate_and_sanitize_note(parsed, transcript)
    
    assert sanitized["assessment"] == "Upper respiratory tract infection"
    assert sanitized["follow_up"] == "Return if symptoms worsen"
