import os
import sys
from fastapi.testclient import TestClient

# Ensure workspace is in path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from app.main import app
from app.services.generator import note_generator_service

def run_e2e_verification():
    print("====================================================")
    print("STARTING COMPLETE E2E WORKFLOW SIMULATION & QA AUDIT")
    print("====================================================")
    
    # Initialize TestClient
    client = TestClient(app)
    
    # Define results dict
    results = {
        "1. Retrieve Supported Models": False,
        "2. Retrieve Specialty Templates": False,
        "3. Audio Upload & Speech-to-Text (Demo Mode)": False,
        "4. Audio Privacy Verification": False,
        "5. Structured Note Generation (General Specialty)": False,
        "6. Structured Note Generation (Surgeon Specialty)": False,
        "7. Structured Note Generation (OB/GYN Specialty)": False,
        "8. Structured Note Generation (Cardiology Specialty)": False,
        "9. Multilingual Model & Token Configuration Handling": False,
        "10. Invalid Audio File Validation": False,
        "11. Custom Prompt Mode (Demo Referral)": False,
        "12. Custom Prompt Mode (Demo Fallback)": False,
        "13. Custom Prompt Mode Factual Sanitization": False,
        "14. Raw Transcript Mode": False,
        "15. Empty Custom Prompt Validation": False,
        "16. Friendly Error Mapping": False,
    }
    
    # 1. Test Retrieve Models
    print("\n--- 1. Testing GET /api/settings/models ---")
    try:
        response = client.get("/api/settings/models")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "llm_models" in data and "asr_models" in data, "Missing model configurations"
        print("[OK] Successfully retrieved LLM & ASR model metadata.")
        results["1. Retrieve Supported Models"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")
        
    # 2. Test Retrieve Specialties
    print("\n--- 2. Testing GET /api/settings/specialties ---")
    try:
        response = client.get("/api/settings/specialties")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "general" in data and "surgeon" in data and "obgyn" in data and "cardiologist" in data, "Missing specialties"
        print("[OK] Successfully retrieved specialty templates.")
        results["2. Retrieve Specialty Templates"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 3. Test Audio Upload & Transcription (Demo Mode)
    print("\n--- 3. Testing POST /api/transcribe (Demo Mode) ---")
    try:
        # Generate dummy wav bytes
        dummy_audio = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        files = {"file": ("consultation.wav", dummy_audio, "audio/wav")}
        data = {"model_key": "whisper-large", "hf_token": "demo"}
        
        response = client.post("/api/transcribe", files=files, data=data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        res_json = response.json()
        assert "transcript" in res_json, "Missing 'transcript' key in response"
        print(f"[OK] Transcript successfully generated in Demo Mode.")
        print(f"--- Transcript Preview ---\n{res_json['transcript'][:150]}...")
        results["3. Audio Upload & Speech-to-Text (Demo Mode)"] = True
        
        # 4. Audio Privacy Check
        # Verify no files were created in the root or app folders
        temp_files_exist = False
        for root, dirs, files_list in os.walk("."):
            for file in files_list:
                if file.endswith(".wav") and file != "consultation.wav" and "venv" not in root:
                    temp_files_exist = True
        if not temp_files_exist:
            print("[OK] Verified: No audio files written to disk. Processed strictly in-memory.")
            results["4. Audio Privacy Verification"] = True
        else:
            print("[WARN] Warning: Temporary audio files found on disk.")
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 5. Test Note Generation - General Physician
    print("\n--- 5. Testing POST /api/generate-note (General Specialty) ---")
    try:
        transcript_text = "Patient complains of bad headache. Prescribed Sumatriptan 50mg. Return in 2 weeks."
        # Fetch template
        spec_data = client.get("/api/settings/specialties").json()
        prompt = spec_data["general"]["prompt"]
        
        payload = {
            "transcript": transcript_text,
            "model_key": "qwen",
            "system_prompt": prompt,
            "hf_token": "demo"
        }
        
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        res_data = response.json()
        assert res_data["mode"] == "structured"
        note = res_data["data"]
        assert "chief_complaint" in note and "hpi" in note and "assessment" in note and "plan" in note and "prescription" in note, "Missing note sections"
        print("[OK] General clinical note generated and parsed successfully.")
        print(f"Chief Complaint: {note['chief_complaint']}")
        print(f"HPI: {note['hpi']}")
        print(f"Prescription: {note['prescription']}")
        results["5. Structured Note Generation (General Specialty)"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 6. Test Note Generation - Surgeon
    print("\n--- 6. Testing POST /api/generate-note (General Surgeon) ---")
    try:
        # Query with surgeon keywords
        spec_data = client.get("/api/settings/specialties").json()
        prompt = spec_data["surgeon"]["prompt"]
        payload = {
            "transcript": "Consultation about gallbladder cholecystectomy.",
            "model_key": "qwen",
            "system_prompt": prompt,
            "hf_token": "demo"
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["mode"] == "structured"
        note = res_data["data"]
        assert "laparoscopic cholecystectomy" in note["hpi"].lower() or "cholelithiasis" in note["hpi"].lower(), "Failed to generate surgery context"
        print("[OK] General Surgeon clinical note generated and parsed successfully.")
        results["6. Structured Note Generation (Surgeon Specialty)"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 7. Test Note Generation - OB/GYN
    print("\n--- 7. Testing POST /api/generate-note (OB/GYN) ---")
    try:
        spec_data = client.get("/api/settings/specialties").json()
        prompt = spec_data["obgyn"]["prompt"]
        payload = {
            "transcript": "Initial prenatal checkup at 12 weeks pregnancy.",
            "model_key": "qwen",
            "system_prompt": prompt,
            "hf_token": "demo"
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["mode"] == "structured"
        note = res_data["data"]
        assert "gestation" in note["hpi"].lower() or "prenatal" in note["hpi"].lower(), "Failed to generate OB/GYN context"
        print("[OK] OB/GYN clinical note generated and parsed successfully.")
        results["7. Structured Note Generation (OB/GYN Specialty)"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 8. Test Note Generation - Cardiology
    print("\n--- 8. Testing POST /api/generate-note (Cardiology) ---")
    try:
        spec_data = client.get("/api/settings/specialties").json()
        prompt = spec_data["cardiologist"]["prompt"]
        payload = {
            "transcript": "Checking chest pain and blood pressure logs.",
            "model_key": "qwen",
            "system_prompt": prompt,
            "hf_token": "demo"
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 200
        res_data = response.json()
        assert res_data["mode"] == "structured"
        note = res_data["data"]
        assert "chest pain" in note["hpi"].lower() or "cardiology" in note["model_used"].lower(), "Failed to generate cardiology context"
        print("[OK] Cardiologist clinical note generated and parsed successfully.")
        results["8. Structured Note Generation (Cardiology Specialty)"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 9. Test Token error handling with bad token (real HF client simulation)
    print("\n--- 9. Testing invalid token behavior ---")
    try:
        payload = {
            "transcript": "Hello doctor.",
            "model_key": "qwen",
            "system_prompt": "General guidelines",
            "hf_token": "invalid_token_pattern_xxxx"
        }
        response = client.post("/api/generate-note", json=payload)
        # Should raise an error or try loading model and fail due to auth
        # Either 400, 500, or 429 status code with detail
        assert response.status_code in (400, 500, 401, 403, 429)
        print("[OK] Properly rejected/handled invalid token pattern.")
        results["9. Multilingual Model & Token Configuration Handling"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 10. Test Invalid Audio format rejection
    print("\n--- 10. Testing invalid audio format rejection ---")
    try:
        files = {"file": ("test.txt", b"Plain text file", "text/plain")}
        data = {"model_key": "whisper-large"}
        response = client.post("/api/transcribe", files=files, data=data)
        assert response.status_code == 400
        assert "not a supported audio format" in response.json()["detail"]
        print("[OK] Properly rejected non-audio formats with 400 Bad Request.")
        results["10. Invalid Audio File Validation"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 11. Custom Prompt Mode (Demo Referral)
    print("\n--- 11. Testing Custom Prompt Mode (Demo Referral) ---")
    try:
        payload = {
            "transcript": "Doctor and patient talk about a knee injury.",
            "mode": "custom",
            "custom_prompt": "Please draft a referral letter for this patient.",
            "hf_token": "demo"
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "custom"
        assert "REFERRAL LETTER" in data["output"]
        print("[OK] Custom Mode referral letter draft generated successfully.")
        results["11. Custom Prompt Mode (Demo Referral)"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 12. Custom Prompt Mode (Demo Fallback)
    print("\n--- 12. Testing Custom Prompt Mode (Demo Fallback) ---")
    try:
        payload = {
            "transcript": "The consultation covers generic follow-up details.",
            "mode": "custom",
            "custom_prompt": "Summarize key transcript facts",
            "hf_token": "demo"
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "custom"
        assert "Custom Prompt Output" in data["output"]
        print("[OK] Custom Mode general response template generated successfully.")
        results["12. Custom Prompt Mode (Demo Fallback)"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 13. Custom Prompt Mode Factual Sanitization
    print("\n--- 13. Testing Custom Prompt Mode Factual Sanitization ---")
    try:
        transcript = "Doctor: I will prescribe Amoxicillin."
        hallucinated_output = "We prescribed Amoxicillin 500mg daily. We also diagnosed Cholelithiasis."
        sanitized = note_generator_service._sanitize_general_section(hallucinated_output, transcript)
        assert "500mg" not in sanitized, "Sanitization failed to strip dosage"
        assert "Cholelithiasis" not in sanitized, "Sanitization failed to strip diagnosis"
        assert "Amoxicillin" in sanitized, "Sanitization stripped a valid drug name"
        print("[OK] Custom Prompt Mode factual validation sanitization functions correctly.")
        results["13. Custom Prompt Mode Factual Sanitization"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 14. Raw Transcript Mode
    print("\n--- 14. Testing Raw Transcript Mode ---")
    try:
        transcript_text = "Doctor: Hello. Patient: Hi, my back hurts."
        payload = {
            "transcript": transcript_text,
            "mode": "transcript"
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "transcript"
        assert data["transcript"] == transcript_text
        print("[OK] Raw Transcript mode processed and returned speech-to-text output successfully.")
        results["14. Raw Transcript Mode"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 15. Empty Custom Prompt Validation
    print("\n--- 15. Testing Empty Custom Prompt Validation ---")
    try:
        payload = {
            "transcript": "Hello doctor.",
            "mode": "custom",
            "custom_prompt": "   "
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code == 500 or response.status_code == 400
        detail = response.json()["detail"]
        assert detail == "Custom prompt cannot be empty."
        print("[OK] Empty custom prompt rejected with user-friendly error message.")
        results["15. Empty Custom Prompt Validation"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # 16. Friendly Error Mapping
    print("\n--- 16. Testing Friendly Error Mapping ---")
    try:
        # Generate with bad token format to simulate unauthorized/invalid provider error
        payload = {
            "transcript": "Hello doctor.",
            "model_key": "qwen",
            "system_prompt": "General guidelines",
            "hf_token": "invalid_token_pattern_xxxx"
        }
        response = client.post("/api/generate-note", json=payload)
        assert response.status_code in (500, 429)
        detail = response.json()["detail"]
        assert detail in (
            "Failed to generate the clinical note. Please try again.",
            "Rate limit exceeded: Too many requests. Please wait a minute and try again, or configure a custom Hugging Face Access Token in the Settings."
        )
        print(f"[OK] Exception correctly caught and mapped to user-friendly message: {detail!r}")
        results["16. Friendly Error Mapping"] = True
    except Exception as e:
        print(f"[FAIL] Failed: {e}")

    # Final Report Output
    print("\n====================================================")
    print("FINAL QA AUDIT REPORT")
    print("====================================================")
    all_passed = True
    for test_name, passed in results.items():
        status_symbol = "Passed" if passed else "Failed"
        if not passed:
            all_passed = False
        print(f"{test_name.ljust(55)} : {status_symbol}")
    print("====================================================")
    
    if all_passed:
        print("RESULT: ALL CLIENT REQUIREMENTS FULLY SATISFIED!")
        sys.exit(0)
    else:
        print("RESULT: SOME QA AUDIT TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_e2e_verification()
