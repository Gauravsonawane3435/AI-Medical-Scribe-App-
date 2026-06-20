import logging
import re
import os
import json
from typing import Optional, Dict, Any, List
import httpx
from huggingface_hub import InferenceClient
from app.config import settings, SUPPORTED_LLM_MODELS

logger = logging.getLogger(__name__)

class PrescriptionService:
    def __init__(self):
        pass

    def parse_note_prescription_local(self, prescription_text: str) -> Dict[str, Any]:
        """
        Fallback parser that extracts structured medications from raw note prescription text
        using regular expressions and string matching.
        """
        medications: List[Dict[str, str]] = []
        if not prescription_text or prescription_text.lower().strip() in ("", "none", "not specified", "n/a"):
            return {"medications": medications}

        # 1. Parse structured blocks:
        # Medication:
        # * Amoxicillin
        # Dosage:
        # 500mg
        # Frequency:
        # Twice Daily
        # Duration:
        # 7 Days
        blocks = re.split(r"\bMedication:\s*", prescription_text)
        if len(blocks) > 1:
            for block in blocks[1:]:
                lines = block.strip().split("\n")
                name = "Not specified"
                dosage = "Not specified"
                frequency = "Not specified"
                duration = "Not specified"

                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith("*"):
                        name = line_stripped.replace("*", "").strip()
                
                # Extract sections within block
                dosage_match = re.search(r"Dosage:\s*\n?\*?\s*([^\n]+)", block, re.IGNORECASE)
                if dosage_match:
                    dosage = dosage_match.group(1).strip()
                
                frequency_match = re.search(r"Frequency:\s*\n?\*?\s*([^\n]+)", block, re.IGNORECASE)
                if frequency_match:
                    frequency = frequency_match.group(1).strip()
                
                duration_match = re.search(r"Duration:\s*\n?\*?\s*([^\n]+)", block, re.IGNORECASE)
                if duration_match:
                    duration = duration_match.group(1).strip()

                if name != "Not specified" or dosage != "Not specified":
                    medications.append({
                        "name": name,
                        "dosage": dosage,
                        "frequency": frequency,
                        "duration": duration
                    })
            return {"medications": medications}

        # 2. Parse parallel list format:
        # * MedicationA
        # * MedicationB
        # Dosage:
        # * Not specified
        # Frequency:
        # * Not specified
        # etc.
        lines = [line.strip() for line in prescription_text.split("\n") if line.strip()]
        names = []
        for line in lines:
            if line.startswith("*") and not any(kw in line.lower() for kw in ["dosage:", "frequency:", "duration:", "not specified"]):
                names.append(line.replace("*", "").strip())
        
        # Check if list of medications is found
        if names:
            dosage = "Not specified"
            frequency = "Not specified"
            duration = "Not specified"

            dosage_match = re.search(r"Dosage:\s*\n?\*?\s*([^\n]+)", prescription_text, re.IGNORECASE)
            if dosage_match:
                dosage = dosage_match.group(1).strip()
            
            frequency_match = re.search(r"Frequency:\s*\n?\*?\s*([^\n]+)", prescription_text, re.IGNORECASE)
            if frequency_match:
                frequency = frequency_match.group(1).strip()
            
            duration_match = re.search(r"Duration:\s*\n?\*?\s*([^\n]+)", prescription_text, re.IGNORECASE)
            if duration_match:
                duration = duration_match.group(1).strip()

            for name in names:
                medications.append({
                    "name": name,
                    "dosage": dosage,
                    "frequency": frequency,
                    "duration": duration
                })
            return {"medications": medications}

        # 3. Last fallback: parse simple single-line entries
        # e.g., "Amoxicillin 500mg twice daily for 7 days"
        for line in prescription_text.split("\n"):
            line_stripped = line.strip("*- ")
            if not line_stripped:
                continue
            words = line_stripped.split()
            if len(words) >= 1:
                name = words[0]
                dosage_match = re.search(r"\b\d+\s*(?:mg|ml|g|mcg|units)\b", line_stripped, re.IGNORECASE)
                dosage = dosage_match.group(0) if dosage_match else "Not specified"
                
                freq_match = re.search(r"\b(?:daily|nightly|once|twice|three|four|every|prn)\b.*", line_stripped, re.IGNORECASE)
                frequency = freq_match.group(0) if freq_match else "Not specified"
                
                dur_match = re.search(r"\bfor\s+\d+\s*(?:days|weeks|months)\b", line_stripped, re.IGNORECASE)
                duration = dur_match.group(0) if dur_match else "Not specified"
                
                medications.append({
                    "name": name,
                    "dosage": dosage,
                    "frequency": frequency,
                    "duration": duration
                })

        return {"medications": medications}

    def _clean_json_response(self, text: str) -> Dict[str, Any]:
        """
        Cleans the markdown formatting or raw code blocks around JSON from model outputs
        and parses it safely.
        """
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end+1])
                except Exception:
                    pass
            raise ValueError(f"Unable to parse structured JSON from text: {text}")

    async def call_sonar_api(
        self,
        api_key: str,
        transcript: str,
        note_prescription: str
    ) -> Dict[str, Any]:
        """
        Attempts to call the Perplexity Sonar API to extract the structured prescription.
        """
        base_url = (os.getenv("SONAR_BASE_URL") or os.getenv("PERPLEXITY_BASE_URL") or "https://api.perplexity.ai").rstrip("/")
        model = os.getenv("SONAR_MODEL") or os.getenv("PERPLEXITY_MODEL") or "sonar-pro"

        system_prompt = (
            "You are a clinical extraction assistant. Extract all prescribed medications from the transcript and clinical notes.\n"
            "Return JSON only in this schema:\n"
            "{\n"
            "  \"medications\": [\n"
            "    {\n"
            "      \"name\": \"string (e.g. Amoxicillin)\",\n"
            "      \"dosage\": \"string (e.g. 500mg, 10ml, or 'Not specified')\",\n"
            "      \"frequency\": \"string (e.g. Twice Daily, every 8 hours, or 'Not specified')\",\n"
            "      \"duration\": \"string (e.g. 7 Days, or 'Not specified')\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        user_content = (
            f"CLINICAL NOTE PRESCRIPTION SECTION:\n{note_prescription}\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"Output only the raw JSON matching the schema."
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.05
        }

        # Try Perplexity standard chat completions and custom sonar path
        endpoints = [f"{base_url}/chat/completions", f"{base_url}/v1/chat/completions", f"{base_url}/v1/sonar"]
        last_error = None

        for endpoint in endpoints:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(endpoint, json=payload, headers=headers, timeout=15.0)
                    if response.status_code == 200:
                        data = response.json()
                        raw_output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        return self._clean_json_response(raw_output)
                    else:
                        last_error = f"HTTP {response.status_code}: {response.text}"
            except Exception as e:
                last_error = str(e)
                continue

        raise RuntimeError(f"Sonar API failed on all endpoints. Last error: {last_error}")

    def call_huggingface_api(
        self,
        token: str,
        model_key: str,
        transcript: str,
        note_prescription: str
    ) -> Dict[str, Any]:
        """
        Extracts structured prescription JSON via Hugging Face completion router.
        """
        model_id = SUPPORTED_LLM_MODELS.get(model_key, SUPPORTED_LLM_MODELS["qwen"])["id"]
        
        system_prompt = (
            "You are a clinical extraction assistant. Extract all prescribed medications from the transcript and clinical notes.\n"
            "Return JSON only in this schema:\n"
            "{\n"
            "  \"medications\": [\n"
            "    {\n"
            "      \"name\": \"string\",\n"
            "      \"dosage\": \"string\",\n"
            "      \"frequency\": \"string\",\n"
            "      \"duration\": \"string\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        user_content = (
            f"CLINICAL NOTE PRESCRIPTION SECTION:\n{note_prescription}\n\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"Output only the raw JSON matching the schema."
        )

        client = InferenceClient(
            token=token,
            base_url="https://router.huggingface.co/v1",
            timeout=15.0
        )
        response = client.chat_completion(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=600,
            temperature=0.05
        )
        raw_output = response.choices[0].message.content
        return self._clean_json_response(raw_output)

    async def extract_prescription_json(
        self,
        transcript: str,
        note_data: Dict[str, Any],
        model_key: str = "qwen",
        hf_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main interface to extract structured prescription JSON.
        Tries Perplexity Sonar first, then Hugging Face router, and falls back to regex matching.
        """
        note_prescription = note_data.get("prescription") or ""
        
        # Strip default/boilerplate if none specified
        if not note_prescription or note_prescription.lower().strip() in ("none", "not specified", "n/a"):
            return {"medications": []}

        # Check for Perplexity API key
        sonar_api_key = os.getenv("SONAR_API_KEY") or os.getenv("PERPLEXITY_API_KEY")
        if sonar_api_key and sonar_api_key.strip():
            logger.info("[PrescriptionService] Running Perplexity Sonar API extraction.")
            try:
                return await self.call_sonar_api(sonar_api_key, transcript, note_prescription)
            except Exception as e:
                logger.error(f"[PrescriptionService] Sonar API failed: {e}. Falling back to Hugging Face API.")

        # Check for Hugging Face token
        token = hf_token or settings.HF_TOKEN
        is_demo = not token or token.strip().lower() in ("", "demo", "test", "mock", "none", "hf_demo")
        
        if not is_demo:
            logger.info("[PrescriptionService] Running Hugging Face API extraction.")
            try:
                return self.call_huggingface_api(token, model_key, transcript, note_prescription)
            except Exception as e:
                logger.error(f"[PrescriptionService] Hugging Face extraction failed: {e}. Falling back to local parser.")
        else:
            logger.info("[PrescriptionService] Demo mode active. Skipping live API extraction.")

        # Fallback to local regex/string parser
        logger.info("[PrescriptionService] Running local regex extraction fallback.")
        return self.parse_note_prescription_local(note_prescription)

prescription_service = PrescriptionService()
