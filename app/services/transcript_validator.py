import logging
import re
import time
from typing import Optional
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from app.config import settings, SUPPORTED_LLM_MODELS

logger = logging.getLogger(__name__)

# Medical term spelling correction map (common ASR misheard words/homophones)
COMMON_ASR_CORRECTIONS = {
    r"\bsir\s*tech\b": "Zyrtec",
    r"\bparacetol\b": "paracetamol",
    r"\bacetaminophin\b": "acetaminophen",
    r"\bco\s*diovan\b": "Co-Diovan",
    r"\bcardivan\b": "Cardivan",
    r"\bmetaformin\b": "metformin",
    r"\blisinopril\b": "lisinopril",
    r"\batorvastin\b": "atorvastatin",
    r"\bsumatryptan\b": "sumatriptan",
    r"\bpenicilin\b": "penicillin",
    r"\bamlodapine\b": "amlodipine",
    r"\bomeprazole\b": "omeprazole",
    r"\bsimvastin\b": "simvastatin",
    r"\bazithromicin\b": "azithromycin",
    r"\bclopidogril\b": "clopidogrel",
    r"\bmontelukast\b": "montelukast",
    r"\bfluticasone\b": "fluticasone",
    r"\bpantoprazole\b": "pantoprazole",
    r"\bfurosemide\b": "furosemide",
    r"\bgabapentin\b": "gabapentin",
    r"\bwheezing\b": "wheezing",
    r"\bcholecystectomy\b": "cholecystectomy",
    r"\bcholelithiasis\b": "cholelithiasis"
}

class TranscriptValidator:
    def __init__(self):
        pass

    def rule_based_cleanup(self, raw_transcript: str) -> str:
        """
        Performs local regex and rule-based cleanup on the transcript.
        Cleans Whisper silence repetition loops and maps common medical homophones.
        """
        if not raw_transcript or not raw_transcript.strip():
            return ""

        text = raw_transcript.strip()

        # 1. Remove common Whisper silence loop artifacts (e.g. "Thank you. Thank you. Thank you.")
        # Match a phrase repeating 3 or more times consecutively
        phrase_patterns = [
            r"(\bThank you\b\.?\s*){3,}",
            r"(\bGood afternoon\b\.?\s*){3,}",
            r"(\bSpeak clearly\b\.?\s*){3,}",
            r"(\bPlease wait\b\.?\s*){3,}",
            r"(\bHello doctor\b\.?\s*){3,}"
        ]
        for pattern in phrase_patterns:
            text = re.sub(pattern, lambda m: m.group(1).strip() + " ", text, flags=re.IGNORECASE)

        # 2. Correct common medical spelling/ASR mistakes using regex map
        for pattern, replacement in COMMON_ASR_CORRECTIONS.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 3. Clean up duplicate consecutive words (e.g. "the the", "I I")
        text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

        return text.strip()

    def validate_and_clean(
        self,
        raw_transcript: str,
        model_key: str = "qwen",
        hf_token: Optional[str] = None
    ) -> str:
        """
        Performs a two-stage cleanup and validation:
        1. Local Rule-Based Cleanup
        2. LLM-Based ASR Validation Layer
        """
        if not raw_transcript or not raw_transcript.strip():
            return ""

        # Stage 1: Rule-Based Cleanup
        cleaned_text = self.rule_based_cleanup(raw_transcript)
        logger.info(f"[TranscriptValidator] Completed rule-based cleanup.")

        # Stage 2: AI Validation Layer
        token = hf_token or settings.HF_TOKEN
        is_demo = not token or token.strip().lower() in ("", "demo", "test", "mock", "none", "hf_demo")

        if is_demo:
            logger.info("[TranscriptValidator] Demo mode active. Skipping validation LLM request.")
            return cleaned_text

        model_id = SUPPORTED_LLM_MODELS.get(model_key, SUPPORTED_LLM_MODELS["qwen"])["id"]
        logger.info(f"[TranscriptValidator] Requesting AI validation layer using model: {model_id}")

        validation_prompt = (
            "Review the transcript below. Correct obvious speech-to-text mistakes, "
            "remove duplicated phrases, fix medication names if confidence is high, "
            "and preserve all clinically relevant information. Do not invent new facts."
        )

        messages = [
            {"role": "system", "content": "You are a medical transcription editor. Fix STT errors but preserve facts exactly. Do not write explanations. Output only the corrected transcript."},
            {"role": "user", "content": f"Task: {validation_prompt}\n\nTranscript:\n{cleaned_text}"}
        ]

        # Retry logic with exponential backoff and timeout
        max_retries = 2
        base_delay = 1.0
        backoff_factor = 2

        for attempt in range(max_retries + 1):
            try:
                client = InferenceClient(
                    token=token,
                    base_url="https://router.huggingface.co/v1",
                    timeout=10.0
                )
                response = client.chat_completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.1
                )
                corrected_text = response.choices[0].message.content
                if corrected_text and corrected_text.strip():
                    logger.info("[TranscriptValidator] AI Validation Layer execution successful.")
                    return corrected_text.strip()
                break
            except Exception as e:
                logger.warning(
                    f"[TranscriptValidator] Attempt {attempt+1} failed during AI validation: {e}."
                )
                if attempt < max_retries:
                    delay = base_delay * (backoff_factor ** attempt)
                    time.sleep(delay)
                else:
                    logger.error(
                        "[TranscriptValidator] Max retries exceeded. Falling back to rule-based cleaned transcript."
                    )

        # Fallback to stage 1 output in case of exceptions/timeouts
        return cleaned_text

transcript_validator_service = TranscriptValidator()
