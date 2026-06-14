import logging
import re
import time
from typing import Optional, Dict
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from app.config import settings, SUPPORTED_LLM_MODELS, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

COMMON_DRUGS_LIST = [
    "amoxicillin", "paracetamol", "ibuprofen", "lisinopril", "atorvastatin", 
    "sumatriptan", "acetaminophen", "aspirin", "penicillin", "metformin", 
    "albuterol", "levothyroxine", "gabapentin", "amlodipine", "omeprazole", 
    "simvastatin", "hydroxychloroquine", "azithromycin", "clopidogrel", 
    "montelukast", "fluticasone", "pantoprazole", "furosemide"
]

class NoteGeneratorService:
    def __init__(self):
        pass

    def generate_note(
        self,
        transcript: str,
        model_key: str = "qwen",
        system_prompt: Optional[str] = None,
        hf_token: Optional[str] = None,
        mode: str = "structured",
        custom_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Generates a structured clinical note or custom formatted text from a transcript using Hugging Face LLMs.
        """
        if mode == "transcript":
            logger.info("Raw Transcript mode selected. Short-circuiting generation service.")
            return {
                "mode": "transcript",
                "transcript": transcript
            }

        token = hf_token or settings.HF_TOKEN
        is_demo = not token or token.strip().lower() in ("", "demo", "test", "mock", "none", "hf_demo")
        
        if is_demo:
            logger.info("Demo mode active for note generation. Generating mock clinical note.")
            if mode == "custom":
                if not custom_prompt or not custom_prompt.strip():
                    raise ValueError("Custom prompt cannot be empty.")
                custom_prompt_lower = (custom_prompt or "").lower()
                if "referral" in custom_prompt_lower:
                    raw_content = (
                        "REFERRAL LETTER\n\n"
                        "Date: June 11, 2026\n"
                        "To: Specialist Clinic\n"
                        "From: Primary Care Scribe\n\n"
                        "Subject: Referral of Patient for further evaluation.\n\n"
                        "Dear Colleague,\n\n"
                        "I am referring this patient for specialist consultation. Based on our latest conversation:\n"
                        f"- Patient presents for evaluation.\n"
                        f"- Main topic discussed: {transcript.strip()[:150]}...\n\n"
                        "Thank you for your assistance with this patient.\n\n"
                        "Sincerely,\n"
                        "Dr. Primary Care Scribe"
                    )
                else:
                    raw_content = (
                        f"Custom Prompt Output (Demo Mode):\n\n"
                        f"We processed the conversation under your custom instruction: '{custom_prompt}'\n\n"
                        f"Key Transcript Facts:\n"
                        f"- Patient presents for evaluation.\n"
                        f"- Main topic discussed: {transcript.strip()[:200]}...\n\n"
                        "Decision: Factual extraction completed. All missing medical details marked 'Not specified'."
                    )
                sanitized_content = self._sanitize_general_section(raw_content, transcript)
                return {
                    "raw_note": raw_content,
                    "model_used": "Demo Mode (Mock Custom Model)",
                    "custom_output": sanitized_content,
                    "mode": "custom"
                }

            prompt_lower = (system_prompt or "").lower()
            transcript_lower = transcript.lower()
            
            if "surgeon" in prompt_lower or "surgery" in prompt_lower or "surgical" in prompt_lower or "cholecystectomy" in transcript_lower:
                raw_content = (
                    "Chief Complaint:\nGallbladder pain and fatty meal intolerance.\n\n"
                    "Procedure:\n42-year-old female presents for pre-operative consultation for symptomatic cholelithiasis and planned laparoscopic cholecystectomy. Reports recurrent right upper quadrant pain radiating to the right shoulder, particularly after fatty meals.\n\n"
                    "Findings:\nSymptomatic cholelithiasis. Indicated for elective laparoscopic cholecystectomy.\n\n"
                    "Recommendations:\nLaparoscopic cholecystectomy scheduled. Risks, benefits, and alternatives discussed. Post-operative wound care and lifting restrictions reviewed.\n\n"
                    "Prescription:\nIbuprofen 600mg PO every 6 hours PRN pain. Acetaminophen 325mg PO PRN.\n\n"
                    "Recommended Tests:\nPre-op CBC, BMP, hepatic function panel, and coagulation profile.\n\n"
                    "Follow-up:\nPost-op clinic visit in 10-14 days for wound check and suture removal."
                )
                model_id = "Demo Mode (Mock Surgeon Model)"
            elif "obgyn" in prompt_lower or "gynecology" in prompt_lower or "obstetrics" in prompt_lower or "pregnancy" in transcript_lower or "pregnancy" in prompt_lower:
                raw_content = (
                    "Chief Complaint:\nInitial pregnancy consultation and morning sickness.\n\n"
                    "Obstetric History:\n29-year-old G1P0 female at 12 weeks gestation presents for initial prenatal visit. Reports mild morning nausea but denies cramping or vaginal bleeding. LMP is March 18, 2026.\n\n"
                    "Assessment:\nIntrauterine pregnancy at 12 weeks gestation. Normal prenatal status.\n\n"
                    "Plan:\nDiscussed prenatal care timeline, nutrition, weight gain limits, and warning signs (cramping, bleeding).\n\n"
                    "Prescription:\nPrenatal vitamins once daily.\n\n"
                    "Recommended Tests:\nObstetric ultrasound, prenatal laboratory panel (CBC, Blood type/Rh, Rubella, HIV, RPR, Hepatitis B).\n\n"
                    "Follow-up:\nNext prenatal visit in 4 weeks."
                )
                model_id = "Demo Mode (Mock OB/GYN Model)"
            elif "cardiologist" in prompt_lower or "cardiology" in prompt_lower or "heart" in prompt_lower or "chest pain" in transcript_lower:
                raw_content = (
                    "Chief Complaint:\nShortness of breath and pressure in chest.\n\n"
                    "HPI:\nPatient is a 58-year-old male presenting with intermittent chest pain and shortness of breath over the past week. Pain is substernal, radiating to the left arm, aggravated by exertion. Blood pressure is 145/90 mmHg.\n\n"
                    "Assessment:\nAngina pectoris vs. ischemic heart disease. Hypertension.\n\n"
                    "Plan:\nDiscussed low-sodium diet and cardiovascular exercise guidelines. Go to nearest ER for chest pain radiating to neck/jaw or worsening dyspnea.\n\n"
                    "Prescription:\nLisinopril 10mg daily. Atorvastatin 20mg nightly.\n\n"
                    "Recommended Tests:\nElectrocardiogram (EKG), Echocardiogram, referral for Exercise Stress Test.\n\n"
                    "Follow-up:\nReturn to cardiology clinic in 2 weeks with home blood pressure logs."
                )
                model_id = "Demo Mode (Mock Cardiology Model)"
            else:
                raw_content = (
                    "Chief Complaint:\nSevere headache for three days.\n\n"
                    "HPI:\nPatient is a 34-year-old female presenting with throbbing left-sided headache for 3 days, accompanied by photophobia and nausea. Reports history of maternal migraine.\n\n"
                    "Assessment:\nMigraine headache without aura.\n\n"
                    "Plan:\nRest in a dark, quiet room. Maintain hydration and trigger diary.\n\n"
                    "Prescription:\nSumatriptan 50mg PO at onset of migraine, may repeat once in 2 hours PRN (Max 100mg/24h).\n\n"
                    "Recommended Tests:\nRoutine lipid panel.\n\n"
                    "Follow-up:\nFollow up in 2 weeks. Return to clinic or ER for sudden vision loss or meningeal signs."
                )
                model_id = "Demo Mode (Mock General Physician Model)"
            
            parsed_sections = self._parse_note_sections(raw_content)
            sanitized_sections = self._validate_and_sanitize_note(parsed_sections, transcript)
            return {
                "raw_note": raw_content,
                "model_used": model_id,
                **sanitized_sections,
                "mode": "structured"
            }

        model_id = SUPPORTED_LLM_MODELS.get(model_key, SUPPORTED_LLM_MODELS["qwen"])["id"]
        
        if mode == "custom":
            if not custom_prompt or not custom_prompt.strip():
                raise ValueError("Custom prompt cannot be empty.")
            sys_prompt = (
                "You are a professional medical AI assistant.\n"
                "Follow the doctor's instruction exactly.\n"
                "Do not invent facts.\n"
                "If information is missing, leave it out."
            )
            user_content = (
                f"Doctor Instruction:\n{custom_prompt}\n\n"
                f"Conversation:\n{transcript}"
            )
        else:
            sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            user_content = f"Here is the doctor-patient conversation transcript:\n\n{transcript}"
        
        logger.info(f"Generating clinical note using Hugging Face model: {model_id} (mode: {mode})")

        try:
            client = InferenceClient(
                token=token,
                base_url="https://router.huggingface.co/v1"
            )
            
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content}
            ]
            
            logger.info(f"Connecting to: https://router.huggingface.co/v1/chat/completions (model: {model_id})")
            
            # Retry loop with exponential backoff for 429 rate limit errors
            max_retries = 3
            backoff_factor = 2
            base_delay = 1.0  # seconds
            
            response = None
            for attempt in range(max_retries + 1):
                try:
                    response = client.chat_completion(
                        model=model_id,
                        messages=messages,
                        max_tokens=1500,
                        temperature=0.1,  # Low temperature for deterministic clinical notes
                    )
                    break # Success!
                except HfHubHTTPError as e:
                    is_rate_limit = False
                    if e.response is not None and e.response.status_code == 429:
                        is_rate_limit = True
                    elif "429" in str(e) or "too many requests" in str(e).lower() or "rate limit" in str(e).lower():
                        is_rate_limit = True
                        
                    if is_rate_limit and attempt < max_retries:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(f"Rate limit (429) hit during note generation. Retrying in {delay:.2f}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(delay)
                    else:
                        raise e
                except Exception as e:
                    raise e
            
            raw_content = response.choices[0].message.content
            
            if mode == "custom":
                sanitized_content = self._sanitize_general_section(raw_content, transcript)
                return {
                    "raw_note": raw_content,
                    "model_used": model_id,
                    "custom_output": sanitized_content,
                    "mode": "custom"
                }
            else:
                parsed_sections = self._parse_note_sections(raw_content)
                sanitized_sections = self._validate_and_sanitize_note(parsed_sections, transcript)
                
                return {
                    "raw_note": raw_content,
                    "model_used": model_id,
                    **sanitized_sections,
                    "mode": "structured"
                }

        except HfHubHTTPError as e:
            logger.error(f"Hugging Face Inference API error: {e}")
            if e.response is not None and e.response.status_code in (401, 403):
                if "Inference Providers" in str(e) or "permissions" in str(e):
                    raise RuntimeError(
                        "Your Hugging Face token is missing the 'Inference Providers' permission. "
                        "Please go to your Hugging Face settings -> Access Tokens (https://huggingface.co/settings/tokens), "
                        "edit your token, check the 'Inference Providers' permission box, and save."
                    )
            if "Model" in str(e) and "is currently loading" in str(e):
                raise RuntimeError(
                    "The selected LLM model is currently loading on Hugging Face. "
                    "Please wait a minute and try again."
                )
            raise RuntimeError(f"Hugging Face Hub error: {e.server_message or str(e)}")
        except Exception as e:
            logger.error(f"Unexpected generation error: {e}")
            if "getaddrinfo failed" in str(e) or "NameResolutionError" in str(e):
                raise RuntimeError(
                    "Connection failed: Unable to resolve Hugging Face server. "
                    "Please verify your internet connection or DNS settings."
                )
            if "'dict' object has no attribute 'strip'" in str(e):
                raise RuntimeError(
                    f"The selected LLM model '{model_id}' is not supported as a serverless chat model by the Hugging Face Router, or the request was invalid. Please try selecting a different model, such as Qwen 2.5."
                )
            raise RuntimeError(f"Note generation failed: {str(e)}")

    def _parse_note_sections(self, raw_note: str) -> Dict[str, str]:
        """
        Parses the raw clinical note into separate sections using regex patterns.
        Supports standard and specialty templates.
        """
        sections = {
            "chief_complaint": "",
            "hpi": "",
            "assessment": "",
            "plan": "",
            "prescription": "",
            "recommended_tests": "",
            "follow_up": ""
        }
        
        # Define regex patterns for each section header, including fallbacks
        patterns = {
            "chief_complaint": r"(?:Chief Complaint|CC)\s*:\s*(.*?)(?=\n(?:HPI|History of Present Illness|Procedure|Obstetric History|Obstetric/gynecological history|Assessment|Findings|Plan|Recommendations|Prescription|Recommended Tests|Recommended Test|Follow-up|Follow Up)|$)",
            "hpi": r"(?:HPI|History of Present Illness|Procedure|Obstetric History|Obstetric/gynecological history)\s*:\s*(.*?)(?=\n(?:Chief Complaint|CC|Assessment|Findings|Plan|Recommendations|Prescription|Recommended Tests|Recommended Test|Follow-up|Follow Up)|$)",
            "assessment": r"(?:Assessment|Findings)\s*:\s*(.*?)(?=\n(?:Chief Complaint|CC|HPI|History of Present Illness|Procedure|Obstetric History|Obstetric/gynecological history|Plan|Recommendations|Prescription|Recommended Tests|Recommended Test|Follow-up|Follow Up)|$)",
            "plan": r"(?:Plan|Recommendations)\s*:\s*(.*?)(?=\n(?:Chief Complaint|CC|HPI|History of Present Illness|Procedure|Obstetric History|Obstetric/gynecological history|Assessment|Findings|Prescription|Recommended Tests|Recommended Test|Follow-up|Follow Up)|$)",
            "prescription": r"(?:Prescription)\s*:\s*(.*?)(?=\n(?:Chief Complaint|CC|HPI|History of Present Illness|Procedure|Obstetric History|Obstetric/gynecological history|Assessment|Findings|Plan|Recommendations|Recommended Tests|Recommended Test|Follow-up|Follow Up)|$)",
            "recommended_tests": r"(?:Recommended Tests|Recommended Test)\s*:\s*(.*?)(?=\n(?:Chief Complaint|CC|HPI|History of Present Illness|Procedure|Obstetric History|Obstetric/gynecological history|Assessment|Findings|Plan|Recommendations|Prescription|Follow-up|Follow Up)|$)",
            "follow_up": r"(?:Follow-up|Follow Up)\s*:\s*(.*?)(?=\n(?:Chief Complaint|CC|HPI|History of Present Illness|Procedure|Obstetric History|Obstetric/gynecological history|Assessment|Findings|Plan|Recommendations|Prescription|Recommended Tests|Recommended Test)|$)"
        }
        
        # Clean up markdown headers if the model adds them (e.g. ### HPI: or **HPI:**)
        cleaned_note = re.sub(r'^(?:###|\*\*|#)\s*', '', raw_note, flags=re.MULTILINE)
        cleaned_note = re.sub(r'\s*\*\*\s*$', '', cleaned_note, flags=re.MULTILINE)
        
        for key, pattern in patterns.items():
            match = re.search(pattern, cleaned_note, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
            else:
                # Fallback: search for header without colon
                fallback_pattern = pattern.replace(r"\s*:\s*", r"\s*:?\s*")
                fallback_match = re.search(fallback_pattern, cleaned_note, re.DOTALL | re.IGNORECASE)
                if fallback_match:
                    sections[key] = fallback_match.group(1).strip()
                else:
                    sections[key] = ""
                    
        return sections

    def _sanitize_general_section(self, text: str, transcript: str) -> str:
        """
        Sanitizes a general note section line-by-line.
        Ensures all mentioned numbers, high-risk diagnoses, and clinical terms are grounded in the transcript.
        Strips speculation words if they are not in the transcript.
        If a line contains ungrounded details, it is removed.
        """
        if not text or text.lower().strip() in ("", "none", "not specified", "n/a"):
            return "Not specified"

        transcript_lower = transcript.lower()
        
        # Split section into lines
        lines = text.split('\n')
        sanitized_lines = []
        
        # Standard synonym map for high-risk clinical terms
        synonym_map = {
            "migraine": ["migraine", "headache", "headaches"],
            "migraines": ["migraine", "headache", "headaches"],
            "headaches": ["migraine", "headache", "headaches"],
            "hypertension": ["hypertension", "blood pressure", "bp", "high pressure"],
            "gerd": ["gerd", "acid reflux", "heartburn", "stomach", "gastric"],
            "gastritis": ["gastritis", "stomach", "gastric", "indigestion", "heartburn"],
            "cholelithiasis": ["cholelithiasis", "gallbladder", "stone", "stones"],
            "cholecystitis": ["cholecystitis", "gallbladder", "stone", "stones"],
            "pregnancy": ["pregnancy", "pregnant", "gestation", "prenatal", "baby", "weeks"],
            "gestation": ["pregnancy", "pregnant", "gestation", "prenatal", "weeks"],
            "angina": ["angina", "chest pain", "heart"],
            "diabetes": ["diabetes", "diabetic", "sugar"],
            "asthma": ["asthma", "wheezing", "inhaler"],
            "pneumonia": ["pneumonia", "cough", "lung", "lungs"],
            "stroke": ["stroke", "tia", "neurological"],
            "arthritis": ["arthritis", "joint pain", "joints"],
            "infection": ["infection", "fever", "throat", "cough", "respiratory", "cold", "flu", "uri"],
            "uri": ["uri", "respiratory", "cold", "flu", "infection", "throat", "cough"],
            "tonsillitis": ["tonsillitis", "throat", "sore", "infection"],
            "bronchitis": ["bronchitis", "cough", "chest", "infection"],
            "blood": ["blood", "bp", "hypertension", "pressure"],
            "pressure": ["pressure", "bp", "hypertension", "blood"],
            "bp": ["bp", "blood pressure", "hypertension"],
            "temp": ["temp", "temperature", "fever"],
            "temperature": ["temperature", "temp", "fever"]
        }

        number_words = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty"}
        
        strict_medical_keywords = {
            # Diagnoses
            "cholelithiasis", "migraine", "migraines", "hypertension", "angina", "diabetes", 
            "asthma", "gerd", "cholecystitis", "gestation", "pregnancy", "ischemic", 
            "cardiovascular", "pneumonia", "cancer", "stroke", "arthritis", "headaches",
            "gastritis", "bronchitis", "tonsillitis", "infection", "uri",
            # Lab tests
            "cbc", "bmp", "ekg", "ecg", "ultrasound", "mri", "ct", "lipid", "x-ray", "labs", 
            "laboratory", "urine", "pathology", "coagulation", "hepatic", "prenatal",
            # Allergies / history
            "allergy", "allergies", "history",
            # Vital signs keywords
            "bp", "blood", "pressure", "temp", "temperature", "pulse", "heartbeat", "respiration", "oxygen"
        }

        stop_words = {
            "we", "also", "has", "of", "the", "a", "an", "is", "was", "for", "with", "to", "in", "on", "at", "by", 
            "from", "and", "but", "or", "if", "then", "else", "this", "that", "these", "those", "it", "they", 
            "he", "she", "you", "i", "my", "your", "his", "her", "their", "our", "him", "them", "me", "us", 
            "who", "whom", "which", "what", "how", "why", "where", "when", "been", "have", "had", "do", "does", 
            "did", "diagnosed", "diagnoses", "prescribed", "prescribe", "patient", "doctor", "presents", 
            "presenting", "complains", "complaining", "here", "summary", "out", "about", "reports", "reporting",
            "denies", "denying", "includes", "including", "history"
        }

        def check_number_grounding(num: str) -> bool:
            if num in transcript_lower:
                return True
            num_to_word = {
                "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
                "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
                "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
                "14": "fourteen", "15": "fifteen", "20": "twenty", "30": "thirty"
            }
            word_to_num = {v: k for k, v in num_to_word.items()}
            if num in num_to_word and num_to_word[num] in transcript_lower:
                return True
            if num in word_to_num and word_to_num[num] in transcript_lower:
                return True
            return False

        def sanitize_line_content(line_content: str) -> str:
            # Split line into sentences
            sentences = re.split(r'(?<=[.!?])\s+', line_content)
            sanitized_sentences = []
            
            for sentence in sentences:
                speculation_words = ["likely", "possible", "probable", "suspected", "rule out", "maybe"]
                for sw in speculation_words:
                    if sw in sentence.lower() and sw not in transcript_lower:
                        sentence = re.sub(r'\b' + re.escape(sw) + r'\b', '', sentence, flags=re.IGNORECASE)
                        
                tokens = sentence.split()
                sanitized_tokens = []
                
                has_keywords_or_digits = False
                kept_keywords_or_digits = False
                
                for token in tokens:
                    clean_token = token.strip(".,;:!?()[]{}*-\"'+")
                    clean_token_lower = clean_token.lower()
                    
                    if not clean_token:
                        sanitized_tokens.append(token)
                        continue
                        
                    is_digit = bool(re.findall(r'\d+', clean_token))
                    is_num_word = clean_token_lower in number_words
                    freq_dur_keywords = {
                        "daily", "nightly", "weekly", "monthly", "hours", "days", "weeks", "months", 
                        "times", "once", "twice", "prn", "hourly", "every"
                    }
                    is_freq_dur = clean_token_lower in freq_dur_keywords
                    is_med_keyword = clean_token_lower in strict_medical_keywords or clean_token_lower in COMMON_DRUGS_LIST
                    
                    is_special = is_digit or is_num_word or is_freq_dur or is_med_keyword
                    if is_special:
                        has_keywords_or_digits = True
                        
                    keep_token = True
                    if is_digit:
                        digits = re.findall(r'\d+', clean_token)
                        for d in digits:
                            if not check_number_grounding(d):
                                keep_token = False
                                break
                    elif is_num_word:
                        if not check_number_grounding(clean_token_lower):
                            keep_token = False
                    elif is_freq_dur:
                        stemmed = clean_token_lower.rstrip('s')
                        if clean_token_lower not in transcript_lower and stemmed not in transcript_lower:
                            keep_token = False
                    elif is_med_keyword:
                        if clean_token_lower == "pregnancy" and ("pregnant" in transcript_lower or "gestation" in transcript_lower):
                            pass
                        elif clean_token_lower == "gestation" and ("pregnancy" in transcript_lower or "pregnant" in transcript_lower):
                            pass
                        elif clean_token_lower in synonym_map:
                            syns = synonym_map[clean_token_lower]
                            if not any(s in transcript_lower for s in syns):
                                keep_token = False
                        else:
                            if clean_token_lower not in transcript_lower:
                                keep_token = False
                                
                    if keep_token:
                        if is_special:
                            kept_keywords_or_digits = True
                        sanitized_tokens.append(token)
                        
                # If the sentence had keywords/digits but none were kept, discard the entire sentence!
                if has_keywords_or_digits and not kept_keywords_or_digits:
                    continue
                    
                # Check if the sanitized sentence contains only stop words
                has_content = False
                for token in sanitized_tokens:
                    clean_token = token.strip(".,;:!?()[]{}*-\"'+")
                    if clean_token and clean_token.lower() not in stop_words:
                        has_content = True
                        break
                        
                if not has_content:
                    continue
                    
                sanitized_sentence = " ".join(sanitized_tokens)
                if re.sub(r'^[.,;:!?\s\-\*]+$', '', sanitized_sentence).strip():
                    sanitized_sentences.append(sanitized_sentence)
                    
            return " ".join(sanitized_sentences)

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.lower() in ("not specified", "n/a", "none"):
                continue
                
            # Keep list markers if present (like - or *)
            marker_match = re.match(r'^(\s*(?:[\*\-\+]\s+|\d+[\.\)]\s+))(.*)$', line_stripped)
            if marker_match:
                marker, content = marker_match.groups()
            else:
                marker, content = "", line_stripped
                
            san_content = sanitize_line_content(content)
            clean_san_content = re.sub(r'^[.,;:!?\s\-\*]+$', '', san_content).strip()
            if clean_san_content:
                sanitized_lines.append(f"{marker}{san_content}")
                
        # Filter out empty or duplicate boilerplate-only results
        if not sanitized_lines:
            return "Not specified"
            
        # Reconstruct the section text
        result_text = "\n".join(sanitized_lines)
        
        # Clean up double bullets or empty strings
        if result_text.strip() == "":
            return "Not specified"
            
        return result_text
            
        return sanitized

    def _validate_and_sanitize_note(self, sections: Dict[str, str], transcript: str) -> Dict[str, str]:
        """
        Validates and sanitizes the parsed note sections against the original transcript.
        Ensures no invented/hallucinated dosages, frequencies, or durations are kept.
        """
        transcript_lower = transcript.lower()
        
        # Words to exclude from being treated as drug names
        exclude_words = {
            "mg", "ml", "g", "mcg", "units", "tablet", "tablets", "capsule", "capsules", 
            "dose", "daily", "every", "hours", "days", "weeks", "months", "take", 
            "prescribed", "prescription", "for", "with", "times", "once", "twice",
            "not", "specified", "none", "and", "or", "the", "a", "an", "to", "i", 
            "will", "prescribe", "some", "any", "but", "of", "patient", "doctor", 
            "morning", "nightly", "day", "week", "month", "need", "needed", "prn", 
            "use", "using", "taking", "caps", "tabs", "po", "by", "mouth", "medication", 
            "medications", "prescriptions", "on", "at", "start", "starting", "let", "us"
        }
        
        # Extract all lines from prescription
        prescription_text = sections.get("prescription", "")
        medications = []
        
        if prescription_text and prescription_text.lower().strip() not in ("", "none", "not specified", "n/a"):
            lines = prescription_text.split('\n')
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # Clean bullet markers
                cleaned_line = re.sub(r'^[\*\-\d\.\s\+]+', '', line_stripped).strip()
                if not cleaned_line:
                    continue
                    
                # Extract all word tokens of length >= 3
                words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned_line)
                drug_name = ""
                
                # 1. Check if any word matches a known common drug that is also in the transcript
                for w in words:
                    w_lower = w.lower()
                    if w_lower in COMMON_DRUGS_LIST and w_lower in transcript_lower:
                        drug_name = w
                        break
                        
                # 2. If no common drug matched, check if any word of length >= 3 is in the transcript and not in exclude_words
                if not drug_name:
                    for w in words:
                        w_lower = w.lower()
                        if w_lower in transcript_lower and w_lower not in exclude_words:
                            if w[0].isupper() or not drug_name:
                                drug_name = w
                                
                # 3. Fallback: if no drug name is matched from the line, but the line contains a word from the transcript
                if not drug_name:
                    for w in words:
                        w_lower = w.lower()
                        if w_lower in transcript_lower and w_lower not in exclude_words:
                            drug_name = w
                            break
                            
                if drug_name:
                    # We found a valid drug name! Now extract dosage, frequency, duration
                    dosage = "Not specified"
                    frequency = "Not specified"
                    duration = "Not specified"
                    
                    # 1. Dosage Check
                    dosage_match = re.search(r'\b\d+\s*(?:mg|ml|g|mcg|units|capsules?|tablets?)\b', cleaned_line, re.IGNORECASE)
                    if dosage_match:
                        val = dosage_match.group(0)
                        val_normalized = re.sub(r'\s+', '', val).lower()
                        # Check if present in transcript
                        if val.lower() in transcript_lower or val_normalized in transcript_lower.replace(" ", ""):
                            dosage = val
                            
                    # 2. Frequency Check
                    freq_keywords = [
                        r'\b(?:once|twice|three|four|\d+)\s*(?:times?\s*)?(?:daily|a\s*day|weekly)\b',
                        r'\bevery\s*(?:\d+|six|eight|twelve|four|two)\s*hours\b',
                        r'\bdaily\b', r'\bnightly\b', r'\bmorning\b', r'\bprn\b', r'\bas\s*needed\b'
                      ]
                    for pattern in freq_keywords:
                        freq_match = re.search(pattern, cleaned_line, re.IGNORECASE)
                        if freq_match:
                            val = freq_match.group(0)
                            val_words = re.findall(r'\b\w+\b', val.lower())
                            if all(vw in transcript_lower for vw in val_words if vw not in ("a", "for", "as")):
                                frequency = val
                                break
                                
                    # 3. Duration Check
                    duration_match = re.search(r'\b(?:for\s+)?\d+\s*(?:days|weeks|months)\b', cleaned_line, re.IGNORECASE)
                    if duration_match:
                        val = duration_match.group(0)
                        val_words = re.findall(r'\b\w+\b', val.lower())
                        if all(vw in transcript_lower for vw in val_words if vw not in ("for", "a")):
                            duration = val
                            
                    # Avoid duplicates of the same drug name
                    if not any(m["name"].lower() == drug_name.lower() for m in medications):
                        medications.append({
                            "name": drug_name,
                            "dosage": dosage,
                            "frequency": frequency,
                            "duration": duration
                        })
        
        # If no medications were parsed from the generated output, let's scan the transcript for known common drugs
        if not medications:
            for drug in COMMON_DRUGS_LIST:
                if drug in transcript_lower:
                    # Capitalize the drug name for output
                    medications.append({
                        "name": drug.capitalize(),
                        "dosage": "Not specified",
                        "frequency": "Not specified",
                        "duration": "Not specified"
                    })
                    
        # Format the prescription section
        if not medications:
            sections["prescription"] = "Not specified"
        else:
            all_missing = all(
                m["dosage"] == "Not specified" and 
                m["frequency"] == "Not specified" and 
                m["duration"] == "Not specified"
                for m in medications
            )
            if len(medications) > 1 and all_missing:
                lines = []
                for m in medications:
                    lines.append(f"* {m['name']}")
                lines.append("")
                lines.append("Dosage:")
                lines.append("* Not specified")
                lines.append("")
                lines.append("Frequency:")
                lines.append("* Not specified")
                lines.append("")
                lines.append("Duration:")
                lines.append("* Not specified")
                sections["prescription"] = "\n".join(lines)
            else:
                blocks = []
                for m in medications:
                    block = (
                        f"Medication:\n"
                        f"* {m['name']}\n"
                        f"Dosage:\n"
                        f"{m['dosage']}\n"
                        f"Frequency:\n"
                        f"{m['frequency']}\n"
                        f"Duration:\n"
                        f"{m['duration']}"
                    )
                    blocks.append(block)
                sections["prescription"] = "\n\n".join(blocks)
                
        # Sanitize all other general sections
        for section_name in ["chief_complaint", "hpi", "assessment", "plan", "recommended_tests", "follow_up"]:
            if section_name in sections:
                sections[section_name] = self._sanitize_general_section(sections[section_name], transcript)
                
        return sections

note_generator_service = NoteGeneratorService()
