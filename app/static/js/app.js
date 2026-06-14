// State Variables
let recognition = null;
let isRecording = false;
let recognitionState = "stopped"; // 'stopped', 'starting', 'recording', 'stopping'
let systemPromptDefault = "";

// Safari / MediaRecorder variables
let mediaRecorder = null;
let audioStream = null;
let audioChunks = [];
let useAudioRecorder = false;

// DOM Elements
const recordButton = document.getElementById("record-button");
const recordIcon = document.getElementById("record-icon");
const recordState = document.getElementById("record-state");
const recordRing = document.getElementById("record-ring");
const transcriptTextarea = document.getElementById("transcript-text");
const asrModelSelect = document.getElementById("asr-model-select");
const llmModelSelect = document.getElementById("llm-model-select");
const systemPromptTextarea = document.getElementById("system-prompt");
const resetPromptBtn = document.getElementById("reset-prompt-btn");
const clearTranscriptBtn = document.getElementById("clear-transcript");
const generateNoteBtn = document.getElementById("generate-note-btn");
const generateSpinner = document.getElementById("generate-spinner");
const apiStatusBadge = document.getElementById("api-status");
const specialtySelect = document.getElementById("specialty-select");
const dictationLangSelect = document.getElementById("dictation-lang-select");
const demoBadge = document.getElementById("demo-badge");
const aiStatusBadge = document.getElementById("ai-status");

// Output Mode and Custom Prompt Elements
const outputModeSelect = document.getElementById("output-mode-select");
const customPromptGroup = document.getElementById("custom-prompt-group");
const customInstruction = document.getElementById("custom-instruction");
const customOutputContainer = document.getElementById("custom-output-container");
const sectionCustomOutput = document.getElementById("section-custom-output");
const noteSectionsContainer = document.getElementById("note-sections-container");
const noteWorkspaceTitle = document.getElementById("note-workspace-title");
const customOutputLabel = document.getElementById("custom-output-label");
const customOutputTitle = document.getElementById("custom-output-title");

let specialtyTemplates = {};
let hfTokenConfigured = false;

// Clinical Note Sections
const sectionChiefComplaint = document.getElementById("section-chief-complaint");
const sectionHpi = document.getElementById("section-hpi");
const sectionAssessment = document.getElementById("section-assessment");
const sectionPlan = document.getElementById("section-plan");
const sectionPrescription = document.getElementById("section-prescription");
const sectionRecommendedTests = document.getElementById("section-recommended-tests");
const sectionFollowUp = document.getElementById("section-follow-up");

// Audio Upload Elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("audio-file-input");
const progressContainer = document.getElementById("upload-progress-container");
const selectedFileName = document.getElementById("selected-file-name");
const selectedFileSize = document.getElementById("selected-file-size");
const uploadProgressBar = document.getElementById("upload-progress-bar");
const uploadStatusText = document.getElementById("upload-status-text");
const cancelUploadBtn = document.getElementById("cancel-upload");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    // 1. Fetch available models and configurations
    fetchSettings();

    // 2. Initialize Web Speech API (Dictation)
    initSpeechRecognition();

    // 3. Set Event Listeners
    setupEventListeners();

    // 4. Load cached data from localStorage
    loadCachedSettings();
});

// Check and toggle Demo Mode indicators
function checkDemoMode() {
    if (!hfTokenConfigured) {
        demoBadge.classList.remove("hidden");
        aiStatusBadge.className = "status-badge disconnected";
        aiStatusBadge.innerHTML = `<span class="status-dot"></span> ✗ AI Service Not Configured`;
    } else {
        demoBadge.classList.add("hidden");
        aiStatusBadge.className = "status-badge connected";
        aiStatusBadge.innerHTML = `<span class="status-dot"></span> ✓ AI Service Connected`;
    }
}

// Load Cached Settings from LocalStorage
function loadCachedSettings() {
    localStorage.removeItem("hf_token"); // Clean up old cached browser tokens for security
    if (localStorage.getItem("selected_llm")) {
        llmModelSelect.value = localStorage.getItem("selected_llm");
    }
    if (localStorage.getItem("selected_asr")) {
        asrModelSelect.value = localStorage.getItem("selected_asr");
    }
    if (localStorage.getItem("selected_specialty")) {
        specialtySelect.value = localStorage.getItem("selected_specialty");
    }
    if (localStorage.getItem("selected_dictation_lang")) {
        dictationLangSelect.value = localStorage.getItem("selected_dictation_lang");
    }
    if (localStorage.getItem("system_prompt")) {
        systemPromptTextarea.value = localStorage.getItem("system_prompt");
    }
    if (outputModeSelect && localStorage.getItem("output_mode")) {
        outputModeSelect.value = localStorage.getItem("output_mode");
    }
    if (customInstruction && localStorage.getItem("custom_prompt_text")) {
        customInstruction.value = localStorage.getItem("custom_prompt_text");
    }
    toggleOutputMode();
    checkDemoMode();
}

// Toggle layout based on Structured Note vs Custom Prompt vs Raw Transcript modes
function toggleOutputMode() {
    if (!outputModeSelect) return;
    
    const mode = outputModeSelect.value;
    const specialtyGroup = specialtySelect ? specialtySelect.closest('.input-group') : null;
    const systemPromptPanel = systemPromptTextarea ? systemPromptTextarea.closest('.panel-section') : null;
    const btnText = generateNoteBtn ? generateNoteBtn.querySelector('.btn-text') : null;
    
    if (mode === "custom") {
        if (customPromptGroup) customPromptGroup.classList.remove("hidden");
        if (customOutputContainer) customOutputContainer.classList.remove("hidden");
        
        if (noteSectionsContainer) noteSectionsContainer.classList.add("hidden");
        if (specialtyGroup) specialtyGroup.classList.add("hidden");
        if (systemPromptPanel) systemPromptPanel.classList.add("hidden");
        
        if (btnText) btnText.textContent = "Generate Custom Response";
        if (noteWorkspaceTitle) {
            noteWorkspaceTitle.innerHTML = `<i class="fa-solid fa-file-invoice"></i> Custom AI Response`;
        }
        if (customOutputLabel) customOutputLabel.textContent = "Output";
        if (customOutputTitle) customOutputTitle.textContent = "Custom Response";
        if (sectionCustomOutput) sectionCustomOutput.placeholder = "[Custom generated response will appear here...]";
    } else if (mode === "transcript") {
        if (customPromptGroup) customPromptGroup.classList.add("hidden");
        if (customOutputContainer) customOutputContainer.classList.remove("hidden");
        
        if (noteSectionsContainer) noteSectionsContainer.classList.add("hidden");
        if (specialtyGroup) specialtyGroup.classList.add("hidden");
        if (systemPromptPanel) systemPromptPanel.classList.add("hidden");
        
        if (btnText) btnText.textContent = "Show Raw Transcript";
        if (noteWorkspaceTitle) {
            noteWorkspaceTitle.innerHTML = `<i class="fa-solid fa-file-lines"></i> Raw Conversation Transcript`;
        }
        if (customOutputLabel) customOutputLabel.textContent = "Transcript";
        if (customOutputTitle) customOutputTitle.textContent = "Raw Dictation / Audio Output";
        if (sectionCustomOutput) sectionCustomOutput.placeholder = "[Raw transcript of the dictation will appear here...]";
    } else {
        if (customPromptGroup) customPromptGroup.classList.add("hidden");
        if (customOutputContainer) customOutputContainer.classList.add("hidden");
        
        if (noteSectionsContainer) noteSectionsContainer.classList.remove("hidden");
        if (specialtyGroup) specialtyGroup.classList.remove("hidden");
        if (systemPromptPanel) systemPromptPanel.classList.remove("hidden");
        
        if (btnText) btnText.textContent = "Generate Structured Clinical Note";
        if (noteWorkspaceTitle) {
            noteWorkspaceTitle.innerHTML = `<i class="fa-solid fa-file-medical"></i> Structured Clinical Note`;
        }
    }
}

// Fetch specialties templates from API
async function fetchSpecialties() {
    try {
        const response = await fetch("/api/settings/specialties");
        if (!response.ok) throw new Error("Could not fetch specialties metadata");
        specialtyTemplates = await response.json();
        
        // Populate specialty select
        specialtySelect.innerHTML = "";
        Object.entries(specialtyTemplates).forEach(([key, spec]) => {
            const option = document.createElement("option");
            option.value = key;
            option.textContent = spec.name;
            specialtySelect.appendChild(option);
        });
        
        // Set Default Specialty
        if (localStorage.getItem("selected_specialty") && specialtyTemplates[localStorage.getItem("selected_specialty")]) {
            specialtySelect.value = localStorage.getItem("selected_specialty");
        } else {
            specialtySelect.value = "general";
        }
        
        systemPromptDefault = specialtyTemplates["general"] ? specialtyTemplates["general"].prompt : "";
        
        if (!systemPromptTextarea.value) {
            const currentSpec = specialtySelect.value;
            systemPromptTextarea.value = specialtyTemplates[currentSpec] ? specialtyTemplates[currentSpec].prompt : systemPromptDefault;
        }
    } catch (err) {
        console.error("Failed to fetch specialties:", err);
    }
}

// Fetch settings from API
async function fetchSettings() {
    try {
        const response = await fetch("/api/settings/models");
        if (!response.ok) throw new Error("Could not fetch models metadata");
        const data = await response.json();
        
        // Populate LLM models
        llmModelSelect.innerHTML = "";
        Object.entries(data.llm_models).forEach(([key, model]) => {
            const option = document.createElement("option");
            option.value = key;
            option.textContent = model.name;
            llmModelSelect.appendChild(option);
        });

        // Populate ASR models
        asrModelSelect.innerHTML = "";
        Object.entries(data.asr_models).forEach(([key, model]) => {
            const option = document.createElement("option");
            option.value = key;
            option.textContent = model.name;
            asrModelSelect.appendChild(option);
        });

        // Set Default Selection if local cache is not set
        if (!localStorage.getItem("selected_llm")) {
            llmModelSelect.value = data.default_llm;
        }
        if (!localStorage.getItem("selected_asr")) {
            asrModelSelect.value = data.default_asr;
        }
        
        // Set configuration state based on backend response
        hfTokenConfigured = !!data.hf_token_configured;
        checkDemoMode();
        
        // Show a friendly warning alert in toast if token is not configured on backend
        if (!hfTokenConfigured) {
            showToast("AI Service Not Configured", "AI service is not configured. Please add HF_API_TOKEN to the server environment.", "info");
        }

        // Fetch specialties next
        await fetchSpecialties();

        apiStatusBadge.className = "status-badge connected";
        apiStatusBadge.innerHTML = `<span class="status-dot"></span> API Server Ready`;
    } catch (error) {
        console.error("Failed to fetch settings:", error);
        apiStatusBadge.className = "status-badge disconnected";
        apiStatusBadge.innerHTML = `<span class="status-dot"></span> Offline / Connection Error`;
        showToast("Connection Error", "Could not connect to backend server.", "error");
    }
}

// Initialize Speech Recognition
function initSpeechRecognition() {
    const isSafari = /^((?!chrome|android|crios).)*safari/i.test(navigator.userAgent);
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    // Fallback if Safari is detected, or SpeechRecognition is completely absent
    if (isSafari || !SpeechRecognition) {
        useAudioRecorder = true;
    }

    if (useAudioRecorder) {
        if (!window.MediaRecorder || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.warn("Audio recording is not supported in this browser.");
            recordState.textContent = "Audio recording is not supported in this browser. Please upload audio files.";
            recordButton.disabled = true;
            recordButton.style.opacity = "0.5";
            return;
        }
        recordState.textContent = "Click microphone to start recording";
        return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = dictationLangSelect.value || "en-US";

    recognition.onstart = () => {
        recognitionState = "recording";
        isRecording = true;
        recordIcon.className = "fa-solid fa-microphone-slash";
        recordButton.classList.add("recording");
        recordState.textContent = "Dictation active... Speak clearly into your mic.";
        showToast("Live Dictation Started", "Listening for your voice...", "info");
    };

    recognition.onresult = (event) => {
        let interimTranscript = "";
        let finalTranscript = "";

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + " ";
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }

        if (interimTranscript) {
            recordState.textContent = `Listening: "${interimTranscript}"`;
        } else {
            recordState.textContent = "Dictation active... Speak clearly into your mic.";
        }

        if (finalTranscript) {
            // Append final result to transcript text area
            transcriptTextarea.value += finalTranscript;
            // Trigger auto-scroll to bottom of textarea
            transcriptTextarea.scrollTop = transcriptTextarea.scrollHeight;
        }
    };

    recognition.onerror = (event) => {
        console.error("Speech Recognition Error:", event.error);
        if (event.error !== "no-speech") {
            showToast("Dictation Error", `Error: ${event.error}`, "error");
        }
    };

    recognition.onend = () => {
        recognitionState = "stopped";
        isRecording = false;
        recordIcon.className = "fa-solid fa-microphone";
        recordButton.classList.remove("recording");
        recordState.textContent = "Click microphone to resume live transcription";
    };
}

function startDictation() {
    if (recognition && recognitionState === "stopped") {
        try {
            recognitionState = "starting";
            recordState.textContent = "Starting microphone...";
            recognition.start();
        } catch (e) {
            console.error("Failed to start speech recognition:", e);
            recognitionState = "stopped";
        }
    }
}

function stopDictation() {
    if (recognition && (recognitionState === "recording" || recognitionState === "starting")) {
        try {
            recognitionState = "stopping";
            recordState.textContent = "Stopping dictation...";
            recognition.stop();
        } catch (e) {
            console.error("Failed to stop speech recognition:", e);
        }
    }
}

function startAudioRecording() {
    if (recognitionState !== "stopped") {
        console.warn("[AudioRecorder] startAudioRecording ignored. Current state:", recognitionState);
        return;
    }
    
    console.log("[AudioRecorder] startAudioRecording() initiated.");
    recognitionState = "starting";
    recordState.textContent = "Starting microphone...";
    
    console.log("[AudioRecorder] Requesting microphone permission via getUserMedia...");
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then((stream) => {
            console.log("[AudioRecorder] Microphone permission granted. Stream tracks:", stream.getTracks().map(t => `${t.kind}:${t.label}`));
            if (recognitionState !== "starting") {
                console.warn("[AudioRecorder] Stream obtained but state changed in-between. Stopping stream.");
                stream.getTracks().forEach(track => track.stop());
                return;
            }
            
            audioStream = stream;
            audioChunks = [];
            
            // Negotiate MIME type
            let options = {};
            const types = [
                "audio/webm;codecs=opus",
                "audio/webm",
                "audio/ogg;codecs=opus",
                "audio/mp4",
                "audio/aac",
                "audio/wav"
            ];
            let selectedType = "";
            for (const type of types) {
                if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(type)) {
                    selectedType = type;
                    options = { mimeType: type };
                    break;
                }
            }
            console.log("[AudioRecorder] Negotiated recording mimeType:", selectedType || "default");
            
            try {
                mediaRecorder = new MediaRecorder(stream, options);
                console.log("[AudioRecorder] MediaRecorder initialized successfully with options:", options);
            } catch (e) {
                console.warn("[AudioRecorder] Failed to initialize MediaRecorder with options, trying default constructor:", e);
                try {
                    mediaRecorder = new MediaRecorder(stream);
                    console.log("[AudioRecorder] Fallback MediaRecorder initialized successfully (default settings).");
                } catch (err2) {
                    console.error("[AudioRecorder] Fatal: Failed to initialize default MediaRecorder:", err2);
                    throw err2;
                }
            }
            
            mediaRecorder.ondataavailable = (event) => {
                console.log("[AudioRecorder] ondataavailable event fired. size:", event.data ? event.data.size : "undefined");
                if (event.data && event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = () => {
                console.log("[AudioRecorder] mediaRecorder.onstop event fired. Total chunks collected:", audioChunks.length);
                uploadRecordedAudio();
                if (audioStream) {
                    console.log("[AudioRecorder] Stopping audioStream tracks...");
                    audioStream.getTracks().forEach(track => track.stop());
                    audioStream = null;
                }
            };
            
            // Start recording with 1000ms timeslice to ensure ondataavailable fires periodically on Safari/iOS
            console.log("[AudioRecorder] Calling mediaRecorder.start(1000)...");
            mediaRecorder.start(1000);
            
            recognitionState = "recording";
            isRecording = true;
            recordIcon.className = "fa-solid fa-microphone-slash";
            recordButton.classList.add("recording");
            recordState.textContent = "Recording audio... Speak clearly into your mic.";
            showToast("Recording Started", "Recording your audio...", "info");
        })
        .catch((err) => {
            console.error("[AudioRecorder] getUserMedia / start recording failed:", err);
            recognitionState = "stopped";
            isRecording = false;
            recordIcon.className = "fa-solid fa-microphone";
            recordButton.classList.remove("recording");
            
            let errorMsg = "Could not access microphone.";
            if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                errorMsg = "Microphone permission denied. Please allow microphone access in your browser settings.";
            } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                errorMsg = "No microphone found on your device.";
            }
            
            showToast("Microphone Error", errorMsg, "error");
            recordState.textContent = "Click microphone to start recording";
        });
}

function stopAudioRecording() {
    console.log("[AudioRecorder] stopAudioRecording() initiated. Current state:", recognitionState);
    if (recognitionState !== "recording" && recognitionState !== "starting") {
        console.warn("[AudioRecorder] stopAudioRecording ignored. Not in active/starting state.");
        return;
    }
    
    recognitionState = "stopping";
    recordState.textContent = "Stopping recording...";
    
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        console.log("[AudioRecorder] Calling mediaRecorder.stop(). Current state:", mediaRecorder.state);
        mediaRecorder.stop();
    } else {
        console.warn("[AudioRecorder] mediaRecorder is inactive or null. Resetting UI manually.");
        recognitionState = "stopped";
        isRecording = false;
        recordIcon.className = "fa-solid fa-microphone";
        recordButton.classList.remove("recording");
        recordState.textContent = "Click microphone to start recording";
        
        if (audioStream) {
            console.log("[AudioRecorder] Stopping audioStream tracks...");
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }
    }
}

function uploadRecordedAudio() {
    console.log("[AudioRecorder] uploadRecordedAudio() called. Total chunks collected:", audioChunks.length);
    if (audioChunks.length === 0) {
        showToast("Recording Empty", "No audio recorded.", "error");
        console.error("[AudioRecorder] Error: No audio chunks captured.");
        resetAudioRecorderUI();
        return;
    }
    
    const mimeType = (mediaRecorder && mediaRecorder.mimeType) || "audio/mp4";
    console.log("[AudioRecorder] Creating Blob from chunks with mimeType:", mimeType);
    const audioBlob = new Blob(audioChunks, { type: mimeType });
    console.log("[AudioRecorder] Created Blob. Size:", audioBlob.size, "bytes");
    
    let extension = "mp4";
    if (mimeType.includes("webm")) {
        extension = "webm";
    } else if (mimeType.includes("ogg")) {
        extension = "ogg";
    } else if (mimeType.includes("wav")) {
        extension = "wav";
    } else if (mimeType.includes("aac")) {
        extension = "aac";
    }
    
    const filename = `recording.${extension}`;
    const file = new File([audioBlob], filename, { type: mimeType });
    
    recordState.textContent = "Transcribing recorded audio... Please wait.";
    showToast("Transcribing", "Uploading audio for transcription...", "info");
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_key", asrModelSelect.value);
    
    console.log("[AudioRecorder] Uploading file to /api/transcribe. Filename:", filename, "size:", file.size);
    
    fetch("/api/transcribe", {
        method: "POST",
        body: formData
    })
    .then(async (response) => {
        console.log("[AudioRecorder] Upload response received. Status:", response.status, "Ok:", response.ok);
        if (!response.ok) {
            const errText = await response.text();
            console.error("[AudioRecorder] Server error response:", errText);
            let detail = "Transcription service failed";
            try {
                const errData = JSON.parse(errText);
                detail = errData.detail || detail;
            } catch (e) {}
            throw new Error(detail);
        }
        return response.json();
    })
    .then((data) => {
        console.log("[AudioRecorder] Transcription response JSON:", data);
        if (data.transcript) {
            if (transcriptTextarea.value.trim() !== "") {
                transcriptTextarea.value += " " + data.transcript;
            } else {
                transcriptTextarea.value = data.transcript;
            }
            transcriptTextarea.scrollTop = transcriptTextarea.scrollHeight;
            
            // Prevent "Input Empty" trigger by dispatching input events
            transcriptTextarea.dispatchEvent(new Event('input'));
            
            showToast("Transcription Successful", "Audio recording transcribed successfully.", "success");
        } else {
            console.warn("[AudioRecorder] Warning: transcript field is empty or missing in response.");
            showToast("Transcription Empty", "No text was transcribed from the audio.", "warning");
        }
    })
    .catch((error) => {
        console.error("[AudioRecorder] Transcription Error:", error);
        showToast("Transcription Failed", error.message, "error");
    })
    .finally(() => {
        resetAudioRecorderUI();
    });
}

function resetAudioRecorderUI() {
    console.log("[AudioRecorder] Resetting audio recorder UI to stopped state.");
    recognitionState = "stopped";
    isRecording = false;
    recordIcon.className = "fa-solid fa-microphone";
    recordButton.classList.remove("recording");
    recordState.textContent = "Click microphone to start recording";
    audioChunks = [];
}

// UI Tab Switcher
function switchInputTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.add("hidden"));
    
    // Set active tab button
    const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find(btn => {
        return btn.getAttribute("onclick").includes(tabId);
    });
    if (activeBtn) activeBtn.classList.add("active");
    
    // Show tab content
    document.getElementById(tabId).classList.remove("hidden");
    
    // If switching away from live and recording, stop recording
    if (tabId !== 'live-tab' && isRecording) {
        if (useAudioRecorder) {
            stopAudioRecording();
        } else {
            stopDictation();
        }
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Record Button
    recordButton.addEventListener("click", () => {
        if (useAudioRecorder) {
            if (recognitionState === "stopped") {
                startAudioRecording();
            } else if (recognitionState === "recording" || recognitionState === "starting") {
                stopAudioRecording();
            }
        } else {
            if (recognitionState === "stopped") {
                startDictation();
            } else if (recognitionState === "recording" || recognitionState === "starting") {
                stopDictation();
            }
        }
    });

    // Save configuration settings changes locally
    llmModelSelect.addEventListener("change", () => {
        localStorage.setItem("selected_llm", llmModelSelect.value);
    });
    asrModelSelect.addEventListener("change", () => {
        localStorage.setItem("selected_asr", asrModelSelect.value);
    });
    specialtySelect.addEventListener("change", () => {
        const specKey = specialtySelect.value;
        localStorage.setItem("selected_specialty", specKey);
        if (specialtyTemplates[specKey]) {
            systemPromptTextarea.value = specialtyTemplates[specKey].prompt;
            localStorage.setItem("system_prompt", specialtyTemplates[specKey].prompt);
            showToast("Specialty Loaded", `${specialtyTemplates[specKey].name} prompt loaded.`, "success");
        }
    });
    dictationLangSelect.addEventListener("change", () => {
        const langKey = dictationLangSelect.value;
        localStorage.setItem("selected_dictation_lang", langKey);
        if (recognition) {
            recognition.lang = langKey;
        }
        showToast("Language Swapped", `Dictation set to ${dictationLangSelect.options[dictationLangSelect.selectedIndex].text}`, "success");
    });
    systemPromptTextarea.addEventListener("input", () => {
        localStorage.setItem("system_prompt", systemPromptTextarea.value);
    });

    if (outputModeSelect) {
        outputModeSelect.addEventListener("change", () => {
            localStorage.setItem("output_mode", outputModeSelect.value);
            toggleOutputMode();
        });
    }

    if (customInstruction) {
        customInstruction.addEventListener("input", () => {
            localStorage.setItem("custom_prompt_text", customInstruction.value);
        });
    }

    // Reset System Prompt
    resetPromptBtn.addEventListener("click", () => {
        if (confirm("Are you sure you want to reset the system prompt to default guidelines?")) {
            const specKey = specialtySelect.value;
            const defaultPrompt = (specialtyTemplates[specKey] && specialtyTemplates[specKey].prompt) || systemPromptDefault;
            systemPromptTextarea.value = defaultPrompt;
            localStorage.setItem("system_prompt", defaultPrompt);
            showToast("Reset Complete", "System prompt restored to template default.", "info");
        }
    });

    // Clear Transcript Textarea
    clearTranscriptBtn.addEventListener("click", () => {
        if (transcriptTextarea.value.trim() !== "") {
            if (confirm("Clear the raw conversation transcript?")) {
                transcriptTextarea.value = "";
                showToast("Cleared", "Conversation transcript cleared.", "info");
            }
        }
    });

    // Drag & Drop Listeners
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleAudioFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleAudioFile(e.target.files[0]);
        }
    });

    // Copy entire note
    document.getElementById("copy-entire-note").addEventListener("click", () => {
        copyClinicalNoteToClipboard();
    });

    // Generate Clinical Note
    generateNoteBtn.addEventListener("click", () => {
        triggerNoteGeneration();
    });
}

// Handle audio files upload
let uploadController = null;

function handleAudioFile(file) {
    // Validate size (max 25MB)
    const maxSize = 25 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast("File Too Large", "Maximum audio file size is 25MB.", "error");
        return;
    }

    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";
    
    // Show progress and hide upload zone
    dropZone.classList.add("hidden");
    progressContainer.classList.remove("hidden");
    uploadProgressBar.style.width = "0%";
    uploadStatusText.textContent = "Uploading audio...";

    // Prepare FormData
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_key", asrModelSelect.value);
    
    // Backend automatically reads API key from environment configuration

    // Cancel controller
    uploadController = new AbortController();

    // Start progress simulation since native fetch doesn't track upload progress directly in simple fetch
    let progress = 0;
    const progressInterval = setInterval(() => {
        if (progress < 85) {
            progress += Math.random() * 10;
            uploadProgressBar.style.width = Math.min(progress, 85) + "%";
            if (progress > 40) {
                uploadStatusText.textContent = "Transcribing consultation...";
            }
        }
    }, 400);

    // Call endpoint
    fetch("/api/transcribe", {
        method: "POST",
        body: formData,
        signal: uploadController.signal
    })
    .then(async (response) => {
        clearInterval(progressInterval);
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Transcription service failed");
        }
        return response.json();
    })
    .then((data) => {
        uploadProgressBar.style.width = "100%";
        uploadStatusText.textContent = "Transcription complete!";
        
        // Append transcription
        if (data.transcript) {
            if (transcriptTextarea.value.trim() !== "") {
                transcriptTextarea.value += "\n\n" + data.transcript;
            } else {
                transcriptTextarea.value = data.transcript;
            }
            showToast("Transcription Successful", "Audio file transcript added to workspace.", "success");
        }
        
        setTimeout(resetUploadUI, 1500);
    })
    .catch((error) => {
        clearInterval(progressInterval);
        if (error.name === "AbortError") {
            showToast("Upload Cancelled", "Transcription process was aborted.", "info");
        } else {
            console.error("Upload Error:", error);
            showToast("Transcription Failed", error.message, "error");
        }
        resetUploadUI();
    });

    cancelUploadBtn.onclick = () => {
        if (uploadController) {
            uploadController.abort();
        }
    };
}

function resetUploadUI() {
    dropZone.classList.remove("hidden");
    progressContainer.classList.add("hidden");
    fileInput.value = "";
    uploadController = null;
}

// Generate structured clinical note, custom response, or raw transcript from transcript text
async function triggerNoteGeneration() {
    const transcript = transcriptTextarea.value.trim();
    if (!transcript) {
        showToast("Input Empty", "Please type, record, or upload a transcript first.", "error");
        return;
    }

    const mode = outputModeSelect ? outputModeSelect.value : "structured";
    if (mode === "custom" && customInstruction && !customInstruction.value.trim()) {
        showToast("Instruction Empty", "Please enter a custom instruction.", "error");
        return;
    }

    if (!hfTokenConfigured && mode !== "transcript") {
        showToast("Demo Mode Active", "AI service is not configured. Running note simulation...", "info");
    }

    // Set Loading State
    generateNoteBtn.disabled = true;
    generateSpinner.classList.remove("hidden");
    
    let loadMsg = "Generating clinical note...";
    if (mode === "custom") {
        loadMsg = "Applying custom instructions...";
    } else if (mode === "transcript") {
        loadMsg = "Processing raw transcript...";
    }
    showToast(mode === "custom" ? "Generating Response" : (mode === "transcript" ? "Processing Transcript" : "Generating Clinical Note"), loadMsg, "info");

    try {
        const payload = {
            transcript: transcript,
            model_key: llmModelSelect.value,
            mode: mode
        };

        if (mode === "custom") {
            payload.custom_prompt = customInstruction.value.trim();
        } else {
            payload.system_prompt = systemPromptTextarea.value.trim();
        }

        const response = await fetch("/api/generate-note", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Note generation failed");
        }

        const data = await response.json();
        
        if (data.mode === "transcript") {
            sectionCustomOutput.value = data.transcript || "";
            showToast("Success", "Raw transcript processed successfully!", "success");
        } else if (data.mode === "custom") {
            sectionCustomOutput.value = data.output || "";
            showToast("Success", "Custom AI response generated successfully!", "success");
        } else {
            // Populating structured note fields
            const noteData = data.data || {};
            sectionChiefComplaint.value = noteData.chief_complaint || "";
            sectionHpi.value = noteData.hpi || "";
            sectionAssessment.value = noteData.assessment || "";
            sectionPlan.value = noteData.plan || "";
            sectionPrescription.value = noteData.prescription || "";
            sectionRecommendedTests.value = noteData.recommended_tests || "";
            sectionFollowUp.value = noteData.follow_up || "";
            showToast("Success", "Structured clinical note generated successfully!", "success");
        }
    } catch (error) {
        console.error("Generation Error:", error);
        showToast("Generation Failed", error.message, "error");
    } finally {
        generateNoteBtn.disabled = false;
        generateSpinner.classList.add("hidden");
    }
}

// Copy Clinical Note to Clipboard
function copyClinicalNoteToClipboard() {
    const noteContent = formatNoteForExport("txt");
    if (!noteContent.trim()) {
        const mode = outputModeSelect ? outputModeSelect.value : "structured";
        let emptyMsg = "Generate a clinical note before copying.";
        if (mode === "custom") {
            emptyMsg = "Generate a custom response before copying.";
        } else if (mode === "transcript") {
            emptyMsg = "Process the transcript before copying.";
        }
        showToast("Note Empty", emptyMsg, "error");
        return;
    }

    navigator.clipboard.writeText(noteContent)
        .then(() => {
            const mode = outputModeSelect ? outputModeSelect.value : "structured";
            let successMsg = "Entire clinical note copied to clipboard.";
            if (mode === "custom") {
                successMsg = "Custom response copied to clipboard.";
            } else if (mode === "transcript") {
                successMsg = "Raw transcript copied to clipboard.";
            }
            showToast("Copied", successMsg, "success");
        })
        .catch(err => {
            console.error("Clipboard Copy Error:", err);
            showToast("Copy Failed", "Could not copy note text.", "error");
        });
}

// Format Note for Export
function formatNoteForExport(format) {
    const mode = outputModeSelect ? outputModeSelect.value : "structured";

    if (mode === "custom" || mode === "transcript") {
        const text = sectionCustomOutput ? sectionCustomOutput.value.trim() : "";
        if (!text) {
            return "";
        }

        if (format === "json") {
            return JSON.stringify({
                output: text,
                mode: mode
            }, null, 4);
        }

        if (format === "md") {
            const heading = mode === "transcript" ? "Raw Conversation Transcript" : "Custom AI Response";
            return `# ${heading}\n\n${text}`;
        }

        return text;
    }

    const cc = sectionChiefComplaint.value.trim();
    const hpi = sectionHpi.value.trim();
    const assessment = sectionAssessment.value.trim();
    const plan = sectionPlan.value.trim();
    const prescription = sectionPrescription.value.trim();
    const tests = sectionRecommendedTests.value.trim();
    const followUp = sectionFollowUp.value.trim();

    if (!cc && !hpi && !assessment && !plan && !prescription && !tests && !followUp) {
        return "";
    }

    if (format === "json") {
        return JSON.stringify({
            chief_complaint: cc,
            hpi,
            assessment,
            plan,
            prescription,
            recommended_tests: tests,
            follow_up: followUp
        }, null, 4);
    }

    if (format === "md") {
        return `# Clinical Note\n\n` +
               `### Chief Complaint:\n${cc || 'N/A'}\n\n` +
               `### HPI:\n${hpi || 'N/A'}\n\n` +
               `### Assessment:\n${assessment || 'N/A'}\n\n` +
               `### Plan:\n${plan || 'N/A'}\n\n` +
               `### Prescription:\n${prescription || 'N/A'}\n\n` +
               `### Recommended Tests:\n${tests || 'N/A'}\n\n` +
               `### Follow-up:\n${followUp || 'N/A'}`;
    }

    // Default TXT Plain format
    return `Chief Complaint:\n${cc || ''}\n\n` +
           `HPI:\n${hpi || ''}\n\n` +
           `Assessment:\n${assessment || ''}\n\n` +
           `Plan:\n${plan || ''}\n\n` +
           `Prescription:\n${prescription || ''}\n\n` +
           `Recommended Tests:\n${tests || ''}\n\n` +
           `Follow-up:\n${followUp || ''}`;
}

// Export Clinical Note file download
function exportNote(format) {
    const text = formatNoteForExport(format);
    if (!text) {
        const mode = outputModeSelect ? outputModeSelect.value : "structured";
        let emptyMsg = "Generate a clinical note before exporting.";
        if (mode === "custom") {
            emptyMsg = "Generate a custom response before exporting.";
        } else if (mode === "transcript") {
            emptyMsg = "Process the transcript before exporting.";
        }
        showToast("Note Empty", emptyMsg, "error");
        return;
    }

    const mode = outputModeSelect ? outputModeSelect.value : "structured";
    let filenamePrefix = "clinical_note";
    if (mode === "custom") {
        filenamePrefix = "custom_response";
    } else if (mode === "transcript") {
        filenamePrefix = "raw_transcript";
    }
    const filename = `${filenamePrefix}_${new Date().toISOString().slice(0,10)}.${format}`;
    const mimeType = format === "json" ? "application/json" : "text/plain";
    
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast("Exported", `File exported as ${filename}`, "success");
}

// Password toggle helper
function togglePasswordVisibility(inputId) {
    const input = document.getElementById(inputId);
    const eyeIcon = document.getElementById("eye-icon");
    if (input.type === "password") {
        input.type = "text";
        eyeIcon.className = "fa-solid fa-eye-slash";
    } else {
        input.type = "password";
        eyeIcon.className = "fa-solid fa-eye";
    }
}

// Toast notification helper
function showToast(title, message, type = "info") {
    const container = document.getElementById("toast-container");
    
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    
    let iconClass = "fa-solid fa-circle-info";
    if (type === "success") iconClass = "fa-solid fa-circle-check";
    if (type === "error") iconClass = "fa-solid fa-triangle-exclamation";
    
    toast.innerHTML = `
        <div class="toast-icon"><i class="${iconClass}"></i></div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()"><i class="fa-solid fa-xmark"></i></button>
    `;
    
    container.appendChild(toast);
    
    // Automatically remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = "slideUp 0.3s reverse forwards";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
