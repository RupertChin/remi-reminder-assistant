"""
Configuration settings for Remi Voice Reminder Assistant.
All constants and configuration values centralized here.
"""

from enum import Enum
from pathlib import Path
import os

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "reminders.db"
JOBS_DB_PATH = DATA_DIR / "jobs.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000  # 16kHz for input
AUDIO_CHANNELS = 1  # Mono
AUDIO_CHUNK_SIZE = 1024  # Samples per frame
AUDIO_FORMAT = "int16"  # 16-bit PCM
AUDIO_BUFFER_DURATION = 3  # Seconds of audio to keep in ring buffer

# Audio Output Configuration
TTS_SAMPLE_RATE = 22050  # 22.05kHz for TTS output (Piper default)
OUTPUT_CHUNK_SIZE = 1024

# Wake Word Configuration
WAKE_WORD_KEYWORDS = ["hey remi"]  # Porcupine wake words
WAKE_WORD_SENSITIVITY = 0.5  # 0.0 to 1.0, higher = more sensitive
WAKE_WORD_COOLDOWN = 2.0  # Seconds before accepting another wake word
PORCUPINE_FRAME_LENGTH = 512  # Porcupine requires 512 samples per frame

# Speech Recognition Configuration
SPEECH_TIMEOUT = 10.0  # Seconds of silence before stopping listening
SPEECH_VAD_THRESHOLD = 0.6  # Voice activity detection threshold
VOSK_MODEL_PATH = MODELS_DIR / "vosk-model-small-en-us-0.15"

# NLP Configuration
NLP_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for rule-based parsing
NLP_FUZZY_MATCH_THRESHOLD = 0.6  # For fuzzy string matching in deletion
SPACY_MODEL = "en_core_web_sm"

# LLM Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma:2b-instruct-q4_K_M"
OLLAMA_TEMPERATURE = 0.7  # Higher for more natural/varied responses
OLLAMA_TIMEOUT = 30.0  # Seconds

# Assistant Personality
ASSISTANT_PERSONALITY = "friendly"  # Options: friendly, professional, casual
ASSISTANT_NAME = "Remi"

# TTS Configuration
PIPER_MODEL_PATH = MODELS_DIR / "en_US-amy-medium.onnx"
PIPER_CONFIG_PATH = MODELS_DIR / "en_US-amy-medium.onnx.json"
TTS_SPEAKING_RATE = 1.0  # 1.0 = normal speed
TTS_VOLUME = 1.0  # 0.0 to 1.0

# LED Configuration
LED_GPIO_PIN = 17  # BCM pin numbering
LED_BLINK_RATE = 2.0  # Hz (blinks per second)
LED_REMINDER_BLINK_RATE = 1.0  # Hz for reminder alerts

# Scheduler Configuration
SCHEDULER_MISFIRE_GRACE_TIME = 300  # Seconds (5 minutes)
SCHEDULER_COALESCE = True  # Merge multiple pending executions
SCHEDULER_MAX_INSTANCES = 3  # Max concurrent reminder executions

# System Configuration
ENABLE_HARDWARE_GPIO = True  # Set to False for development on non-Pi systems
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"


# Enums for type safety
class SystemState(Enum):
    """System state machine states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    PARSING = "parsing"
    EXECUTING = "executing"
    RESPONDING = "responding"
    ERROR = "error"


class EventType(Enum):
    """Event bus event types."""
    WAKE_WORD_DETECTED = "wake_word_detected"
    SPEECH_RECOGNIZED = "speech_recognized"
    SPEECH_TIMEOUT = "speech_timeout"
    INTENT_PARSED = "intent_parsed"
    INTENT_FAILED = "intent_failed"
    REMINDER_CREATED = "reminder_created"
    REMINDER_DELETED = "reminder_deleted"
    REMINDER_TRIGGERED = "reminder_triggered"
    TTS_STARTED = "tts_started"
    TTS_FINISHED = "tts_finished"
    STATE_CHANGED = "state_changed"
    ERROR_OCCURRED = "error_occurred"
    SHUTDOWN_REQUESTED = "shutdown_requested"


class IntentType(Enum):
    """Intent classification types."""
    SET_REMINDER = "set_reminder"
    LIST_REMINDERS = "list_reminders"
    DELETE_REMINDER = "delete_reminder"
    UNKNOWN = "unknown"
