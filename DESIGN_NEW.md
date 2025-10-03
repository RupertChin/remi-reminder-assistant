# Remi Voice Reminder Assistant - Technical Design Specification

> **🤖 OPTIMIZED FOR LLM/AGENTIC CODING TOOLS**  
> This document provides complete architectural specifications without implementation code. Use this as a blueprint to build the entire project in one pass.

---

## Project Overview

**Goal**: Build an offline voice assistant for Raspberry Pi 5 that manages reminders using natural language.

**Hardware Requirements**:
- Raspberry Pi 5 (8GB RAM)
- USB microphone
- Audio output (speaker/headphones)
- LED indicator (GPIO pin 17)

**User Experience**:
1. User says "Hey Remi"
2. LED lights up, system starts listening
3. User speaks command: "Set a reminder to call mom tomorrow at 3pm"
4. System confirms: "Reminder set for tomorrow at 3pm"
5. At scheduled time: LED blinks, system speaks "Reminder: call mom"

---

## Architecture Principles

### Design Philosophy
- **Modular**: Each component is independently testable/replaceable
- **Async-First**: All I/O operations use Python asyncio
- **Event-Driven**: Components communicate via event bus (loosely coupled)
- **Fail-Safe**: Graceful degradation when components fail
- **Offline-First**: All processing happens locally (privacy-focused)

### Technology Stack
- **Language**: Python 3.11+ with full type hints
- **Concurrency**: asyncio for non-blocking operations
- **Audio**: PyAudio for capture/playback
- **Wake Word**: Porcupine (Picovoice) - requires free API key
- **Speech-to-Text**: Vosk (offline, 40MB model)
- **NLP**: spaCy + dateutil (rule-based) → Ollama (LLM fallback)
- **LLM**: Ollama with Gemma 2B (quantized to ~1.5GB)
- **Database**: SQLite3
- **Scheduling**: APScheduler with persistent job store
- **TTS**: Piper (neural TTS, optimized for Pi)
- **GPIO**: RPi.GPIO for LED control

---

## Project Structure

```
remi-reminder-assistant/
├── .env                          # API keys (gitignored)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── config/
│   ├── settings.py               # All configuration constants
│   └── logging_config.py         # Structured logging setup
├── src/
│   ├── main.py                   # Application entry point
│   ├── audio/
│   │   ├── audio_buffer.py       # Thread-safe ring buffer
│   │   └── pipeline_manager.py   # Audio I/O singleton
│   ├── wake_word/
│   │   └── detector.py           # Porcupine integration
│   ├── speech/
│   │   └── recognizer.py         # Vosk STT service
│   ├── nlp/
│   │   ├── intent_parser.py      # Rule-based parsing
│   │   └── llm_service.py        # Ollama fallback
│   ├── reminder/
│   │   ├── models.py             # Data classes
│   │   ├── repository.py         # Database CRUD
│   │   └── scheduler.py          # APScheduler wrapper
│   ├── tts/
│   │   └── synthesizer.py        # Piper TTS integration
│   ├── hardware/
│   │   └── led_controller.py     # GPIO LED control
│   └── core/
│       ├── event_bus.py          # Pub/sub messaging
│       ├── state_machine.py      # System state management
│       └── coordinator.py        # Main orchestration
├── tests/                        # Unit & integration tests
├── data/
│   ├── models/                   # Downloaded AI models
│   └── reminders.db              # SQLite database (runtime)
├── logs/                         # Application logs (runtime)
└── scripts/
    ├── download_models.sh        # Download Vosk/Piper models
    └── setup_environment.sh      # System dependencies
```

---

## Component Specifications

### 1. Configuration System (`config/`)

**Purpose**: Centralize all settings in one place

**`settings.py`** - Define constants:
- Audio config: sample rate (16kHz), channels (mono), chunk size (1024)
- Wake word: keywords, sensitivity (0.5), cooldown (2s)
- Speech: timeout (10s), VAD threshold (0.6)
- NLP: confidence thresholds, fuzzy match threshold (0.6)
- LLM: Ollama host, model, temperature (0.3), timeout (30s)
- TTS: model path, speaking rate
- LED: GPIO pin (17), blink rate (2Hz)
- Database paths: reminders.db, jobs.db
- Enums: SystemState, EventType, IntentType

**`logging_config.py`** - Setup logging:
- Console handler with colored output
- File handler for persistent logs
- Silence noisy third-party libraries

---

### 2. Audio Pipeline (`src/audio/`)

**`audio_buffer.py`** - Ring buffer class:
- Thread-safe circular buffer for audio history
- Store last N seconds (configurable)
- Methods: write(), read_last_n_seconds(), clear(), get_current_level()

**`pipeline_manager.py`** - Audio I/O manager (singleton):
- Enumerate audio input/output devices
- Open PyAudio input stream with callback for capture
- Open PyAudio output stream for playback
- Distribute captured audio to multiple subscribers via queues (multicast)
- Ring buffer stores history for wake word context
- Methods: start_capture(), stop_capture(), subscribe(), unsubscribe(), play_audio(), stop_playback()

**Key Design Points**:
- PyAudio callback runs in separate thread for capture
- Non-blocking queue distribution to subscribers
- Handle queue full scenarios gracefully
- Output stream supports 22050Hz sample rate for TTS playback
- Auto-detect default output device (3.5mm jack/HDMI/USB/Bluetooth)
- Non-blocking writes to output stream prevent UI freeze
- Support buffer clearing for audio interruption

---

### 3. Wake Word Detection (`src/wake_word/`)

**`detector.py`** - Porcupine integration:
- Initialize Porcupine with API key and keywords
- Subscribe to audio pipeline queue
- Process audio in fixed-size frames (512 samples)
- Emit detection events via callback
- Implement cooldown to prevent double-triggering

**Key Design Points**:
- Runs in dedicated worker thread
- Accumulate audio chunks to match Porcupine frame length
- Sensitivity parameter balances accuracy vs false positives

---

### 4. Speech Recognition (`src/speech/`)

**`recognizer.py`** - Vosk STT service:
- Load Vosk model at startup (expensive, do once)
- Activate only after wake word detection
- Stream audio chunks to recognizer
- Detect speech end via silence timeout (asyncio)
- Return final transcription with confidence

**Key Design Points**:
- Use Vosk's partial results for real-time feedback
- Implement timeout using asyncio.wait_for()
- Handle cases where no speech detected

---

### 5. Intent Processing (`src/nlp/`)

**`intent_parser.py`** - Rule-based NLP:
- Use spaCy for tokenization and entity extraction
- Regex patterns for time expressions (absolute, relative, natural)
- dateutil.parser for flexible date parsing
- Intent classification by verb patterns:
  - SET: "set", "create", "remind", "add"
  - LIST: "list", "show", "what", "tell"
  - DELETE: "delete", "remove", "cancel"
- Extract reminder content by removing time/intent keywords
- Fuzzy matching for deletion (difflib)

**`llm_service.py`** - Ollama LLM fallback:
- Connect to Ollama HTTP API (localhost:11434)
- Structured prompts requesting JSON output
- Implement timeout and retry logic
- Parse LLM response into intent structure
- Cache common queries

**Output Format**:
```python
{
  "intent": "SET_REMINDER",
  "entities": {
    "content": "wash my dishes",
    "scheduled_time": datetime(2025, 10, 2, 17, 0, 0),
    "confidence": 0.92
  },
  "raw_text": "..."
}
```

---

### 6. Reminder Management (`src/reminder/`)

**`models.py`** - Data classes:
- Reminder dataclass with fields: id, content, scheduled_time, created_at, is_active, is_completed

**`repository.py`** - Database operations (repository pattern):
- SQLite schema with indexes on scheduled_time and is_active
- CRUD methods with parameterized queries
- Thread-safe connection handling
- Soft deletes for audit trail

**Schema**:
```sql
CREATE TABLE reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    scheduled_time DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    is_completed BOOLEAN DEFAULT 0,
    user_input TEXT
);
```

**`scheduler.py`** - APScheduler wrapper:
- Configure with SQLAlchemyJobStore for persistence
- DateTrigger for one-time reminders
- Execute reminders: trigger TTS + LED, mark completed
- Handle misfires with grace period
- Job recovery on startup

---

### 7. Text-to-Speech (`src/tts/`)

**`synthesizer.py`** - Piper TTS integration:
- Load ONNX model at startup
- Queue pattern for serialized speech output
- Generate 16-bit PCM audio at 22050Hz (Piper output format)
- Pass generated audio to pipeline_manager.play_audio() for playback
- Support volume and rate adjustment
- Implement speech interruption (clear queue + stop playback)

**Key Design Points**:
- Use direct ONNX inference for audio generation (numpy arrays)
- asyncio.Queue for speech requests
- Background worker processes queue
- Convert Piper output to format compatible with PyAudio
- Handle playback errors gracefully (fallback: log error, continue)

**Audio Output Flow**:
```
Text → Piper ONNX → numpy array (16-bit PCM, 22050Hz) →
pipeline_manager.play_audio() → PyAudio output stream →
Raspberry Pi audio output (3.5mm/HDMI/USB/Bluetooth)
```

---

### 8. Hardware Control (`src/hardware/`)

**`led_controller.py`** - GPIO LED management:
- Initialize GPIO pin in BCM mode
- Async blinking patterns (different rates for events)
- Methods: on(), off(), blink(), stop_blink()
- Clean up GPIO on shutdown

---

### 9. Core Orchestration (`src/core/`)

**`event_bus.py`** - Pub/sub messaging:
- asyncio.Queue-based event system
- Support event filtering by type
- Handle slow subscribers without blocking
- Event types: WAKE_WORD_DETECTED, SPEECH_RECOGNIZED, INTENT_PARSED, etc.

**`state_machine.py`** - System state management:
- Define states: IDLE, LISTENING, PROCESSING, PARSING, EXECUTING, RESPONDING, ERROR
- Validate state transitions
- Emit STATE_CHANGED events
- Provide state history for debugging

**`coordinator.py`** - Main orchestration logic:
- Initialize all components
- Wire up event handlers
- Main application loop
- Coordinate shutdown sequence

---

### 10. Main Entry Point (`src/main.py`)

**Responsibilities**:
- Parse command line arguments (--debug, --device, etc.)
- Setup logging
- Load environment variables (.env)
- Initialize components in dependency order:
  1. Config
  2. Event bus
  3. Audio pipeline
  4. Wake word detector
  5. Speech recognizer
  6. Intent parser
  7. LLM service
  8. Reminder repository
  9. Scheduler
  10. TTS
  11. LED controller
  12. State machine
  13. Coordinator
- Start main event loop
- Handle SIGINT/SIGTERM for graceful shutdown

---

## System Flow

### Happy Path: Set Reminder

```
1. System IDLE → Wake word detector listening
2. User: "Hey Remi"
3. Wake word detected → LISTENING state
4. LED turns on
5. Speech recognizer activated
6. User: "Set a reminder to call mom tomorrow at 3pm"
7. Speech timeout → PROCESSING state
8. Vosk transcribes text → PARSING state
9. Intent parser extracts:
   - Intent: SET_REMINDER
   - Content: "call mom"
   - Time: tomorrow 3pm
10. EXECUTING state → Save to database
11. Scheduler creates job for tomorrow 3pm
12. RESPONDING state → TTS generates audio
13. Audio plays through speaker: "Reminder set for tomorrow at 3pm"
14. LED turns off → Return to IDLE

### Scheduled Reminder Execution

1. Scheduler triggers at 3pm tomorrow
2. Retrieve reminder from database
3. LED starts blinking
4. TTS generates audio: "Reminder: call mom"
5. Audio plays through speaker via pipeline_manager.play_audio()
6. Mark reminder as completed
7. LED stops blinking
```

### Audio Output Details

**Raspberry Pi Audio Output Options**:
- 3.5mm audio jack (analog output)
- HDMI audio (if display connected)
- USB audio device (external speaker/DAC)
- Bluetooth speaker (if paired)

**PyAudio Auto-Detection**:
- Enumerates all output devices on startup
- Uses system default output device
- Can be overridden via config or CLI argument

**Audio Format**:
- Input (capture): 16kHz, 16-bit PCM, mono
- Output (playback): 22050Hz, 16-bit PCM, mono
- Different sample rates handled by separate streams

---

## Configuration Requirements

### Environment Variables (`.env`)
```bash
PICOVOICE_ACCESS_KEY=your_key_here
```

### Python Dependencies (`requirements.txt`)
```
python-dotenv==1.0.0
pyaudio==0.2.14
pvporcupine==3.0.0
vosk==0.3.45
spacy==3.7.2
python-dateutil==2.8.2
ollama==0.1.6
sqlalchemy==2.0.23
apscheduler==3.10.4
RPi.GPIO==0.7.1
colorlog==6.8.0
pytest==7.4.3
pytest-asyncio==0.21.1
```

### System Dependencies
```bash
# Raspberry Pi OS
sudo apt-get install portaudio19-dev python3-dev
```

### Model Downloads
```bash
# Vosk model (40MB)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -d data/models/

# Piper model
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en_US-amy-medium.tar.gz
tar -xzf voice-en_US-amy-medium.tar.gz -C data/models/

# spaCy model
python -m spacy download en_core_web_sm

# Ollama + Gemma
curl https://ollama.ai/install.sh | sh
ollama pull gemma:2b-instruct-q2_K
```

---

## Error Handling Strategy

### Component-Level
- Each component method should have try/except blocks
- Log errors with context (component name, operation)
- Return error results rather than raising exceptions
- Provide fallback behaviors where possible

### System-Level
- State machine handles ERROR state
- Event bus emits ERROR_OCCURRED events
- Coordinator can retry operations
- TTS announces errors to user when appropriate

### Examples
- Wake word fails → Log error, continue in IDLE
- Speech timeout → Return to IDLE, no error announcement
- Intent parsing fails → Try LLM fallback
- LLM timeout → Ask user to rephrase
- Database error → Announce "Unable to save reminder"

---

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock dependencies (database, audio, etc.)
- Test edge cases and error conditions

### Integration Tests
- Test component interactions
- Use test fixtures for audio/database
- Test full workflows (set/list/delete reminders)

### Manual Testing on Pi
- Test with various accents and speech patterns
- Verify wake word accuracy
- Test scheduler across reboots
- Check LED behavior
- Measure resource usage (CPU, memory)

---

## Performance Targets

- **Wake word latency**: <250ms from speech end
- **Speech recognition**: <1s for typical command
- **Intent parsing**: <100ms (rule-based), <5s (LLM fallback)
- **TTS latency**: <500ms to start speaking
- **CPU usage**: <10% idle, <30% during active processing
- **Memory usage**: <2GB total (including Ollama)

---

## Implementation Checklist

- [ ] Setup project structure and dependencies
- [ ] Implement configuration system
- [ ] Build audio pipeline with ring buffer (input capture)
- [ ] Add audio output (playback) to pipeline manager
- [ ] Test audio I/O on Raspberry Pi (verify device detection)
- [ ] Integrate Porcupine wake word detection
- [ ] Implement Vosk speech recognition
- [ ] Build rule-based intent parser
- [ ] Integrate Ollama LLM fallback
- [ ] Create database schema and repository
- [ ] Implement APScheduler wrapper
- [ ] Integrate Piper TTS
- [ ] Build LED controller
- [ ] Create event bus
- [ ] Implement state machine
- [ ] Build coordinator
- [ ] Create main entry point
- [ ] Add comprehensive error handling
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Test on Raspberry Pi hardware
- [ ] Optimize performance
- [ ] Document setup and usage

---

## Key Implementation Notes for LLM Tools

### Type Hints
- Use throughout: `def function(arg: str) -> bool:`
- Import from typing: Optional, List, Dict, Tuple, Callable

### Async Patterns
- Mark I/O functions as async: `async def read_data():`
- Use await for async calls: `result = await async_func()`
- Use asyncio.create_task() for concurrent operations
- Handle asyncio.CancelledError for cleanup

### Error Handling
- Use specific exceptions, not bare except
- Log errors with logger.error(msg, exc_info=True)
- Provide user-friendly error messages
- Implement retry logic with exponential backoff

### Resource Management
- Use context managers: `with resource:`
- Implement cleanup methods
- Handle shutdown signals (SIGINT, SIGTERM)
- Close connections and streams properly

### Logging
- Use module-level logger: `logger = logging.getLogger(__name__)`
- Log at appropriate levels: DEBUG, INFO, WARNING, ERROR
- Include context in log messages
- Avoid logging sensitive data

### Configuration
- All constants in settings.py
- Use environment variables for secrets
- Provide sensible defaults
- Document all configuration options

---

**This design specification is complete and ready for implementation by LLM/agentic coding tools.**
