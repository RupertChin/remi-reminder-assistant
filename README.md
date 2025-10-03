# Remi Voice Reminder Assistant

A voice-controlled reminder system for Raspberry Pi 5. Say "Hey Remi" to create, list, and delete reminders using natural language. Everything runs locally—no internet required.

## What You Need

**Hardware:**
- Raspberry Pi 5 (8GB RAM required for LLM)
- USB microphone
- Speaker (3.5mm jack, HDMI, USB, or Bluetooth)
- LED (optional, connects to GPIO pin 17)
- MicroSD card (32GB+)

**Software:**
- Raspberry Pi OS (64-bit)
- Picovoice API key ([free tier available](https://console.picovoice.ai/))

## Setup

### 1. Download the Project

```bash
git clone <repository-url>
cd remi-reminder-assistant
```

### 2. Run the Setup Script

```bash
bash scripts/setup_environment.sh
```

This installs all dependencies and creates a virtual environment.

### 3. Add Your API Key

Get a free Picovoice API key at https://console.picovoice.ai/

Edit the `.env` file:
```bash
nano .env
```

Add your key:
```
PICOVOICE_ACCESS_KEY=your_key_here
```

Save and exit (Ctrl+X, then Y, then Enter).

### 4. Download AI Models

```bash
source venv/bin/activate
bash scripts/download_models.sh
```

This downloads and installs:
- Vosk speech recognition model (~40MB)
- Piper text-to-speech model (~20MB)
- spaCy NLP model (~12MB)
- Ollama LLM with Gemma 2B q4_K_M model (~2.5GB)

**Note:** The Gemma model download is large and may take 15-20 minutes depending on your internet speed. This higher-quality model enables more natural, conversational responses.

### 5. Start the Assistant

```bash
python -m src.main
```

You'll hear "Hello! I'm ready to help with reminders."

## How to Use

Just say **"Hey Remi"** and wait for the LED to turn on, then speak your command.

### Examples

**Create reminders:**
- "Set a reminder to call mom tomorrow at 3pm"
- "Remind me to take out the trash in 30 minutes"
- "Create a reminder for my dentist appointment Friday at 2pm"

**Check reminders:**
- "List my reminders"
- "What reminders do I have?"

**Delete reminders:**
- "Delete the reminder about calling mom"
- "Cancel my trash reminder"

### Natural Language Support

The assistant understands flexible time expressions:
- Relative: "in 30 minutes", "in 2 hours"
- Natural: "tomorrow", "Friday", "next Monday"
- Specific: "3pm", "9:30am"
- Time of day: "tomorrow morning", "tonight"

## Troubleshooting

### "Hey Remi" doesn't work

1. **Check your microphone**:
   ```bash
   python -m src.main --list-devices
   ```

2. **Select the right device**:
   ```bash
   python -m src.main --device 2
   ```
   (Replace 2 with your microphone's index)

3. **Increase sensitivity**: Edit `config/settings.py` and change:
   ```python
   WAKE_WORD_SENSITIVITY = 0.7  # Higher = more sensitive (max 1.0)
   ```

4. **Test your API key**: Check that `.env` has the correct key

### Commands not recognized

- Speak clearly and wait for the LED before talking
- Reduce background noise
- Move microphone closer
- Try simpler commands first

### No sound output

Check available speakers:
```bash
python -m src.main --list-devices
```

Make sure your speaker is selected as the default output in Raspberry Pi audio settings.

### System running slowly

- Close other applications
- Monitor resources: `htop`
- Reduce logging: remove `--debug` flag

## Advanced Options

### Run on Startup

Create a systemd service:
```bash
sudo nano /etc/systemd/system/remi.service
```

Add:
```ini
[Unit]
Description=Remi Voice Assistant
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/remi-reminder-assistant
ExecStart=/home/pi/remi-reminder-assistant/venv/bin/python -m src.main
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable remi.service
sudo systemctl start remi.service
```

### Configuration

Edit `config/settings.py` to customize:
- **Assistant personality**: `ASSISTANT_PERSONALITY = "friendly"` (options: friendly, professional, casual)
- Wake word sensitivity
- Speech timeout duration
- LED GPIO pin
- Audio sample rates
- LLM temperature (higher = more creative responses)

## Command Line Options

```bash
# Enable detailed logging
python -m src.main --debug

# List audio devices
python -m src.main --list-devices

# Use specific microphone
python -m src.main --device 2

# Run without LED (development mode)
python -m src.main --no-gpio
```

## How It Works

1. Listens for "Hey Remi" (wake word detection)
2. LED turns on when activated
3. Records your speech (offline speech recognition)
4. Understands your intent using rule-based NLP + LLM fallback
5. Creates/lists/deletes reminders (SQLite database)
6. **Generates natural response** using local LLM
7. Responds with conversational voice confirmation
8. Triggers reminders at scheduled times
9. Returns to idle (LED off)

**Natural Conversations:** The assistant uses a local Gemma 2B LLM to generate varied, natural responses instead of robotic templates. Responses will vary based on context and feel more like talking to a helpful friend. Response time is typically 5-10 seconds for the full interaction.

**Intent Understanding:** The system first tries fast rule-based parsing (~100ms). If it's not confident, it uses the local Gemma LLM for better accuracy with complex queries (~5s). All processing happens on your Raspberry Pi—no data leaves your device.

## Credits

Built with:
- [Picovoice](https://picovoice.ai/) - Wake word detection
- [Vosk](https://alphacephei.com/vosk/) - Speech recognition
- [Piper](https://github.com/rhasspy/piper) - Text-to-speech
- [spaCy](https://spacy.io/) - Natural language processing
- [Ollama](https://ollama.ai/) - Local LLM inference
