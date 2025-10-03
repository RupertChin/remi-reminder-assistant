#!/bin/bash
# Download AI models for Remi Voice Reminder Assistant

set -e

echo "========================================"
echo "Downloading AI Models"
echo "========================================"

# Create models directory
MODELS_DIR="../data/models"
mkdir -p "$MODELS_DIR"

# Download Vosk model (40MB, English)
echo ""
echo "Downloading Vosk speech recognition model..."
if [ ! -d "$MODELS_DIR/vosk-model-small-en-us-0.15" ]; then
    cd "$MODELS_DIR"
    wget -q --show-progress https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip -q vosk-model-small-en-us-0.15.zip
    rm vosk-model-small-en-us-0.15.zip
    echo "✓ Vosk model downloaded"
    cd - > /dev/null
else
    echo "✓ Vosk model already exists"
fi

# Download Piper TTS model
echo ""
echo "Downloading Piper TTS model..."
if [ ! -f "$MODELS_DIR/en_US-amy-medium.onnx" ]; then
    cd "$MODELS_DIR"
    wget -q --show-progress https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en_US-amy-medium.tar.gz
    tar -xzf voice-en_US-amy-medium.tar.gz
    rm voice-en_US-amy-medium.tar.gz
    echo "✓ Piper model downloaded"
    cd - > /dev/null
else
    echo "✓ Piper model already exists"
fi

# Download spaCy model
echo ""
echo "Downloading spaCy NLP model..."
python -m spacy download en_core_web_sm --quiet 2>&1 | grep -v "Requirement already satisfied" || true
echo "✓ spaCy model downloaded"

# Install Ollama
echo ""
echo "========================================"
echo "Installing Ollama LLM"
echo "========================================"
echo ""

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama already installed"
else
    echo "Installing Ollama (this may take a few minutes)..."
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✓ Ollama installed"
fi

# Download Gemma model
echo ""
echo "Downloading Gemma 2B model (~2.5GB, this will take a while)..."

# Check if model already exists
if ollama list | grep -q "gemma:2b-instruct-q4_K_M"; then
    echo "✓ Gemma model already downloaded"
else
    ollama pull gemma:2b-instruct-q4_K_M
    echo "✓ Gemma model downloaded"
fi

echo ""
echo "========================================"
echo "All models downloaded successfully!"
echo "========================================"
echo ""
echo "The system will now use:"
echo "  - Rule-based parsing for common commands (fast)"
echo "  - LLM fallback for complex queries (more accurate)"
echo ""
