#!/bin/bash
# Download AI models for Remi Voice Reminder Assistant

set -e

echo "========================================"
echo "Downloading AI Models"
echo "========================================"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
MODELS_DIR="$PROJECT_ROOT/data/models"

# Create models directory
mkdir -p "$MODELS_DIR"

# Download Vosk model (40MB, English)
echo ""
echo "Downloading Vosk speech recognition model..."
if [ ! -d "$MODELS_DIR/vosk-model-small-en-us-0.15" ]; then
    wget -q --show-progress -P "$MODELS_DIR" https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip -q "$MODELS_DIR/vosk-model-small-en-us-0.15.zip" -d "$MODELS_DIR"
    rm "$MODELS_DIR/vosk-model-small-en-us-0.15.zip"
    echo "✓ Vosk model downloaded"
else
    echo "✓ Vosk model already exists"
fi

# Download Piper TTS model
echo ""
echo "Downloading Piper TTS model..."
if [ ! -f "$MODELS_DIR/en_US-amy-medium.onnx" ]; then
    # Download model file from HuggingFace
    wget -q --show-progress \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx?download=true" \
        -O "$MODELS_DIR/en_US-amy-medium.onnx"

    # Download config file from HuggingFace
    wget -q --show-progress \
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json?download=true" \
        -O "$MODELS_DIR/en_US-amy-medium.onnx.json"

    echo "✓ Piper model downloaded"
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

# Start Ollama server if not running
echo ""
echo "Checking Ollama server status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server in background..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!

    # Wait for server to start (max 10 seconds)
    echo "Waiting for Ollama server to be ready..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✓ Ollama server is ready"
            break
        fi
        sleep 1
    done

    # Check if server actually started
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Error: Failed to start Ollama server"
        exit 1
    fi
else
    echo "✓ Ollama server already running"
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
