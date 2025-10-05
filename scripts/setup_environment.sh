#!/bin/bash
# Setup environment for Remi Voice Reminder Assistant on Raspberry Pi OS

set -e

echo "========================================"
echo "Remi Voice Reminder Assistant Setup"
echo "========================================"
echo ""

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    echo "Continuing anyway, but GPIO and hardware features may not work."
    echo ""
fi

# Update package list
echo "Updating package list..."
sudo apt-get update -qq

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt-get install -y -qq \
    portaudio19-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    unzip

echo "✓ System dependencies installed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt -q
echo "✓ Python dependencies installed"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ Created .env file"
    echo ""
else
    echo ""
    echo "✓ .env file already exists"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  2. Run: bash scripts/download_models.sh"
echo "  3. Run: source venv/bin/activate"
echo "  4. Run: python -m src.main"
echo ""
