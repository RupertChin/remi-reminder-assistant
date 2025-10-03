"""
Main entry point for Remi Voice Reminder Assistant.
Handles CLI arguments, environment setup, and application lifecycle.
"""

import asyncio
import signal
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

from config.logging_config import setup_logging, get_logger
from config import settings
from src.core.coordinator import Coordinator

# Setup logging first
logger = setup_logging("remi")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Remi Voice Reminder Assistant - Offline voice assistant for Raspberry Pi 5"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (use --list-devices to see available devices)"
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )

    parser.add_argument(
        "--no-gpio",
        action="store_true",
        help="Disable GPIO (LED) for development on non-Pi systems"
    )

    return parser.parse_args()


def load_environment():
    """
    Load environment variables from .env file.

    Returns:
        True if .env loaded successfully (or doesn't exist), False if critical vars missing
    """
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f".env file not found at {env_path}")
        logger.info("Create .env file with PICOVOICE_ACCESS_KEY for wake word detection")

    # Check for required environment variables
    picovoice_key = os.getenv("PICOVOICE_ACCESS_KEY")

    if not picovoice_key:
        logger.error("PICOVOICE_ACCESS_KEY not found in environment")
        logger.error("Please create .env file with: PICOVOICE_ACCESS_KEY=your_key_here")
        return False, None

    return True, picovoice_key


def list_audio_devices():
    """List available audio devices and exit."""
    from src.audio.pipeline_manager import PipelineManager

    logger.info("Available audio devices:")

    pipeline = PipelineManager()
    devices = pipeline.enumerate_devices()

    logger.info("\nInput Devices:")
    for device in devices['input']:
        logger.info(
            f"  [{device['index']}] {device['name']} "
            f"(channels: {device['max_input_channels']}, "
            f"rate: {device['default_sample_rate']}Hz)"
        )

    logger.info("\nOutput Devices:")
    for device in devices['output']:
        logger.info(
            f"  [{device['index']}] {device['name']} "
            f"(channels: {device['max_output_channels']}, "
            f"rate: {device['default_sample_rate']}Hz)"
        )

    pipeline.cleanup()


async def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel("DEBUG")
        logger.info("Debug logging enabled")

    # Disable GPIO if requested
    if args.no_gpio:
        settings.ENABLE_HARDWARE_GPIO = False
        logger.info("GPIO disabled")

    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return 0

    logger.info("=" * 60)
    logger.info("Remi Voice Reminder Assistant")
    logger.info("=" * 60)

    # Load environment
    env_ok, picovoice_key = load_environment()
    if not env_ok:
        return 1

    # Create coordinator
    coordinator = Coordinator()

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialize coordinator
        logger.info("Initializing application...")

        if not await coordinator.initialize(picovoice_key):
            logger.error("Failed to initialize application")
            return 1

        # Start coordinator
        await coordinator.start()

        logger.info("Application started successfully")
        logger.info("Say 'Hey Remi' to activate the assistant")
        logger.info("Press Ctrl+C to stop")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        # Stop coordinator
        logger.info("Shutting down...")
        await coordinator.stop()

    logger.info("Application stopped")
    return 0


if __name__ == "__main__":
    """Entry point when run directly."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
