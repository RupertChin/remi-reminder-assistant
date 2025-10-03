"""
Porcupine wake word detection.
Subscribes to audio pipeline and detects wake words with cooldown.
"""

import threading
import time
import numpy as np
from typing import Optional, Callable
import queue
import os
from config.logging_config import get_logger
from config import settings

logger = get_logger(__name__)

# Import Porcupine only if available
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False
    logger.warning("pvporcupine not available, wake word detection will be disabled")


class WakeWordDetector:
    """
    Porcupine wake word detector.
    Runs in dedicated worker thread and processes audio frames.
    """

    def __init__(self, access_key: str, keywords: list = None,
                 sensitivity: float = None, cooldown: float = None):
        """
        Initialize wake word detector.

        Args:
            access_key: Picovoice API key
            keywords: List of wake words (default from settings)
            sensitivity: Detection sensitivity 0.0-1.0 (default from settings)
            cooldown: Seconds before accepting another detection (default from settings)
        """
        if not PORCUPINE_AVAILABLE:
            raise RuntimeError("pvporcupine is not installed")

        self.access_key = access_key
        self.keywords = keywords or settings.WAKE_WORD_KEYWORDS
        self.sensitivity = sensitivity if sensitivity is not None else settings.WAKE_WORD_SENSITIVITY
        self.cooldown = cooldown if cooldown is not None else settings.WAKE_WORD_COOLDOWN

        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self.audio_queue: Optional[queue.Queue] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        # Cooldown tracking
        self.last_detection_time = 0.0

        # Audio buffer accumulation (Porcupine needs exact frame length)
        self.frame_buffer = np.array([], dtype=np.int16)
        self.frame_length = settings.PORCUPINE_FRAME_LENGTH

        # Callback for detections
        self.on_detection: Optional[Callable] = None

        logger.info(
            f"WakeWordDetector initialized: keywords={self.keywords}, "
            f"sensitivity={self.sensitivity}, cooldown={self.cooldown}s"
        )

    def initialize(self) -> bool:
        """
        Initialize Porcupine engine.

        Returns:
            True if successful
        """
        try:
            self.porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=self.keywords,
                sensitivities=[self.sensitivity] * len(self.keywords)
            )

            logger.info(
                f"Porcupine initialized: frame_length={self.porcupine.frame_length}, "
                f"sample_rate={self.porcupine.sample_rate}Hz"
            )

            # Verify sample rate matches our audio pipeline
            if self.porcupine.sample_rate != settings.AUDIO_SAMPLE_RATE:
                logger.error(
                    f"Sample rate mismatch: Porcupine={self.porcupine.sample_rate}Hz, "
                    f"Pipeline={settings.AUDIO_SAMPLE_RATE}Hz"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}", exc_info=True)
            return False

    def start(self, audio_queue: queue.Queue) -> bool:
        """
        Start wake word detection.

        Args:
            audio_queue: Queue receiving audio from pipeline manager

        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("Wake word detector already running")
            return True

        if self.porcupine is None:
            if not self.initialize():
                return False

        self.audio_queue = audio_queue
        self.running = True

        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

        logger.info("Wake word detection started")
        return True

    def stop(self) -> None:
        """Stop wake word detection."""
        self.running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

        logger.info("Wake word detection stopped")

    def _worker(self) -> None:
        """Worker thread that processes audio and detects wake words."""
        logger.debug("Wake word detector worker thread started")

        while self.running:
            try:
                # Get audio chunk from queue (blocking with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.5)

                # Accumulate audio until we have enough for Porcupine
                self.frame_buffer = np.append(self.frame_buffer, audio_chunk)

                # Process frames while we have enough data
                while len(self.frame_buffer) >= self.frame_length:
                    # Extract one frame
                    frame = self.frame_buffer[:self.frame_length]
                    self.frame_buffer = self.frame_buffer[self.frame_length:]

                    # Process with Porcupine
                    keyword_index = self.porcupine.process(frame)

                    if keyword_index >= 0:
                        # Wake word detected!
                        self._handle_detection(keyword_index)

            except queue.Empty:
                # No audio available, continue
                continue

            except Exception as e:
                logger.error(f"Error in wake word detector worker: {e}", exc_info=True)
                # Continue running despite errors

        logger.debug("Wake word detector worker thread stopped")

    def _handle_detection(self, keyword_index: int) -> None:
        """
        Handle wake word detection with cooldown.

        Args:
            keyword_index: Index of detected keyword
        """
        current_time = time.time()

        # Check cooldown
        if current_time - self.last_detection_time < self.cooldown:
            time_since = current_time - self.last_detection_time
            logger.debug(
                f"Wake word detected but in cooldown "
                f"({time_since:.1f}s < {self.cooldown}s)"
            )
            return

        # Update last detection time
        self.last_detection_time = current_time

        keyword = self.keywords[keyword_index]
        logger.info(f"Wake word detected: '{keyword}'")

        # Call callback if set
        if self.on_detection:
            try:
                self.on_detection(keyword, keyword_index)
            except Exception as e:
                logger.error(f"Error in detection callback: {e}", exc_info=True)

    def set_callback(self, callback: Callable) -> None:
        """
        Set callback function for wake word detections.

        Args:
            callback: Function(keyword: str, keyword_index: int) -> None
        """
        self.on_detection = callback
        logger.debug("Wake word detection callback set")

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop()

        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None

        logger.info("WakeWordDetector cleanup complete")
