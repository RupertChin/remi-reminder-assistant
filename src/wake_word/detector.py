"""
openWakeWord wake word detection.
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

# Import openWakeWord only if available
try:
    from openwakeword.model import Model
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False
    logger.warning("openwakeword not available, wake word detection will be disabled")


class WakeWordDetector:
    """
    openWakeWord wake word detector.
    Runs in dedicated worker thread and processes audio frames.
    """

    def __init__(self, keywords: list = None,
                 threshold: float = None, cooldown: float = None):
        """
        Initialize wake word detector.

        Args:
            keywords: List of wake word models to load (default from settings)
            threshold: Detection threshold 0.0-1.0 (default from settings)
            cooldown: Seconds before accepting another detection (default from settings)
        """
        if not OPENWAKEWORD_AVAILABLE:
            raise RuntimeError("openwakeword is not installed")

        self.keywords = keywords or settings.WAKE_WORD_KEYWORDS
        self.threshold = threshold if threshold is not None else settings.WAKE_WORD_THRESHOLD
        self.cooldown = cooldown if cooldown is not None else settings.WAKE_WORD_COOLDOWN

        self.model: Optional[Model] = None
        self.audio_queue: Optional[queue.Queue] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False

        # Cooldown tracking
        self.last_detection_time = 0.0

        # Audio buffer accumulation (openWakeWord needs 1280 samples per frame @ 16kHz = 80ms)
        self.frame_buffer = np.array([], dtype=np.int16)
        self.frame_length = settings.WAKE_WORD_FRAME_LENGTH

        # Callback for detections
        self.on_detection: Optional[Callable] = None

        logger.info(
            f"WakeWordDetector initialized: keywords={self.keywords}, "
            f"threshold={self.threshold}, cooldown={self.cooldown}s"
        )

    def initialize(self) -> bool:
        """
        Initialize openWakeWord model.

        Returns:
            True if successful
        """
        try:
            # Download models if not present
            from openwakeword import utils
            models_dir = settings.MODELS_DIR / "openwakeword"
            models_dir.mkdir(exist_ok=True)

            # Download pre-trained models if needed
            if not any(models_dir.iterdir()):
                logger.info("Downloading openWakeWord models...")
                utils.download_models(models_dir)

            # Load the model
            # Use specific models if they match keywords, otherwise use all available
            model_paths = []
            for keyword in self.keywords:
                # Map keyword to model file
                model_file = models_dir / f"{keyword.replace(' ', '_')}.tflite"
                if model_file.exists():
                    model_paths.append(str(model_file))

            # If no specific models found, use default models
            if not model_paths:
                logger.info(f"No specific models found for {self.keywords}, using default models")
                # Try common alternatives
                default_models = ["alexa", "hey_jarvis"]
                for model_name in default_models:
                    model_file = models_dir / f"{model_name}.tflite"
                    if model_file.exists():
                        model_paths.append(str(model_file))
                        logger.info(f"Using {model_name} model as wake word")
                        break

            if not model_paths:
                logger.error("No wake word models found. Please train a custom model or use pre-trained ones.")
                return False

            self.model = Model(wakeword_models=model_paths)

            logger.info(f"openWakeWord initialized with models: {model_paths}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize openWakeWord: {e}", exc_info=True)
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

        if self.model is None:
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
        logger.info("Stopping wake word detection...")
        self.running = False

        # Clear the queue to unblock the worker thread
        if self.audio_queue:
            try:
                while not self.audio_queue.empty():
                    self.audio_queue.get_nowait()
            except:
                pass

        # Wait for thread with longer timeout
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning("Wake word detector thread did not stop cleanly")
            self.worker_thread = None

        logger.info("Wake word detection stopped")

    def _worker(self) -> None:
        """Worker thread that processes audio and detects wake words."""
        logger.debug("Wake word detector worker thread started")

        while self.running:
            try:
                # Get audio chunk from queue (blocking with timeout)
                audio_chunk = self.audio_queue.get(timeout=0.5)

                # Accumulate audio until we have enough for openWakeWord
                self.frame_buffer = np.append(self.frame_buffer, audio_chunk)

                # Process frames while we have enough data
                while len(self.frame_buffer) >= self.frame_length:
                    # Extract one frame
                    frame = self.frame_buffer[:self.frame_length]
                    self.frame_buffer = self.frame_buffer[self.frame_length:]

                    # Process with openWakeWord
                    # Model expects float32 normalized to [-1, 1]
                    frame_float = frame.astype(np.float32) / 32768.0

                    prediction = self.model.predict(frame_float)

                    # Check if any keyword exceeds threshold
                    for keyword, score in prediction.items():
                        if score >= self.threshold:
                            # Wake word detected!
                            self._handle_detection(keyword, score)
                            break

            except queue.Empty:
                # No audio available, continue
                continue

            except Exception as e:
                logger.error(f"Error in wake word detector worker: {e}", exc_info=True)
                # Continue running despite errors

        logger.debug("Wake word detector worker thread stopped")

    def _handle_detection(self, keyword: str, score: float) -> None:
        """
        Handle wake word detection with cooldown.

        Args:
            keyword: Detected keyword
            score: Detection score
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

        logger.info(f"Wake word detected: '{keyword}' (score: {score:.2f})")

        # Call callback if set
        if self.on_detection:
            try:
                # For compatibility, pass keyword and index (always 0 for openWakeWord)
                self.on_detection(keyword, 0)
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

        if self.model:
            # openWakeWord Model doesn't have explicit cleanup
            self.model = None

        logger.info("WakeWordDetector cleanup complete")
