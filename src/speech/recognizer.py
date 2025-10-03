"""
Vosk speech-to-text recognition.
Activates after wake word and streams audio until silence timeout.
"""

import asyncio
import json
import numpy as np
import queue
from typing import Optional, Dict, Any
from config.logging_config import get_logger
from config import settings

logger = get_logger(__name__)

# Import Vosk only if available
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.warning("Vosk not available, speech recognition will be disabled")


class SpeechRecognizer:
    """
    Vosk-based speech recognition service.
    Streams audio and returns transcription with confidence.
    """

    def __init__(self, model_path: str = None, sample_rate: int = None):
        """
        Initialize speech recognizer.

        Args:
            model_path: Path to Vosk model (default from settings)
            sample_rate: Audio sample rate (default from settings)
        """
        if not VOSK_AVAILABLE:
            raise RuntimeError("Vosk is not installed")

        self.model_path = model_path or str(settings.VOSK_MODEL_PATH)
        self.sample_rate = sample_rate or settings.AUDIO_SAMPLE_RATE

        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None

        logger.info(f"SpeechRecognizer initialized: model_path={self.model_path}")

    def initialize(self) -> bool:
        """
        Load Vosk model (expensive operation, do once).

        Returns:
            True if successful
        """
        try:
            logger.info("Loading Vosk model (this may take a moment)...")
            self.model = Model(self.model_path)
            logger.info("Vosk model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}", exc_info=True)
            return False

    def create_recognizer(self) -> bool:
        """
        Create a new recognizer instance.

        Returns:
            True if successful
        """
        try:
            if self.model is None:
                if not self.initialize():
                    return False

            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetMaxAlternatives(0)  # Only get best result
            self.recognizer.SetWords(True)  # Include word-level timing

            logger.debug("Vosk recognizer created")
            return True

        except Exception as e:
            logger.error(f"Failed to create recognizer: {e}", exc_info=True)
            return False

    async def recognize_from_queue(
        self,
        audio_queue: queue.Queue,
        timeout: float = None
    ) -> Dict[str, Any]:
        """
        Recognize speech from audio queue with timeout.

        Args:
            audio_queue: Queue receiving audio chunks
            timeout: Silence timeout in seconds (default from settings)

        Returns:
            Dictionary with 'text', 'confidence', 'success' keys
        """
        if timeout is None:
            timeout = settings.SPEECH_TIMEOUT

        if self.recognizer is None:
            if not self.create_recognizer():
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': 'Failed to create recognizer'
                }

        # Reset recognizer for new recognition
        self.recognizer.Reset()

        logger.info(f"Starting speech recognition (timeout: {timeout}s)")

        partial_results = []
        last_audio_time = asyncio.get_event_loop().time()
        start_time = last_audio_time

        try:
            while True:
                current_time = asyncio.get_event_loop().time()

                # Check for overall timeout
                if current_time - start_time > timeout:
                    logger.info("Speech recognition timeout")
                    break

                # Get audio chunk (non-blocking with short timeout)
                try:
                    audio_chunk = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: audio_queue.get(timeout=0.1)
                    )
                except queue.Empty:
                    # Check for silence timeout
                    silence_duration = current_time - last_audio_time
                    if silence_duration > 2.0:  # 2 seconds of silence
                        logger.debug(f"Silence detected ({silence_duration:.1f}s)")
                        break
                    continue

                # Update last audio time
                last_audio_time = current_time

                # Convert numpy array to bytes
                if isinstance(audio_chunk, np.ndarray):
                    audio_bytes = audio_chunk.tobytes()
                else:
                    audio_bytes = audio_chunk

                # Process audio
                if self.recognizer.AcceptWaveform(audio_bytes):
                    # Final result available
                    result = json.loads(self.recognizer.Result())
                    if result.get('text'):
                        logger.debug(f"Final result: {result.get('text')}")
                        partial_results.append(result.get('text'))
                else:
                    # Partial result
                    partial = json.loads(self.recognizer.PartialResult())
                    if partial.get('partial'):
                        logger.debug(f"Partial: {partial.get('partial')}")

            # Get final result
            final_result = json.loads(self.recognizer.FinalResult())
            final_text = final_result.get('text', '')

            # Combine with any partial results
            if partial_results:
                all_text = ' '.join(partial_results + [final_text])
            else:
                all_text = final_text

            all_text = all_text.strip()

            # Calculate confidence (Vosk doesn't provide it directly)
            # Use a heuristic: longer transcriptions = higher confidence
            confidence = min(len(all_text.split()) / 10.0, 1.0) if all_text else 0.0

            logger.info(f"Recognition complete: '{all_text}' (confidence: {confidence:.2f})")

            return {
                'success': True,
                'text': all_text,
                'confidence': confidence,
                'error': None
            }

        except asyncio.CancelledError:
            logger.info("Speech recognition cancelled")
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': 'Cancelled'
            }

        except Exception as e:
            logger.error(f"Speech recognition error: {e}", exc_info=True)
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }

    async def recognize_from_data(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """
        Recognize speech from audio data array.

        Args:
            audio_data: Audio samples as numpy array (int16)

        Returns:
            Dictionary with 'text', 'confidence', 'success' keys
        """
        if self.recognizer is None:
            if not self.create_recognizer():
                return {
                    'success': False,
                    'text': '',
                    'confidence': 0.0,
                    'error': 'Failed to create recognizer'
                }

        self.recognizer.Reset()

        try:
            # Process audio
            audio_bytes = audio_data.tobytes()
            self.recognizer.AcceptWaveform(audio_bytes)

            # Get result
            result = json.loads(self.recognizer.FinalResult())
            text = result.get('text', '').strip()

            confidence = min(len(text.split()) / 10.0, 1.0) if text else 0.0

            logger.info(f"Recognition result: '{text}' (confidence: {confidence:.2f})")

            return {
                'success': True,
                'text': text,
                'confidence': confidence,
                'error': None
            }

        except Exception as e:
            logger.error(f"Speech recognition error: {e}", exc_info=True)
            return {
                'success': False,
                'text': '',
                'confidence': 0.0,
                'error': str(e)
            }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.recognizer = None
        self.model = None
        logger.info("SpeechRecognizer cleanup complete")
