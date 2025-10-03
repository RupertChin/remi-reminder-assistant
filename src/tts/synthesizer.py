"""
Piper TTS integration.
Generates speech audio and plays through pipeline manager.
"""

import asyncio
import numpy as np
from typing import Optional
from pathlib import Path
import json

from config.logging_config import get_logger
from config import settings

logger = get_logger(__name__)

# Import piper-tts only if available
try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    logger.warning("Piper TTS not available, speech output will be disabled")


class TTSSynthesizer:
    """
    Piper TTS synthesizer.
    Generates speech audio from text and handles playback queue.
    """

    def __init__(self, model_path: str = None, config_path: str = None,
                 pipeline_manager=None):
        """
        Initialize TTS synthesizer.

        Args:
            model_path: Path to Piper ONNX model (default from settings)
            config_path: Path to model config JSON (default from settings)
            pipeline_manager: Audio pipeline manager for playback
        """
        if not PIPER_AVAILABLE:
            raise RuntimeError("Piper TTS is not installed")

        self.model_path = model_path or str(settings.PIPER_MODEL_PATH)
        self.config_path = config_path or str(settings.PIPER_CONFIG_PATH)
        self.pipeline_manager = pipeline_manager

        self.voice: Optional[PiperVoice] = None

        # Queue for speech requests
        self.speech_queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
        self.running = False

        # Speech control
        self.current_speech_active = False

        logger.info(f"TTSSynthesizer initialized: model={self.model_path}")

    def initialize(self) -> bool:
        """
        Load Piper model.

        Returns:
            True if successful
        """
        try:
            logger.info("Loading Piper TTS model...")

            # Load model
            self.voice = PiperVoice.load(
                self.model_path,
                config_path=self.config_path,
                use_cuda=False  # Raspberry Pi doesn't have CUDA
            )

            logger.info(f"Piper model loaded: sample_rate={self.voice.config.sample_rate}Hz")

            # Verify sample rate matches our expectations
            if self.voice.config.sample_rate != settings.TTS_SAMPLE_RATE:
                logger.warning(
                    f"Sample rate mismatch: Piper={self.voice.config.sample_rate}Hz, "
                    f"Expected={settings.TTS_SAMPLE_RATE}Hz"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to load Piper model: {e}", exc_info=True)
            return False

    async def start(self) -> bool:
        """
        Start the TTS worker.

        Returns:
            True if started successfully
        """
        if self.running:
            logger.warning("TTS synthesizer already running")
            return True

        if self.voice is None:
            if not self.initialize():
                return False

        self.running = True
        self.worker_task = asyncio.create_task(self._worker())

        logger.info("TTS synthesizer started")
        return True

    async def stop(self) -> None:
        """Stop the TTS worker."""
        self.running = False

        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        logger.info("TTS synthesizer stopped")

    async def speak(self, text: str, priority: bool = False) -> None:
        """
        Queue text for speech synthesis.

        Args:
            text: Text to speak
            priority: If True, add to front of queue
        """
        if not text:
            return

        logger.debug(f"Queuing speech: '{text}' (priority={priority})")

        if priority:
            # For priority, we'd need a priority queue, but for simplicity
            # we'll just add to the queue normally
            # TODO: Implement proper priority queue if needed
            await self.speech_queue.put(text)
        else:
            await self.speech_queue.put(text)

    async def interrupt(self) -> None:
        """Interrupt current speech and clear queue."""
        logger.info("Interrupting speech")

        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Stop current playback
        if self.pipeline_manager:
            self.pipeline_manager.stop_playback()

        self.current_speech_active = False

    async def _worker(self) -> None:
        """Worker coroutine that processes speech queue."""
        logger.debug("TTS worker started")

        while self.running:
            try:
                # Get next speech request (with timeout)
                text = await asyncio.wait_for(
                    self.speech_queue.get(),
                    timeout=0.5
                )

                # Synthesize and play
                await self._synthesize_and_play(text)

            except asyncio.TimeoutError:
                # No speech requests, continue
                continue

            except asyncio.CancelledError:
                logger.debug("TTS worker cancelled")
                break

            except Exception as e:
                logger.error(f"Error in TTS worker: {e}", exc_info=True)

        logger.debug("TTS worker stopped")

    async def _synthesize_and_play(self, text: str) -> None:
        """
        Synthesize text to audio and play it.

        Args:
            text: Text to synthesize
        """
        try:
            self.current_speech_active = True
            logger.info(f"Synthesizing: '{text}'")

            # Generate audio using Piper
            # Piper returns a generator of audio chunks
            audio_chunks = []

            for audio_chunk in self.voice.synthesize(text):
                # audio_chunk is numpy array of int16 samples
                audio_chunks.append(audio_chunk)

            # Concatenate all chunks
            if audio_chunks:
                audio_data = np.concatenate(audio_chunks)

                logger.debug(f"Generated {len(audio_data)} samples")

                # Play through pipeline manager
                if self.pipeline_manager:
                    success = self.pipeline_manager.play_audio(
                        audio_data,
                        sample_rate=self.voice.config.sample_rate,
                        blocking=True  # Wait for speech to finish
                    )

                    if not success:
                        logger.error("Failed to play audio")
                else:
                    logger.warning("No pipeline manager available for playback")

            self.current_speech_active = False
            logger.debug("Speech synthesis complete")

        except Exception as e:
            logger.error(f"Speech synthesis error: {e}", exc_info=True)
            self.current_speech_active = False

    def is_speaking(self) -> bool:
        """
        Check if currently speaking.

        Returns:
            True if speech is active
        """
        return self.current_speech_active

    def get_queue_size(self) -> int:
        """
        Get number of queued speech requests.

        Returns:
            Queue size
        """
        return self.speech_queue.qsize()

    async def speak_and_wait(self, text: str) -> None:
        """
        Speak text and wait for completion.

        Args:
            text: Text to speak
        """
        await self.speak(text)

        # Wait for queue to empty and speech to complete
        while self.get_queue_size() > 0 or self.is_speaking():
            await asyncio.sleep(0.1)

    def cleanup(self) -> None:
        """Clean up resources."""
        self.voice = None
        logger.info("TTSSynthesizer cleanup complete")
