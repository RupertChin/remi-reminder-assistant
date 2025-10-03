"""
Audio I/O manager (singleton).
Handles audio capture, distribution to subscribers, and playback.
"""

import pyaudio
import numpy as np
import threading
import queue
from typing import Optional, List, Callable, Dict
from config.logging_config import get_logger
from config import settings
from src.audio.audio_buffer import AudioBuffer

logger = get_logger(__name__)


class PipelineManager:
    """
    Singleton audio pipeline manager.
    Manages audio input/output streams and distributes audio to subscribers.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the pipeline manager."""
        # Only initialize once
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.pyaudio = pyaudio.PyAudio()

        # Audio input configuration
        self.sample_rate = settings.AUDIO_SAMPLE_RATE
        self.channels = settings.AUDIO_CHANNELS
        self.chunk_size = settings.AUDIO_CHUNK_SIZE

        # Streams
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None

        # Ring buffer for audio history
        self.ring_buffer = AudioBuffer(
            duration_seconds=settings.AUDIO_BUFFER_DURATION,
            sample_rate=self.sample_rate,
            channels=self.channels
        )

        # Subscriber management
        self.subscribers: Dict[str, queue.Queue] = {}
        self.subscriber_lock = threading.Lock()

        # Playback
        self.playback_active = False
        self.playback_lock = threading.Lock()

        logger.info("PipelineManager initialized")

    def enumerate_devices(self) -> Dict[str, List[Dict]]:
        """
        Enumerate all audio input and output devices.

        Returns:
            Dictionary with 'input' and 'output' device lists
        """
        devices = {'input': [], 'output': []}

        for i in range(self.pyaudio.get_device_count()):
            info = self.pyaudio.get_device_info_by_index(i)
            device_info = {
                'index': i,
                'name': info['name'],
                'max_input_channels': info['maxInputChannels'],
                'max_output_channels': info['maxOutputChannels'],
                'default_sample_rate': info['defaultSampleRate']
            }

            if info['maxInputChannels'] > 0:
                devices['input'].append(device_info)
            if info['maxOutputChannels'] > 0:
                devices['output'].append(device_info)

        logger.debug(f"Found {len(devices['input'])} input and {len(devices['output'])} output devices")
        return devices

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio callback for audio capture.
        Runs in a separate thread.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)

        # Store in ring buffer
        self.ring_buffer.write(audio_data)

        # Distribute to subscribers (non-blocking)
        with self.subscriber_lock:
            for name, q in list(self.subscribers.items()):
                try:
                    q.put_nowait(audio_data.copy())
                except queue.Full:
                    logger.warning(f"Subscriber '{name}' queue full, dropping frame")

        return (in_data, pyaudio.paContinue)

    def start_capture(self, input_device_index: Optional[int] = None) -> bool:
        """
        Start audio capture.

        Args:
            input_device_index: Specific input device to use (None = default)

        Returns:
            True if started successfully, False otherwise
        """
        if self.input_stream is not None:
            logger.warning("Capture already running")
            return True

        try:
            self.input_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            logger.info(f"Audio capture started (device: {input_device_index or 'default'})")
            return True

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}", exc_info=True)
            return False

    def stop_capture(self) -> None:
        """Stop audio capture."""
        if self.input_stream is not None:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            logger.info("Audio capture stopped")

    def subscribe(self, name: str, queue_size: int = 100) -> queue.Queue:
        """
        Subscribe to audio stream.

        Args:
            name: Unique subscriber name
            queue_size: Maximum queue size for buffering

        Returns:
            Queue that will receive audio data
        """
        with self.subscriber_lock:
            if name in self.subscribers:
                logger.warning(f"Subscriber '{name}' already exists, returning existing queue")
                return self.subscribers[name]

            q = queue.Queue(maxsize=queue_size)
            self.subscribers[name] = q
            logger.info(f"Subscriber '{name}' added")
            return q

    def unsubscribe(self, name: str) -> None:
        """
        Unsubscribe from audio stream.

        Args:
            name: Subscriber name to remove
        """
        with self.subscriber_lock:
            if name in self.subscribers:
                del self.subscribers[name]
                logger.info(f"Subscriber '{name}' removed")

    def play_audio(self, audio_data: np.ndarray, sample_rate: int = None,
                   blocking: bool = False) -> bool:
        """
        Play audio through output device.

        Args:
            audio_data: Audio samples as numpy array (int16)
            sample_rate: Sample rate of the audio (None = use TTS rate)
            blocking: If True, wait for playback to finish

        Returns:
            True if playback started successfully
        """
        if sample_rate is None:
            sample_rate = settings.TTS_SAMPLE_RATE

        try:
            with self.playback_lock:
                # Open output stream if not already open or sample rate changed
                if (self.output_stream is None or
                    self.output_stream._rate != sample_rate):
                    self.stop_playback()

                    self.output_stream = self.pyaudio.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        output=True,
                        frames_per_buffer=settings.OUTPUT_CHUNK_SIZE
                    )

                self.playback_active = True

                # Convert to bytes and play
                audio_bytes = audio_data.tobytes()

                if blocking:
                    self.output_stream.write(audio_bytes)
                else:
                    # Non-blocking write
                    self.output_stream.write(audio_bytes,
                                            exception_on_underflow=False)

                logger.debug(f"Playing {len(audio_data)} samples at {sample_rate}Hz")
                return True

        except Exception as e:
            logger.error(f"Playback error: {e}", exc_info=True)
            self.playback_active = False
            return False

    def stop_playback(self) -> None:
        """Stop audio playback and close output stream."""
        with self.playback_lock:
            if self.output_stream is not None:
                try:
                    self.output_stream.stop_stream()
                    self.output_stream.close()
                except Exception as e:
                    logger.error(f"Error stopping playback: {e}")
                finally:
                    self.output_stream = None

            self.playback_active = False
            logger.debug("Playback stopped")

    def clear_buffer(self) -> None:
        """Clear the audio ring buffer."""
        self.ring_buffer.clear()

    def get_buffer_data(self, seconds: float) -> np.ndarray:
        """
        Get historical audio data from ring buffer.

        Args:
            seconds: How many seconds of audio to retrieve

        Returns:
            Audio data as numpy array
        """
        return self.ring_buffer.read_last_n_seconds(seconds)

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up PipelineManager")
        self.stop_capture()
        self.stop_playback()

        with self.subscriber_lock:
            self.subscribers.clear()

        self.pyaudio.terminate()
        logger.info("PipelineManager cleanup complete")
