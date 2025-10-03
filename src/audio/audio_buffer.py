"""
Thread-safe ring buffer for audio history.
Stores last N seconds of audio for wake word context.
"""

import threading
import numpy as np
from typing import Optional
from config.logging_config import get_logger

logger = get_logger(__name__)


class AudioBuffer:
    """Thread-safe circular buffer for audio data."""

    def __init__(self, duration_seconds: float, sample_rate: int, channels: int = 1):
        """
        Initialize audio ring buffer.

        Args:
            duration_seconds: How many seconds of audio to store
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
        """
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels

        # Calculate buffer size in samples
        self.capacity = int(duration_seconds * sample_rate * channels)
        self.buffer = np.zeros(self.capacity, dtype=np.int16)

        self.write_pos = 0
        self.lock = threading.Lock()

        logger.debug(
            f"AudioBuffer initialized: {duration_seconds}s capacity "
            f"({self.capacity} samples at {sample_rate}Hz)"
        )

    def write(self, data: np.ndarray) -> None:
        """
        Write audio data to the ring buffer.

        Args:
            data: Audio samples as numpy array (int16)
        """
        with self.lock:
            data_len = len(data)

            # Handle wrapping around the buffer
            space_to_end = self.capacity - self.write_pos

            if data_len <= space_to_end:
                # Data fits without wrapping
                self.buffer[self.write_pos:self.write_pos + data_len] = data
                self.write_pos = (self.write_pos + data_len) % self.capacity
            else:
                # Data wraps around
                self.buffer[self.write_pos:] = data[:space_to_end]
                remainder = data_len - space_to_end
                self.buffer[:remainder] = data[space_to_end:]
                self.write_pos = remainder

    def read_last_n_seconds(self, seconds: float) -> np.ndarray:
        """
        Read the last N seconds of audio from the buffer.

        Args:
            seconds: How many seconds of audio to retrieve

        Returns:
            Audio data as numpy array (int16)
        """
        with self.lock:
            num_samples = min(
                int(seconds * self.sample_rate * self.channels),
                self.capacity
            )

            # Calculate read start position
            read_start = (self.write_pos - num_samples) % self.capacity

            if read_start < self.write_pos:
                # Data is contiguous
                return self.buffer[read_start:self.write_pos].copy()
            else:
                # Data wraps around
                return np.concatenate([
                    self.buffer[read_start:],
                    self.buffer[:self.write_pos]
                ])

    def clear(self) -> None:
        """Clear the buffer (fill with zeros)."""
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            logger.debug("AudioBuffer cleared")

    def get_current_level(self) -> float:
        """
        Get current audio level (RMS).

        Returns:
            RMS amplitude (0.0 to 1.0)
        """
        with self.lock:
            # Calculate RMS of recent samples
            recent_samples = 1024
            start = max(0, self.write_pos - recent_samples)
            samples = self.buffer[start:self.write_pos]

            if len(samples) == 0:
                return 0.0

            # Normalize to 0.0-1.0 range
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            return min(rms / 32768.0, 1.0)  # int16 max = 32768
