"""
GPIO LED controller for visual feedback.
Supports on/off and blinking patterns with async control.
"""

import asyncio
from typing import Optional

from config.logging_config import get_logger
from config import settings

logger = get_logger(__name__)

# Import RPi.GPIO only if available and hardware is enabled
GPIO_AVAILABLE = False
GPIO = None

if settings.ENABLE_HARDWARE_GPIO:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
    except ImportError:
        logger.warning("RPi.GPIO not available, LED control will be simulated")


class LEDController:
    """
    LED controller using GPIO.
    Falls back to simulation mode if GPIO is not available.
    """

    def __init__(self, pin: int = None):
        """
        Initialize LED controller.

        Args:
            pin: GPIO pin number (BCM mode) (default from settings)
        """
        self.pin = pin or settings.LED_GPIO_PIN
        self.simulation_mode = not GPIO_AVAILABLE

        self.is_on = False
        self.is_blinking = False
        self.blink_task: Optional[asyncio.Task] = None

        if not self.simulation_mode:
            self._initialize_gpio()
        else:
            logger.info(f"LED controller in SIMULATION mode (pin {self.pin})")

    def _initialize_gpio(self) -> None:
        """Initialize GPIO for LED control."""
        try:
            # Set mode to BCM (Broadcom pin numbering)
            GPIO.setmode(GPIO.BCM)

            # Disable warnings about pins already in use
            GPIO.setwarnings(False)

            # Setup pin as output
            GPIO.setup(self.pin, GPIO.OUT)

            # Ensure LED starts off
            GPIO.output(self.pin, GPIO.LOW)

            logger.info(f"LED controller initialized on GPIO pin {self.pin}")

        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}", exc_info=True)
            self.simulation_mode = True
            logger.info("Falling back to simulation mode")

    def on(self) -> None:
        """Turn LED on."""
        if self.is_blinking:
            self.stop_blink()

        if self.simulation_mode:
            logger.debug("LED: ON (simulated)")
        else:
            GPIO.output(self.pin, GPIO.HIGH)
            logger.debug("LED: ON")

        self.is_on = True

    def off(self) -> None:
        """Turn LED off."""
        if self.is_blinking:
            self.stop_blink()

        if self.simulation_mode:
            logger.debug("LED: OFF (simulated)")
        else:
            GPIO.output(self.pin, GPIO.LOW)
            logger.debug("LED: OFF")

        self.is_on = False

    async def blink(self, rate: float = None, duration: Optional[float] = None) -> None:
        """
        Start blinking LED.

        Args:
            rate: Blink rate in Hz (default from settings)
            duration: Optional duration in seconds (None = indefinite)
        """
        if rate is None:
            rate = settings.LED_BLINK_RATE

        if self.is_blinking:
            logger.debug("LED already blinking, restarting with new rate")
            self.stop_blink()

        self.is_blinking = True
        self.blink_task = asyncio.create_task(self._blink_worker(rate, duration))

        logger.debug(f"LED: BLINK started at {rate}Hz")

    def stop_blink(self) -> None:
        """Stop blinking LED."""
        if self.is_blinking:
            self.is_blinking = False

            if self.blink_task:
                self.blink_task.cancel()
                self.blink_task = None

            # Ensure LED is off
            self.off()

            logger.debug("LED: BLINK stopped")

    async def _blink_worker(self, rate: float, duration: Optional[float]) -> None:
        """
        Worker coroutine for blinking.

        Args:
            rate: Blink rate in Hz
            duration: Optional duration in seconds
        """
        interval = 1.0 / (rate * 2)  # Divide by 2 for on/off cycle
        start_time = asyncio.get_event_loop().time()

        try:
            while self.is_blinking:
                # Check duration
                if duration is not None:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration:
                        break

                # Toggle LED
                if self.simulation_mode:
                    logger.debug(f"LED: {'ON' if not self.is_on else 'OFF'} (simulated)")
                else:
                    GPIO.output(self.pin, GPIO.HIGH if not self.is_on else GPIO.LOW)

                self.is_on = not self.is_on

                # Wait for interval
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.debug("Blink worker cancelled")

        finally:
            self.is_blinking = False
            self.off()

    async def pulse(self, count: int = 3, rate: float = None) -> None:
        """
        Pulse LED a specific number of times.

        Args:
            count: Number of pulses
            rate: Blink rate in Hz (default from settings)
        """
        if rate is None:
            rate = settings.LED_BLINK_RATE

        interval = 1.0 / (rate * 2)

        for _ in range(count):
            self.on()
            await asyncio.sleep(interval)
            self.off()
            await asyncio.sleep(interval)

        logger.debug(f"LED: Pulsed {count} times")

    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        if self.is_blinking:
            self.stop_blink()

        self.off()

        if not self.simulation_mode and GPIO_AVAILABLE:
            try:
                GPIO.cleanup(self.pin)
                logger.info("LED GPIO cleanup complete")
            except Exception as e:
                logger.error(f"GPIO cleanup error: {e}")
