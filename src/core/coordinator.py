"""
Main application coordinator.
Orchestrates all components and handles the main event loop.
"""

import asyncio
import os
from typing import Optional
from datetime import datetime

from config.logging_config import get_logger
from config.settings import SystemState, EventType, IntentType, NLP_CONFIDENCE_THRESHOLD
from src.core.event_bus import EventBus
from src.core.state_machine import StateMachine
from src.audio.pipeline_manager import PipelineManager
from src.wake_word.detector import WakeWordDetector
from src.speech.recognizer import SpeechRecognizer
from src.nlp.intent_parser import IntentParser
from src.nlp.llm_service import LLMService
from src.reminder.repository import ReminderRepository
from src.reminder.scheduler import ReminderScheduler
from src.reminder.models import Reminder
from src.tts.synthesizer import TTSSynthesizer
from src.hardware.led_controller import LEDController

logger = get_logger(__name__)


class Coordinator:
    """
    Main application coordinator.
    Initializes components, wires up event handlers, and manages the main loop.
    """

    def __init__(self):
        """Initialize coordinator."""
        logger.info("Initializing Coordinator")

        # Core components
        self.event_bus: Optional[EventBus] = None
        self.state_machine: Optional[StateMachine] = None

        # Audio/speech components
        self.pipeline_manager: Optional[PipelineManager] = None
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.speech_recognizer: Optional[SpeechRecognizer] = None

        # NLP components
        self.intent_parser: Optional[IntentParser] = None
        self.llm_service: Optional[LLMService] = None

        # Reminder components
        self.reminder_repo: Optional[ReminderRepository] = None
        self.scheduler: Optional[ReminderScheduler] = None

        # Output components
        self.tts: Optional[TTSSynthesizer] = None
        self.led: Optional[LEDController] = None

        # Runtime state
        self.running = False
        self.main_task: Optional[asyncio.Task] = None

    async def initialize(self, picovoice_key: str) -> bool:
        """
        Initialize all components in dependency order.

        Args:
            picovoice_key: Picovoice API key for wake word detection

        Returns:
            True if all components initialized successfully
        """
        try:
            logger.info("Initializing components...")

            # 1. Event bus
            self.event_bus = EventBus()

            # 2. State machine
            self.state_machine = StateMachine(self.event_bus)

            # 3. Audio pipeline
            self.pipeline_manager = PipelineManager()

            # 4. Wake word detector
            self.wake_word_detector = WakeWordDetector(
                access_key=picovoice_key
            )
            self.wake_word_detector.set_callback(self._on_wake_word_detected)

            # 5. Speech recognizer
            self.speech_recognizer = SpeechRecognizer()
            if not self.speech_recognizer.initialize():
                logger.warning("Speech recognizer failed to initialize")

            # 6. Intent parser
            self.intent_parser = IntentParser()

            # 7. LLM service (optional, may not be available)
            try:
                self.llm_service = LLMService()
                logger.info("LLM service initialized")
            except Exception as e:
                logger.warning(f"LLM service not available: {e}")
                self.llm_service = None

            # 8. Reminder repository
            self.reminder_repo = ReminderRepository()

            # 9. Scheduler
            self.scheduler = ReminderScheduler(callback=self._on_reminder_triggered)
            self.scheduler.start()

            # 10. TTS
            self.tts = TTSSynthesizer(pipeline_manager=self.pipeline_manager)
            await self.tts.start()

            # 11. LED controller
            self.led = LEDController()

            # Start audio capture
            if not self.pipeline_manager.start_capture():
                logger.error("Failed to start audio capture")
                return False

            # Subscribe wake word detector to audio
            audio_queue = self.pipeline_manager.subscribe("wake_word")
            self.wake_word_detector.start(audio_queue)

            # Restore scheduled reminders
            await self._restore_scheduled_reminders()

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    async def _restore_scheduled_reminders(self) -> None:
        """Restore scheduled reminders from database."""
        try:
            reminders = self.reminder_repo.get_all_active()
            now = datetime.now()

            for reminder in reminders:
                if reminder.scheduled_time > now:
                    self.scheduler.schedule_reminder(reminder)
                    logger.info(f"Restored scheduled reminder: {reminder.id}")
                else:
                    logger.debug(f"Skipping past reminder: {reminder.id}")

        except Exception as e:
            logger.error(f"Failed to restore reminders: {e}", exc_info=True)

    async def start(self) -> None:
        """Start the main coordinator loop."""
        if self.running:
            logger.warning("Coordinator already running")
            return

        self.running = True
        self.main_task = asyncio.create_task(self._main_loop())

        logger.info("Coordinator started")

        # Initial greeting
        await self.tts.speak("Hello! I'm ready to help with reminders.")

    async def stop(self) -> None:
        """Stop the coordinator."""
        logger.info("Stopping coordinator...")

        self.running = False

        # Cancel main task
        if self.main_task:
            self.main_task.cancel()
            try:
                await self.main_task
            except asyncio.CancelledError:
                pass

        # Cleanup components
        await self._cleanup()

        logger.info("Coordinator stopped")

    async def _cleanup(self) -> None:
        """Clean up all components."""
        logger.info("Cleaning up components...")

        if self.wake_word_detector:
            self.wake_word_detector.cleanup()

        if self.speech_recognizer:
            self.speech_recognizer.cleanup()

        if self.scheduler:
            self.scheduler.shutdown()

        if self.tts:
            await self.tts.stop()
            self.tts.cleanup()

        if self.led:
            self.led.cleanup()

        if self.pipeline_manager:
            self.pipeline_manager.cleanup()

        logger.info("Cleanup complete")

    async def _main_loop(self) -> None:
        """Main coordinator event loop."""
        logger.info("Main loop started")

        try:
            while self.running:
                # Main loop just keeps running; actual work is event-driven
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.debug("Main loop cancelled")

        logger.info("Main loop stopped")

    def _on_wake_word_detected(self, keyword: str, keyword_index: int) -> None:
        """
        Callback for wake word detection.

        Args:
            keyword: Detected keyword
            keyword_index: Index of the keyword
        """
        logger.info(f"Wake word callback: {keyword}")

        # Schedule async handler
        asyncio.create_task(self._handle_wake_word())

    async def _handle_wake_word(self) -> None:
        """Handle wake word detection."""
        try:
            # Transition to LISTENING state
            await self.state_machine.transition(SystemState.LISTENING)

            # Turn on LED
            self.led.on()

            # Publish event
            await self.event_bus.publish(EventType.WAKE_WORD_DETECTED)

            # Subscribe to audio for speech recognition
            speech_queue = self.pipeline_manager.subscribe("speech")

            # Recognize speech
            await self.state_machine.transition(SystemState.PROCESSING)

            result = await self.speech_recognizer.recognize_from_queue(speech_queue)

            # Unsubscribe from audio
            self.pipeline_manager.unsubscribe("speech")

            if result['success'] and result['text']:
                logger.info(f"Speech recognized: {result['text']}")

                # Publish event
                await self.event_bus.publish(
                    EventType.SPEECH_RECOGNIZED,
                    {'text': result['text'], 'confidence': result['confidence']}
                )

                # Parse intent
                await self._handle_speech(result['text'])

            else:
                logger.info("No speech recognized")
                await self.tts.speak("I didn't catch that. Please try again.")

                # Return to idle
                await self.state_machine.transition(SystemState.IDLE)
                self.led.off()

        except Exception as e:
            logger.error(f"Error handling wake word: {e}", exc_info=True)
            await self.state_machine.to_error({'error': str(e)})
            await self.tts.speak("Sorry, I encountered an error.")
            self.led.off()

    async def _handle_speech(self, text: str) -> None:
        """
        Handle recognized speech.

        Args:
            text: Recognized text
        """
        try:
            await self.state_machine.transition(SystemState.PARSING)

            # Parse intent
            intent_result = self.intent_parser.parse(text)

            # If confidence is low and LLM is available, try LLM fallback
            if intent_result['confidence'] < NLP_CONFIDENCE_THRESHOLD and self.llm_service:
                logger.info("Low confidence, trying LLM fallback")
                intent_result = await self.llm_service.parse_intent(text)

            logger.info(f"Intent: {intent_result['intent']}, confidence: {intent_result['confidence']:.2f}")

            # Publish event
            await self.event_bus.publish(EventType.INTENT_PARSED, intent_result)

            # Execute based on intent
            await self.state_machine.transition(SystemState.EXECUTING)

            if intent_result['intent'] == IntentType.SET_REMINDER.value:
                await self._handle_set_reminder(intent_result)

            elif intent_result['intent'] == IntentType.LIST_REMINDERS.value:
                await self._handle_list_reminders(intent_result)

            elif intent_result['intent'] == IntentType.DELETE_REMINDER.value:
                await self._handle_delete_reminder(intent_result)

            else:
                logger.warning(f"Unknown intent: {intent_result['intent']}")
                await self.tts.speak("I'm not sure what you want me to do.")

            # Return to idle
            await self.state_machine.transition(SystemState.IDLE)
            self.led.off()

        except Exception as e:
            logger.error(f"Error handling speech: {e}", exc_info=True)
            await self.state_machine.to_error({'error': str(e)})
            await self.tts.speak("Sorry, I couldn't process that.")
            self.led.off()

    async def _handle_set_reminder(self, intent_result: dict) -> None:
        """Handle SET_REMINDER intent."""
        entities = intent_result['entities']

        content = entities.get('content', 'reminder')
        scheduled_time = entities.get('scheduled_time')

        if not scheduled_time:
            # Generate natural error response
            response = await self._generate_response(
                action='error_no_time',
                user_request=intent_result['raw_text'],
                details={}
            )
            await self.tts.speak(response)
            return

        await self.state_machine.transition(SystemState.RESPONDING)

        # Create reminder in database
        reminder = self.reminder_repo.create(
            content=content,
            scheduled_time=scheduled_time,
            user_input=intent_result['raw_text']
        )

        # Schedule it
        self.scheduler.schedule_reminder(reminder)

        # Publish event
        await self.event_bus.publish(EventType.REMINDER_CREATED, {'reminder': reminder})

        # Generate natural confirmation response
        time_str = scheduled_time.strftime("%I:%M %p on %B %d")
        response = await self._generate_response(
            action='reminder_created',
            user_request=intent_result['raw_text'],
            details={
                'content': content,
                'time_str': time_str
            }
        )

        await self.tts.speak(response)
        logger.info(f"Created reminder: {reminder}")

    async def _handle_list_reminders(self, intent_result: dict) -> None:
        """Handle LIST_REMINDERS intent."""
        await self.state_machine.transition(SystemState.RESPONDING)

        reminders = self.reminder_repo.get_upcoming(limit=5)

        if not reminders:
            # Generate natural "no reminders" response
            response = await self._generate_response(
                action='no_reminders',
                user_request=intent_result['raw_text'],
                details={}
            )
            await self.tts.speak(response)
            return

        # Build reminders summary for LLM
        reminders_summary = []
        for reminder in reminders:
            time_str = reminder.scheduled_time.strftime("%I:%M %p on %B %d")
            reminders_summary.append(f"{reminder.content} at {time_str}")

        # Generate natural list response
        response = await self._generate_response(
            action='reminders_listed',
            user_request=intent_result['raw_text'],
            details={
                'count': len(reminders),
                'reminders_summary': '; '.join(reminders_summary)
            }
        )

        await self.tts.speak(response)

    async def _handle_delete_reminder(self, intent_result: dict) -> None:
        """Handle DELETE_REMINDER intent."""
        entities = intent_result['entities']
        content = entities.get('content', '')

        await self.state_machine.transition(SystemState.RESPONDING)

        # Get all active reminders
        reminders = self.reminder_repo.get_all_active()

        if not reminders:
            # Generate natural "no reminders" response
            response = await self._generate_response(
                action='no_reminders',
                user_request=intent_result['raw_text'],
                details={}
            )
            await self.tts.speak(response)
            return

        # Find matching reminder
        reminder_contents = [r.content for r in reminders]
        match_index = self.intent_parser.find_matching_reminder(content, reminder_contents)

        if match_index is not None:
            reminder = reminders[match_index]

            # Delete from database
            self.reminder_repo.delete(reminder.id)

            # Cancel scheduled job
            self.scheduler.cancel_reminder(reminder.id)

            # Publish event
            await self.event_bus.publish(EventType.REMINDER_DELETED, {'reminder': reminder})

            # Generate natural confirmation
            response = await self._generate_response(
                action='reminder_deleted',
                user_request=intent_result['raw_text'],
                details={
                    'content': reminder.content
                }
            )
            await self.tts.speak(response)
            logger.info(f"Deleted reminder: {reminder.id}")

        else:
            # Generate natural "not found" response
            response = await self._generate_response(
                action='error_no_match',
                user_request=intent_result['raw_text'],
                details={}
            )
            await self.tts.speak(response)

    async def _generate_response(self, action: str, user_request: str,
                                 details: dict) -> str:
        """
        Generate natural language response using LLM.

        Args:
            action: Type of action (e.g., 'reminder_created')
            user_request: Original user request
            details: Action-specific details

        Returns:
            Natural language response string
        """
        if self.llm_service:
            try:
                context = {
                    'action': action,
                    'user_request': user_request,
                    'details': details,
                    'personality': settings.ASSISTANT_PERSONALITY
                }
                return await self.llm_service.generate_response(context)
            except Exception as e:
                logger.error(f"LLM response generation failed: {e}", exc_info=True)
                # Fall through to default responses

        # Fallback if LLM not available or fails
        from src.nlp.llm_service import LLMService
        return LLMService._fallback_response(None, action, details)

    async def _on_reminder_triggered(self, reminder: Reminder) -> None:
        """
        Callback for triggered reminders.

        Args:
            reminder: Triggered reminder
        """
        logger.info(f"Reminder triggered: {reminder}")

        try:
            # Publish event
            await self.event_bus.publish(EventType.REMINDER_TRIGGERED, {'reminder': reminder})

            # Blink LED
            await self.led.blink(rate=1.0, duration=5.0)

            # Speak reminder
            message = f"Reminder: {reminder.content}"
            await self.tts.speak(message)

            # Mark as completed
            self.reminder_repo.mark_completed(reminder.id)

        except Exception as e:
            logger.error(f"Error triggering reminder: {e}", exc_info=True)
