"""
Ollama LLM fallback service.
Used when rule-based parsing fails or has low confidence.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from datetime import datetime

from config.logging_config import get_logger
from config import settings

logger = get_logger(__name__)

# Import ollama only if available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available, LLM fallback will be disabled")


class LLMService:
    """
    Ollama LLM service for intent parsing fallback.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that extracts reminder information from user requests.
Your task is to identify:
1. The intent: "set_reminder", "list_reminders", "delete_reminder", or "unknown"
2. For set_reminder: the reminder content and scheduled time
3. For delete_reminder: the reminder content to match

Respond ONLY with valid JSON in this format:
{
  "intent": "set_reminder",
  "content": "call mom",
  "scheduled_time": "2025-10-02T15:00:00",
  "confidence": 0.9
}

For relative times, calculate from the current time. Use ISO 8601 format for datetime.
If you cannot determine the intent or time, set confidence below 0.5."""

    def __init__(self, host: str = None, model: str = None,
                 temperature: float = None, timeout: float = None):
        """
        Initialize LLM service.

        Args:
            host: Ollama API host (default from settings)
            model: Model name (default from settings)
            temperature: Temperature setting (default from settings)
            timeout: Request timeout (default from settings)
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama is not installed")

        self.host = host or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_MODEL
        self.temperature = temperature if temperature is not None else settings.OLLAMA_TEMPERATURE
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

        # Simple cache for common queries
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = 50

        logger.info(
            f"LLMService initialized: host={self.host}, "
            f"model={self.model}, temperature={self.temperature}"
        )

    async def parse_intent(self, text: str, current_time: datetime = None) -> Dict[str, Any]:
        """
        Parse intent using LLM.

        Args:
            text: User input text
            current_time: Current datetime for relative time calculation

        Returns:
            Dictionary with intent, entities, confidence, raw_text
        """
        if not text:
            return self._create_error_result(text, "Empty input")

        # Check cache
        cache_key = text.lower().strip()
        if cache_key in self.cache:
            logger.debug(f"Cache hit for: '{text}'")
            return self.cache[cache_key]

        if current_time is None:
            current_time = datetime.now()

        # Build user prompt
        user_prompt = f"""Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}

User request: "{text}"

Extract the reminder information and respond with JSON only."""

        try:
            logger.info(f"Querying LLM for: '{text}'")

            # Call Ollama API with timeout
            response = await asyncio.wait_for(
                self._call_ollama(user_prompt),
                timeout=self.timeout
            )

            # Parse response
            result = self._parse_llm_response(response, text)

            # Cache result
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = result

            return result

        except asyncio.TimeoutError:
            logger.error(f"LLM request timeout after {self.timeout}s")
            return self._create_error_result(text, "LLM timeout")

        except Exception as e:
            logger.error(f"LLM request failed: {e}", exc_info=True)
            return self._create_error_result(text, str(e))

    async def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama API.

        Args:
            prompt: User prompt

        Returns:
            Response text
        """
        try:
            # Use ollama python library
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': self.SYSTEM_PROMPT
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    options={
                        'temperature': self.temperature,
                    }
                )
            )

            return response['message']['content']

        except Exception as e:
            logger.error(f"Ollama API error: {e}", exc_info=True)
            raise

    def _parse_llm_response(self, response: str, raw_text: str) -> Dict[str, Any]:
        """
        Parse LLM JSON response.

        Args:
            response: LLM response text
            raw_text: Original user input

        Returns:
            Standardized result dictionary
        """
        try:
            # Try to extract JSON from response
            # LLM might include extra text, so find JSON block
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            # Extract fields
            intent = data.get('intent', 'unknown')
            content = data.get('content', '')
            scheduled_time_str = data.get('scheduled_time')
            confidence = float(data.get('confidence', 0.5))

            # Parse datetime if present
            scheduled_time = None
            if scheduled_time_str:
                try:
                    scheduled_time = datetime.fromisoformat(scheduled_time_str)
                except Exception as e:
                    logger.warning(f"Failed to parse datetime: {e}")

            # Build entities
            entities = {}
            if content:
                entities['content'] = content
            if scheduled_time:
                entities['scheduled_time'] = scheduled_time
            entities['confidence'] = confidence

            logger.info(f"LLM parsed: intent={intent}, confidence={confidence:.2f}")

            return {
                'intent': intent,
                'entities': entities,
                'confidence': confidence,
                'raw_text': raw_text
            }

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}", exc_info=True)
            logger.debug(f"Response was: {response}")
            return self._create_error_result(raw_text, f"Parse error: {e}")

    def _create_error_result(self, raw_text: str, error: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'intent': 'unknown',
            'entities': {},
            'confidence': 0.0,
            'raw_text': raw_text,
            'error': error
        }

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.cache.clear()
        logger.debug("LLM cache cleared")

    async def generate_response(self, context: Dict[str, Any]) -> str:
        """
        Generate a natural language response based on context.

        Args:
            context: Dictionary containing:
                - action: What action was taken (e.g., "reminder_created")
                - user_request: Original user input
                - details: Action-specific details (reminder info, etc.)
                - personality: Tone to use (friendly, professional, casual)

        Returns:
            Natural language response string
        """
        action = context.get('action', 'unknown')
        user_request = context.get('user_request', '')
        details = context.get('details', {})
        personality = context.get('personality', settings.ASSISTANT_PERSONALITY)

        # Build personality-specific system prompt
        personality_prompts = {
            'friendly': f"You are {settings.ASSISTANT_NAME}, a friendly and warm voice assistant. Respond naturally and conversationally, like talking to a friend.",
            'professional': f"You are {settings.ASSISTANT_NAME}, a professional voice assistant. Be clear, concise, and courteous.",
            'casual': f"You are {settings.ASSISTANT_NAME}, a casual and relaxed voice assistant. Keep it brief and natural, like a helpful roommate."
        }

        system_prompt = personality_prompts.get(personality, personality_prompts['friendly'])
        system_prompt += "\n\nRespond with ONLY the spoken response - no explanations, no JSON, just what you would say out loud. Keep it under 30 words."

        # Build user prompt based on action
        user_prompt = self._build_response_prompt(action, user_request, details)

        try:
            logger.debug(f"Generating response for action: {action}")

            # Call Ollama API
            response = await asyncio.wait_for(
                self._call_ollama_for_response(system_prompt, user_prompt),
                timeout=self.timeout
            )

            # Clean up response (remove quotes, extra whitespace)
            response = response.strip().strip('"').strip("'").strip()

            logger.info(f"Generated response: '{response}'")
            return response

        except asyncio.TimeoutError:
            logger.error(f"Response generation timeout after {self.timeout}s")
            return self._fallback_response(action, details)

        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            return self._fallback_response(action, details)

    def _build_response_prompt(self, action: str, user_request: str, details: Dict) -> str:
        """Build the user prompt for response generation."""
        prompts = {
            'reminder_created': f"""The user said: "{user_request}"
You successfully created a reminder:
- Content: {details.get('content', 'reminder')}
- Time: {details.get('time_str', 'soon')}

Generate a brief, natural confirmation response.""",

            'reminders_listed': f"""The user asked: "{user_request}"
You have this information:
- Number of reminders: {details.get('count', 0)}
- Reminders: {details.get('reminders_summary', 'none')}

Generate a natural response listing their reminders.""",

            'reminder_deleted': f"""The user said: "{user_request}"
You successfully deleted the reminder: {details.get('content', 'reminder')}

Generate a brief confirmation.""",

            'no_reminders': f"""The user asked: "{user_request}"
They have no reminders.

Generate a brief, friendly response.""",

            'error_no_time': f"""The user said: "{user_request}"
You couldn't understand when they want the reminder.

Ask them politely to specify the time.""",

            'error_no_match': f"""The user said: "{user_request}"
You couldn't find a matching reminder to delete.

Tell them politely that you couldn't find it.""",

            'error_general': f"""The user said: "{user_request}"
An error occurred processing their request.

Apologize briefly and ask them to try again."""
        }

        return prompts.get(action, f"The user said: '{user_request}'. Respond appropriately.")

    async def _call_ollama_for_response(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama for response generation."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ollama.chat(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': system_prompt
                        },
                        {
                            'role': 'user',
                            'content': user_prompt
                        }
                    ],
                    options={
                        'temperature': self.temperature,
                    }
                )
            )

            return response['message']['content']

        except Exception as e:
            logger.error(f"Ollama response generation error: {e}", exc_info=True)
            raise

    def _fallback_response(self, action: str, details: Dict) -> str:
        """Provide fallback responses if LLM fails."""
        fallbacks = {
            'reminder_created': f"Reminder set for {details.get('time_str', 'the scheduled time')}",
            'reminders_listed': f"You have {details.get('count', 0)} upcoming reminders",
            'reminder_deleted': "Reminder deleted",
            'no_reminders': "You have no upcoming reminders",
            'error_no_time': "I couldn't understand when you want the reminder",
            'error_no_match': "I couldn't find that reminder",
            'error_general': "Sorry, I encountered an error"
        }

        return fallbacks.get(action, "I'm not sure how to respond to that")
