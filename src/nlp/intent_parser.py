"""
Rule-based NLP intent parser.
Uses spaCy and dateutil for intent classification and entity extraction.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import difflib

from config.logging_config import get_logger
from config.settings import IntentType, NLP_CONFIDENCE_THRESHOLD, NLP_FUZZY_MATCH_THRESHOLD

logger = get_logger(__name__)


class IntentParser:
    """
    Rule-based intent parser using regex patterns and dateutil.
    """

    # Intent classification patterns
    SET_PATTERNS = [
        r'\b(set|create|add|make|schedule|remind)\b',
        r'\bremind\s+me\b',
        r'\breminder\s+(to|for)\b'
    ]

    LIST_PATTERNS = [
        r'\b(list|show|tell|what|display)\b.*\breminder',
        r'\breminders?\b.*\b(do|have|got)\b'
    ]

    DELETE_PATTERNS = [
        r'\b(delete|remove|cancel|clear)\b.*\breminder',
        r'\bforget\s+about\b'
    ]

    # Time expression patterns
    TIME_PATTERNS = {
        'absolute': r'\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b',
        'relative_time': r'\bin\s+(\d+)\s+(minute|hour|day|week|month)s?\b',
        'day_time': r'\b(morning|afternoon|evening|night|noon|midnight)\b',
        'relative_day': r'\b(today|tomorrow|tonight|yesterday)\b',
        'weekday': r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        'date': r'\b(\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b'
    }

    def __init__(self):
        """Initialize intent parser."""
        logger.info("Intent parser initialized")

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse text into intent and entities.

        Args:
            text: User input text

        Returns:
            Dictionary with intent, entities, confidence, raw_text
        """
        text = text.strip().lower()

        if not text:
            return self._create_result(IntentType.UNKNOWN, {}, 0.0, text)

        logger.debug(f"Parsing text: '{text}'")

        # Classify intent
        intent_type, intent_confidence = self._classify_intent(text)

        # Extract entities based on intent
        if intent_type == IntentType.SET_REMINDER:
            entities = self._extract_reminder_entities(text)
        elif intent_type == IntentType.LIST_REMINDERS:
            entities = {}
        elif intent_type == IntentType.DELETE_REMINDER:
            entities = self._extract_delete_entities(text)
        else:
            entities = {}

        overall_confidence = intent_confidence * entities.get('confidence', 1.0)

        result = self._create_result(intent_type, entities, overall_confidence, text)
        logger.info(f"Parse result: {intent_type.value}, confidence: {overall_confidence:.2f}")

        return result

    def _classify_intent(self, text: str) -> tuple:
        """
        Classify intent from text.

        Returns:
            (IntentType, confidence)
        """
        # Check SET patterns
        for pattern in self.SET_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return IntentType.SET_REMINDER, 0.9

        # Check LIST patterns
        for pattern in self.LIST_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return IntentType.LIST_REMINDERS, 0.9

        # Check DELETE patterns
        for pattern in self.DELETE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return IntentType.DELETE_REMINDER, 0.9

        return IntentType.UNKNOWN, 0.0

    def _extract_reminder_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract reminder content and scheduled time.

        Returns:
            Dictionary with 'content', 'scheduled_time', 'confidence'
        """
        # Extract time
        scheduled_time, time_confidence = self._extract_time(text)

        if scheduled_time is None:
            logger.warning("No time found in reminder text")
            return {
                'content': text,
                'scheduled_time': None,
                'confidence': 0.3
            }

        # Extract content by removing time expressions and intent keywords
        content = self._extract_reminder_content(text)

        return {
            'content': content,
            'scheduled_time': scheduled_time,
            'confidence': time_confidence
        }

    def _extract_time(self, text: str) -> tuple:
        """
        Extract datetime from text.

        Returns:
            (datetime, confidence)
        """
        now = datetime.now()

        # Try relative time (e.g., "in 30 minutes")
        match = re.search(self.TIME_PATTERNS['relative_time'], text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()

            if unit.startswith('minute'):
                scheduled_time = now + timedelta(minutes=value)
            elif unit.startswith('hour'):
                scheduled_time = now + timedelta(hours=value)
            elif unit.startswith('day'):
                scheduled_time = now + timedelta(days=value)
            elif unit.startswith('week'):
                scheduled_time = now + timedelta(weeks=value)
            elif unit.startswith('month'):
                scheduled_time = now + relativedelta(months=value)
            else:
                return None, 0.0

            logger.debug(f"Extracted relative time: {scheduled_time}")
            return scheduled_time, 0.95

        # Try relative day (e.g., "tomorrow")
        match = re.search(self.TIME_PATTERNS['relative_day'], text, re.IGNORECASE)
        if match:
            day_word = match.group(1).lower()

            if day_word == 'today':
                base_date = now
            elif day_word in ['tomorrow', 'tonight']:
                base_date = now + timedelta(days=1)
            elif day_word == 'yesterday':
                base_date = now - timedelta(days=1)
            else:
                base_date = now

            # Look for time of day
            time_match = re.search(self.TIME_PATTERNS['absolute'], text, re.IGNORECASE)
            if time_match:
                time_str = time_match.group(1)
                try:
                    parsed_time = date_parser.parse(time_str)
                    scheduled_time = base_date.replace(
                        hour=parsed_time.hour,
                        minute=parsed_time.minute,
                        second=0,
                        microsecond=0
                    )
                    logger.debug(f"Extracted day + time: {scheduled_time}")
                    return scheduled_time, 0.9
                except Exception as e:
                    logger.debug(f"Failed to parse time: {e}")

            # Default to specific time based on context
            default_hour = 9 if day_word == 'tomorrow' else 20  # 9am or 8pm
            scheduled_time = base_date.replace(
                hour=default_hour,
                minute=0,
                second=0,
                microsecond=0
            )
            logger.debug(f"Extracted relative day: {scheduled_time}")
            return scheduled_time, 0.7

        # Try absolute time (e.g., "3:00 pm")
        match = re.search(self.TIME_PATTERNS['absolute'], text, re.IGNORECASE)
        if match:
            time_str = match.group(1)
            try:
                parsed_time = date_parser.parse(time_str)
                scheduled_time = now.replace(
                    hour=parsed_time.hour,
                    minute=parsed_time.minute,
                    second=0,
                    microsecond=0
                )

                # If time has passed today, schedule for tomorrow
                if scheduled_time <= now:
                    scheduled_time += timedelta(days=1)

                logger.debug(f"Extracted absolute time: {scheduled_time}")
                return scheduled_time, 0.85
            except Exception as e:
                logger.debug(f"Failed to parse time: {e}")

        # Try day/time phrases (e.g., "morning", "afternoon")
        match = re.search(self.TIME_PATTERNS['day_time'], text, re.IGNORECASE)
        if match:
            day_time = match.group(1).lower()

            time_map = {
                'morning': 9,
                'noon': 12,
                'afternoon': 15,
                'evening': 18,
                'night': 20,
                'midnight': 0
            }

            hour = time_map.get(day_time, 12)
            scheduled_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)

            # If time has passed, schedule for tomorrow
            if scheduled_time <= now:
                scheduled_time += timedelta(days=1)

            logger.debug(f"Extracted day time: {scheduled_time}")
            return scheduled_time, 0.7

        # Try using dateutil as fallback
        try:
            parsed = date_parser.parse(text, fuzzy=True, default=now)
            if parsed > now:
                logger.debug(f"Extracted via dateutil: {parsed}")
                return parsed, 0.6
        except Exception:
            pass

        return None, 0.0

    def _extract_reminder_content(self, text: str) -> str:
        """
        Extract reminder content by removing time expressions and intent keywords.
        """
        content = text

        # Remove time patterns
        for pattern in self.TIME_PATTERNS.values():
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Remove intent keywords
        intent_words = r'\b(remind|reminder|set|create|add|make|schedule|to|me|about|for|that|at|on)\b'
        content = re.sub(intent_words, '', content, flags=re.IGNORECASE)

        # Clean up whitespace
        content = ' '.join(content.split())

        return content.strip() or "reminder"

    def _extract_delete_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract content to identify which reminder to delete.

        Returns:
            Dictionary with 'content', 'confidence'
        """
        # Remove delete keywords
        delete_words = r'\b(delete|remove|cancel|clear|forget|about|reminder)\b'
        content = re.sub(delete_words, '', text, flags=re.IGNORECASE)

        content = ' '.join(content.split()).strip()

        return {
            'content': content,
            'confidence': 0.8 if content else 0.3
        }

    def find_matching_reminder(self, query: str, reminders: List[str]) -> Optional[int]:
        """
        Find best matching reminder using fuzzy matching.

        Args:
            query: Query text
            reminders: List of reminder content strings

        Returns:
            Index of best match, or None if no good match
        """
        if not reminders:
            return None

        matches = difflib.get_close_matches(
            query,
            reminders,
            n=1,
            cutoff=NLP_FUZZY_MATCH_THRESHOLD
        )

        if matches:
            best_match = matches[0]
            index = reminders.index(best_match)
            logger.debug(f"Fuzzy match: '{query}' -> '{best_match}' (index {index})")
            return index

        return None

    def _create_result(self, intent: IntentType, entities: Dict,
                      confidence: float, raw_text: str) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            'intent': intent.value,
            'entities': entities,
            'confidence': confidence,
            'raw_text': raw_text
        }
