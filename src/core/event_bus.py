"""
Event bus for pub/sub messaging between components.
Enables loose coupling via asynchronous event-driven architecture.
"""

import asyncio
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from config.logging_config import get_logger
from config.settings import EventType

logger = get_logger(__name__)


@dataclass
class Event:
    """Event data structure."""
    event_type: EventType
    data: Any
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventBus:
    """
    Asynchronous event bus using pub/sub pattern.
    Supports event filtering and handles slow subscribers gracefully.
    """

    def __init__(self):
        """Initialize the event bus."""
        self.subscribers: Dict[EventType, List[asyncio.Queue]] = {}
        self.all_subscribers: List[asyncio.Queue] = []  # Subscribe to all events
        self.lock = asyncio.Lock()
        logger.info("EventBus initialized")

    async def subscribe(self, event_type: Optional[EventType] = None,
                       queue_size: int = 100) -> asyncio.Queue:
        """
        Subscribe to events.

        Args:
            event_type: Specific event type to subscribe to (None = all events)
            queue_size: Maximum queue size for buffering

        Returns:
            Queue that will receive events
        """
        queue = asyncio.Queue(maxsize=queue_size)

        async with self.lock:
            if event_type is None:
                # Subscribe to all events
                self.all_subscribers.append(queue)
                logger.debug("New subscriber added for ALL events")
            else:
                # Subscribe to specific event type
                if event_type not in self.subscribers:
                    self.subscribers[event_type] = []
                self.subscribers[event_type].append(queue)
                logger.debug(f"New subscriber added for {event_type.value}")

        return queue

    async def unsubscribe(self, queue: asyncio.Queue,
                         event_type: Optional[EventType] = None) -> None:
        """
        Unsubscribe from events.

        Args:
            queue: Queue to remove
            event_type: Event type to unsubscribe from (None = all)
        """
        async with self.lock:
            if event_type is None:
                # Remove from all subscribers
                if queue in self.all_subscribers:
                    self.all_subscribers.remove(queue)
                    logger.debug("Subscriber removed from ALL events")
            else:
                # Remove from specific event type
                if event_type in self.subscribers and queue in self.subscribers[event_type]:
                    self.subscribers[event_type].remove(queue)
                    logger.debug(f"Subscriber removed from {event_type.value}")

    async def publish(self, event_type: EventType, data: Any = None) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event
            data: Event data payload
        """
        event = Event(event_type=event_type, data=data)

        logger.debug(f"Publishing event: {event_type.value}")

        # Get all queues that should receive this event
        target_queues = []

        async with self.lock:
            # Add subscribers for this specific event type
            if event_type in self.subscribers:
                target_queues.extend(self.subscribers[event_type])

            # Add subscribers for all events
            target_queues.extend(self.all_subscribers)

        # Publish to all target queues (non-blocking)
        for q in target_queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    f"Subscriber queue full for {event_type.value}, "
                    "dropping event (slow subscriber)"
                )

    async def wait_for_event(self, event_type: EventType,
                            timeout: Optional[float] = None) -> Optional[Event]:
        """
        Wait for a specific event (one-time subscription).

        Args:
            event_type: Event type to wait for
            timeout: Optional timeout in seconds

        Returns:
            Event if received, None if timeout
        """
        queue = await self.subscribe(event_type, queue_size=10)

        try:
            if timeout:
                event = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                event = await queue.get()

            await self.unsubscribe(queue, event_type)
            return event

        except asyncio.TimeoutError:
            await self.unsubscribe(queue, event_type)
            logger.debug(f"Timeout waiting for {event_type.value}")
            return None

    def publish_sync(self, event_type: EventType, data: Any = None) -> None:
        """
        Synchronous wrapper for publish (for non-async code).

        Args:
            event_type: Type of event
            data: Event data payload
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is running, schedule the coroutine
                asyncio.create_task(self.publish(event_type, data))
            else:
                # If event loop is not running, run it
                loop.run_until_complete(self.publish(event_type, data))
        except RuntimeError:
            # No event loop, create a new one
            asyncio.run(self.publish(event_type, data))

    async def clear_all(self) -> None:
        """Clear all subscribers (for cleanup)."""
        async with self.lock:
            self.subscribers.clear()
            self.all_subscribers.clear()
            logger.info("All event bus subscribers cleared")
