"""
System state machine.
Manages state transitions and validates state changes.
"""

import asyncio
from typing import Optional, List, Tuple
from datetime import datetime
from config.logging_config import get_logger
from config.settings import SystemState, EventType
from src.core.event_bus import EventBus

logger = get_logger(__name__)


class StateMachine:
    """
    Manages system state with validation and history tracking.
    Emits STATE_CHANGED events on transitions.
    """

    # Valid state transitions (from_state -> list of valid to_states)
    VALID_TRANSITIONS = {
        SystemState.IDLE: [SystemState.LISTENING, SystemState.ERROR],
        SystemState.LISTENING: [SystemState.PROCESSING, SystemState.IDLE, SystemState.ERROR],
        SystemState.PROCESSING: [SystemState.PARSING, SystemState.IDLE, SystemState.ERROR],
        SystemState.PARSING: [SystemState.EXECUTING, SystemState.RESPONDING, SystemState.IDLE, SystemState.ERROR],
        SystemState.EXECUTING: [SystemState.RESPONDING, SystemState.IDLE, SystemState.ERROR],
        SystemState.RESPONDING: [SystemState.IDLE, SystemState.ERROR],
        SystemState.ERROR: [SystemState.IDLE],
    }

    def __init__(self, event_bus: EventBus, initial_state: SystemState = SystemState.IDLE):
        """
        Initialize state machine.

        Args:
            event_bus: Event bus for publishing state changes
            initial_state: Starting state
        """
        self.event_bus = event_bus
        self.current_state = initial_state
        self.previous_state: Optional[SystemState] = None
        self.lock = asyncio.Lock()

        # State history for debugging (keep last 100 transitions)
        self.history: List[Tuple[SystemState, SystemState, datetime]] = []
        self.max_history = 100

        logger.info(f"StateMachine initialized in {initial_state.value} state")

    async def transition(self, new_state: SystemState, force: bool = False) -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            force: If True, skip validation (use with caution)

        Returns:
            True if transition successful, False if invalid
        """
        async with self.lock:
            old_state = self.current_state

            # Skip if already in target state
            if old_state == new_state:
                logger.debug(f"Already in {new_state.value} state")
                return True

            # Validate transition
            if not force:
                if old_state not in self.VALID_TRANSITIONS:
                    logger.error(f"No transitions defined for {old_state.value}")
                    return False

                if new_state not in self.VALID_TRANSITIONS[old_state]:
                    logger.warning(
                        f"Invalid transition: {old_state.value} -> {new_state.value}"
                    )
                    return False

            # Execute transition
            self.previous_state = old_state
            self.current_state = new_state

            # Record in history
            self.history.append((old_state, new_state, datetime.now()))
            if len(self.history) > self.max_history:
                self.history.pop(0)

            logger.info(f"State transition: {old_state.value} -> {new_state.value}")

            # Emit state change event
            await self.event_bus.publish(
                EventType.STATE_CHANGED,
                {
                    'old_state': old_state,
                    'new_state': new_state,
                    'timestamp': datetime.now()
                }
            )

            return True

    def get_state(self) -> SystemState:
        """Get current state."""
        return self.current_state

    def get_previous_state(self) -> Optional[SystemState]:
        """Get previous state."""
        return self.previous_state

    def is_state(self, state: SystemState) -> bool:
        """
        Check if currently in a specific state.

        Args:
            state: State to check

        Returns:
            True if in that state
        """
        return self.current_state == state

    def is_busy(self) -> bool:
        """
        Check if system is busy (not idle).

        Returns:
            True if not in IDLE state
        """
        return self.current_state != SystemState.IDLE

    def get_history(self, limit: int = 10) -> List[Tuple[SystemState, SystemState, datetime]]:
        """
        Get recent state transition history.

        Args:
            limit: Number of recent transitions to return

        Returns:
            List of (from_state, to_state, timestamp) tuples
        """
        return self.history[-limit:]

    async def reset(self) -> None:
        """Reset to IDLE state (forced transition)."""
        await self.transition(SystemState.IDLE, force=True)
        logger.info("State machine reset to IDLE")

    async def to_error(self, error_info: Optional[dict] = None) -> None:
        """
        Transition to ERROR state and publish error event.

        Args:
            error_info: Optional error details
        """
        await self.transition(SystemState.ERROR, force=True)

        if error_info:
            await self.event_bus.publish(EventType.ERROR_OCCURRED, error_info)
