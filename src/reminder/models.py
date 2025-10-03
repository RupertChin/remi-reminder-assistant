"""
Data models for reminders.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Reminder:
    """Reminder data model."""

    id: Optional[int]
    content: str
    scheduled_time: datetime
    created_at: datetime
    is_active: bool = True
    is_completed: bool = False
    user_input: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        time_str = self.scheduled_time.strftime("%Y-%m-%d %H:%M")
        status = "✓" if self.is_completed else ("●" if self.is_active else "○")
        return f"{status} {self.content} @ {time_str}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'scheduled_time': self.scheduled_time.isoformat(),
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
            'is_completed': self.is_completed,
            'user_input': self.user_input
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Reminder':
        """Create from dictionary."""
        return cls(
            id=data.get('id'),
            content=data['content'],
            scheduled_time=datetime.fromisoformat(data['scheduled_time']),
            created_at=datetime.fromisoformat(data['created_at']),
            is_active=data.get('is_active', True),
            is_completed=data.get('is_completed', False),
            user_input=data.get('user_input')
        )
