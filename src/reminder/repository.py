"""
Database repository for reminders.
Uses SQLite with repository pattern for thread-safe CRUD operations.
"""

import sqlite3
import threading
from typing import List, Optional
from datetime import datetime
from contextlib import contextmanager

from config.logging_config import get_logger
from config import settings
from src.reminder.models import Reminder

logger = get_logger(__name__)


class ReminderRepository:
    """
    Thread-safe SQLite repository for reminders.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS reminders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        scheduled_time DATETIME NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        is_completed BOOLEAN DEFAULT 0,
        user_input TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_scheduled_time ON reminders(scheduled_time);
    CREATE INDEX IF NOT EXISTS idx_is_active ON reminders(is_active);
    """

    def __init__(self, db_path: str = None):
        """
        Initialize repository.

        Args:
            db_path: Path to SQLite database (default from settings)
        """
        self.db_path = db_path or str(settings.DB_PATH)
        self.lock = threading.Lock()

        logger.info(f"ReminderRepository initialized: {self.db_path}")
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create database schema if it doesn't exist."""
        try:
            with self._get_connection() as conn:
                conn.executescript(self.SCHEMA)
                conn.commit()
            logger.info("Database schema initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise

    @contextmanager
    def _get_connection(self):
        """
        Get database connection (context manager).

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
        finally:
            conn.close()

    def create(self, content: str, scheduled_time: datetime,
               user_input: str = None) -> Reminder:
        """
        Create a new reminder.

        Args:
            content: Reminder content
            scheduled_time: When to trigger the reminder
            user_input: Original user input (optional)

        Returns:
            Created Reminder object
        """
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO reminders (content, scheduled_time, user_input)
                    VALUES (?, ?, ?)
                    """,
                    (content, scheduled_time, user_input)
                )
                conn.commit()

                reminder_id = cursor.lastrowid

                # Fetch the created reminder
                reminder = self.get_by_id(reminder_id)

                logger.info(f"Created reminder: {reminder}")
                return reminder

    def get_by_id(self, reminder_id: int) -> Optional[Reminder]:
        """
        Get reminder by ID.

        Args:
            reminder_id: Reminder ID

        Returns:
            Reminder object or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM reminders WHERE id = ?",
                (reminder_id,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_reminder(row)
            return None

    def get_all_active(self) -> List[Reminder]:
        """
        Get all active reminders.

        Returns:
            List of active Reminder objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM reminders
                WHERE is_active = 1 AND is_completed = 0
                ORDER BY scheduled_time ASC
                """
            )
            rows = cursor.fetchall()

            return [self._row_to_reminder(row) for row in rows]

    def get_upcoming(self, limit: int = 10) -> List[Reminder]:
        """
        Get upcoming active reminders.

        Args:
            limit: Maximum number to return

        Returns:
            List of upcoming Reminder objects
        """
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM reminders
                WHERE is_active = 1 AND is_completed = 0 AND scheduled_time > ?
                ORDER BY scheduled_time ASC
                LIMIT ?
                """,
                (now, limit)
            )
            rows = cursor.fetchall()

            return [self._row_to_reminder(row) for row in rows]

    def update(self, reminder: Reminder) -> bool:
        """
        Update an existing reminder.

        Args:
            reminder: Reminder object with updated fields

        Returns:
            True if updated successfully
        """
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE reminders
                    SET content = ?,
                        scheduled_time = ?,
                        is_active = ?,
                        is_completed = ?,
                        user_input = ?
                    WHERE id = ?
                    """,
                    (
                        reminder.content,
                        reminder.scheduled_time,
                        reminder.is_active,
                        reminder.is_completed,
                        reminder.user_input,
                        reminder.id
                    )
                )
                conn.commit()

                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"Updated reminder {reminder.id}")
                return success

    def mark_completed(self, reminder_id: int) -> bool:
        """
        Mark reminder as completed.

        Args:
            reminder_id: Reminder ID

        Returns:
            True if marked successfully
        """
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE reminders
                    SET is_completed = 1
                    WHERE id = ?
                    """,
                    (reminder_id,)
                )
                conn.commit()

                success = cursor.rowcount > 0
                if success:
                    logger.info(f"Marked reminder {reminder_id} as completed")
                return success

    def delete(self, reminder_id: int, soft: bool = True) -> bool:
        """
        Delete a reminder.

        Args:
            reminder_id: Reminder ID
            soft: If True, mark as inactive; if False, permanently delete

        Returns:
            True if deleted successfully
        """
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if soft:
                    # Soft delete (mark as inactive)
                    cursor.execute(
                        """
                        UPDATE reminders
                        SET is_active = 0
                        WHERE id = ?
                        """,
                        (reminder_id,)
                    )
                else:
                    # Hard delete
                    cursor.execute(
                        "DELETE FROM reminders WHERE id = ?",
                        (reminder_id,)
                    )

                conn.commit()

                success = cursor.rowcount > 0
                if success:
                    delete_type = "soft deleted" if soft else "deleted"
                    logger.info(f"Reminder {reminder_id} {delete_type}")
                return success

    def search(self, query: str) -> List[Reminder]:
        """
        Search reminders by content.

        Args:
            query: Search query

        Returns:
            List of matching Reminder objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM reminders
                WHERE is_active = 1 AND content LIKE ?
                ORDER BY scheduled_time ASC
                """,
                (f"%{query}%",)
            )
            rows = cursor.fetchall()

            return [self._row_to_reminder(row) for row in rows]

    def _row_to_reminder(self, row: sqlite3.Row) -> Reminder:
        """
        Convert database row to Reminder object.

        Args:
            row: SQLite row

        Returns:
            Reminder object
        """
        return Reminder(
            id=row['id'],
            content=row['content'],
            scheduled_time=datetime.fromisoformat(row['scheduled_time']),
            created_at=datetime.fromisoformat(row['created_at']),
            is_active=bool(row['is_active']),
            is_completed=bool(row['is_completed']),
            user_input=row['user_input']
        )

    def cleanup_old_reminders(self, days: int = 30) -> int:
        """
        Clean up old completed/inactive reminders.

        Args:
            days: Delete reminders older than this many days

        Returns:
            Number of reminders deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM reminders
                    WHERE (is_completed = 1 OR is_active = 0)
                    AND created_at < ?
                    """,
                    (cutoff_date,)
                )
                conn.commit()

                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old reminders")

                return deleted_count


# Import at end to avoid circular dependency
from datetime import timedelta
