"""
APScheduler wrapper for reminder scheduling.
Manages timed reminder execution with persistence.
"""

import asyncio
from typing import Callable, Optional
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.date import DateTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from config.logging_config import get_logger
from config import settings
from src.reminder.models import Reminder

logger = get_logger(__name__)


class ReminderScheduler:
    """
    APScheduler wrapper for managing reminder jobs.
    Provides persistence and recovery across restarts.
    """

    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize scheduler.

        Args:
            callback: Async function to call when reminder triggers
                     Signature: async def callback(reminder: Reminder) -> None
        """
        self.callback = callback

        # Configure job store for persistence
        jobstores = {
            'default': SQLAlchemyJobStore(url=f'sqlite:///{settings.JOBS_DB_PATH}')
        }

        # Configure scheduler
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            job_defaults={
                'coalesce': settings.SCHEDULER_COALESCE,
                'max_instances': settings.SCHEDULER_MAX_INSTANCES,
                'misfire_grace_time': settings.SCHEDULER_MISFIRE_GRACE_TIME
            }
        )

        # Add event listeners
        self.scheduler.add_listener(
            self._job_executed,
            EVENT_JOB_EXECUTED
        )
        self.scheduler.add_listener(
            self._job_error,
            EVENT_JOB_ERROR
        )

        logger.info("ReminderScheduler initialized")

    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started")

            # Log recovered jobs
            jobs = self.scheduler.get_jobs()
            if jobs:
                logger.info(f"Recovered {len(jobs)} scheduled jobs")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the scheduler.

        Args:
            wait: Wait for running jobs to complete
        """
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler shutdown")

    def schedule_reminder(self, reminder: Reminder) -> bool:
        """
        Schedule a reminder for execution.

        Args:
            reminder: Reminder object to schedule

        Returns:
            True if scheduled successfully
        """
        if reminder.scheduled_time <= datetime.now():
            logger.warning(
                f"Cannot schedule reminder {reminder.id} in the past: "
                f"{reminder.scheduled_time}"
            )
            return False

        try:
            job_id = f"reminder_{reminder.id}"

            # Check if job already exists
            existing_job = self.scheduler.get_job(job_id)
            if existing_job:
                logger.warning(f"Job {job_id} already exists, replacing")
                self.scheduler.remove_job(job_id)

            # Schedule the job
            self.scheduler.add_job(
                func=self._execute_reminder,
                trigger=DateTrigger(run_date=reminder.scheduled_time),
                id=job_id,
                args=[reminder],
                replace_existing=True
            )

            logger.info(
                f"Scheduled reminder {reminder.id} for "
                f"{reminder.scheduled_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to schedule reminder {reminder.id}: {e}", exc_info=True)
            return False

    def cancel_reminder(self, reminder_id: int) -> bool:
        """
        Cancel a scheduled reminder.

        Args:
            reminder_id: Reminder ID to cancel

        Returns:
            True if cancelled successfully
        """
        job_id = f"reminder_{reminder_id}"

        try:
            self.scheduler.remove_job(job_id)
            logger.info(f"Cancelled reminder {reminder_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cancel reminder {reminder_id}: {e}")
            return False

    def reschedule_reminder(self, reminder: Reminder) -> bool:
        """
        Reschedule an existing reminder.

        Args:
            reminder: Reminder with updated scheduled_time

        Returns:
            True if rescheduled successfully
        """
        # Cancel existing job
        self.cancel_reminder(reminder.id)

        # Schedule new job
        return self.schedule_reminder(reminder)

    def get_scheduled_count(self) -> int:
        """
        Get number of currently scheduled jobs.

        Returns:
            Number of scheduled jobs
        """
        return len(self.scheduler.get_jobs())

    async def _execute_reminder(self, reminder: Reminder) -> None:
        """
        Execute a reminder (internal).

        Args:
            reminder: Reminder to execute
        """
        logger.info(f"Executing reminder {reminder.id}: {reminder.content}")

        if self.callback:
            try:
                await self.callback(reminder)
            except Exception as e:
                logger.error(
                    f"Error in reminder callback for {reminder.id}: {e}",
                    exc_info=True
                )

    def _job_executed(self, event) -> None:
        """
        Event listener for successful job execution.

        Args:
            event: Job execution event
        """
        logger.debug(f"Job executed: {event.job_id}")

    def _job_error(self, event) -> None:
        """
        Event listener for job errors.

        Args:
            event: Job error event
        """
        logger.error(
            f"Job error: {event.job_id}, "
            f"exception: {event.exception}",
            exc_info=event.exception
        )

    def set_callback(self, callback: Callable) -> None:
        """
        Set or update the reminder callback function.

        Args:
            callback: Async function to call when reminder triggers
        """
        self.callback = callback
        logger.debug("Reminder callback set")

    def get_next_run_time(self, reminder_id: int) -> Optional[datetime]:
        """
        Get next run time for a reminder.

        Args:
            reminder_id: Reminder ID

        Returns:
            Next run time or None if not scheduled
        """
        job_id = f"reminder_{reminder_id}"
        job = self.scheduler.get_job(job_id)

        if job:
            return job.next_run_time

        return None
