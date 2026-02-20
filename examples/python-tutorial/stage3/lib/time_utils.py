"""A module for time-related utilities."""

from datetime import datetime


def get_time_of_day() -> str:
    """Return a time-of-day greeting based on the current hour.

    Returns:
        A string indicating morning, afternoon, or evening.
    """
    hour = datetime.now().hour

    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    else:
        return "evening"


def get_formatted_time() -> str:
    """Return the current time in a human-readable format.

    Returns:
        A formatted time string.
    """
    return datetime.now().strftime("%H:%M:%S")
