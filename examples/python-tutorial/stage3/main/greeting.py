"""A module for greeting functionality."""


def get_greeting(name: str = "World") -> str:
    """Return a greeting message.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"


def get_time_greeting(name: str, time_of_day: str) -> str:
    """Return a time-aware greeting message.

    Args:
        name: The name to greet.
        time_of_day: The time of day (morning, afternoon, evening).

    Returns:
        A time-aware greeting string.
    """
    return f"Good {time_of_day}, {name}!"
