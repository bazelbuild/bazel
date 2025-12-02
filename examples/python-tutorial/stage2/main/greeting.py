"""A module for greeting functionality."""


def get_greeting(name: str = "World") -> str:
    """Return a greeting message.

    Args:
        name: The name to greet. Defaults to "World".

    Returns:
        A greeting string.
    """
    return f"Hello, {name}!"
