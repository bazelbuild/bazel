"""A Hello World program using multiple packages."""

from lib import time_utils
from main import greeting


def main():
    """Print time-aware greeting messages."""
    # Get the current time of day
    time_of_day = time_utils.get_time_of_day()
    current_time = time_utils.get_formatted_time()

    # Print a simple greeting
    print(greeting.get_greeting())

    # Print a time-aware greeting
    print(greeting.get_time_greeting("Bazel User", time_of_day))

    # Print the current time
    print(f"The current time is {current_time}")


if __name__ == "__main__":
    main()
