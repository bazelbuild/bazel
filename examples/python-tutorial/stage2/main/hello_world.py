"""A Hello World program using a greeting library."""

from main import greeting


def main():
    """Print a greeting message using the greeting module."""
    message = greeting.get_greeting()
    print(message)

    # Also greet Bazel!
    message = greeting.get_greeting("Bazel")
    print(message)


if __name__ == "__main__":
    main()
