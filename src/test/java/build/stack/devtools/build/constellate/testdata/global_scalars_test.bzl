"""Test file for global scalar value extraction.

This file contains various global scalar constants that should be captured
in the Module.global field.
"""

# String constants
VERSION = "1.2.3"
TOOLCHAIN_NAME = "my_toolchain"
DEFAULT_TAG = "latest"

# Integer constants
MAX_SIZE = 100
DEFAULT_TIMEOUT = 60

# Boolean constants
ENABLE_FEATURE = True
DEBUG_MODE = False

# List constants
SUPPORTED_PLATFORMS = ["linux", "darwin", "windows"]
EMPTY_LIST = []

# List comprehension
NUMBERS = [x * 2 for x in range(5)]  # Should be [0, 2, 4, 6, 8]
FILTERED_NUMBERS = [x for x in range(10) if x % 2 == 0]  # Should be [0, 2, 4, 6, 8]

# Private constant (starts with _) - NOT importable
_INTERNAL_VALUE = "internal"

def sample_function():
    """A sample function that uses the global constants.

    Returns:
        The version string
    """
    return VERSION
