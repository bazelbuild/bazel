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

# Private constant (starts with _) - NOT importable
_INTERNAL_VALUE = "internal"

def sample_function():
    """A sample function that uses the global constants.

    Returns:
        The version string
    """
    return VERSION
