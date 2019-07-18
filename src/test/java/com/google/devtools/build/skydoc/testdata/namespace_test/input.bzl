"""A test that verifies documenting a namespace of functions."""

def _min(integers):
    """Returns the minimum of given elements.

    Args:
      integers: A list of integers. Must not be empty.

    Returns:
      The minimum integer in the given list.
    """
    _ignore = [integers]
    return 42

def _assert_non_empty(some_list, other_list):
    """Asserts the two given lists are not empty.

    Args:
      some_list: The first list
      other_list: The second list
    """
    _ignore = [some_list, other_list]
    fail("Not implemented")

def _join_strings(strings, delimiter = ", "):
    """Joins the given strings with a delimiter.

    Args:
      strings: A list of strings to join.
      delimiter: The delimiter to use

    Returns:
      The joined string.
    """
    _ignore = [strings, delimiter]
    return ""

my_namespace = struct(
    dropped_field = "Note this field should not be documented",
    assert_non_empty = _assert_non_empty,
    min = _min,
    join_strings = _join_strings,
)
