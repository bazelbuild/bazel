"""A test that verifies documenting a multi-leveled namespace of functions."""

def _min(integers):
    """Returns the minimum of given elements.

    Args:
      integers: A list of integers. Must not be empty.

    Returns:
      The minimum integer in the given list.
    """
    _ignore = [integers]
    return 42

def _does_nothing():
    """This function does nothing."""
    pass

my_namespace = struct(
    dropped_field = "Note this field should not be documented",
    min = _min,
    math = struct(min = _min),
    foo = struct(
        bar = struct(baz = _does_nothing),
        num = 12,
        string = "Hello!",
    ),
    one = struct(
        two = struct(min = _min),
        three = struct(does_nothing = _does_nothing),
    ),
)
