"""A test that verifies documenting a multi-leveled namespace of functions with whitelist symbols.
The whitelist symbols should cause everything in my_namespace to to be documented, but only a
specific symbol in other_namespace to be documented."""

def _min(integers):
    """Returns the minimum of given elements."""
    _ignore = [integers]
    return 42

def _does_nothing():
    """This function does nothing."""
    pass

my_namespace = struct(
    dropped_field = "Note this field should not be documented",
    min = _min,
    math = struct(min = _min),
)

other_namespace = struct(
    foo = struct(nothing = _does_nothing),
    min = _min,
)
