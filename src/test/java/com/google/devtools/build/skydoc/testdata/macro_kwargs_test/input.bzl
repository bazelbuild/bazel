"""Tests for functions which use *args or **kwargs"""

def macro_with_kwargs(name, config, deps = [], **kwargs):
    """My kwargs macro is the best.

    This is a long multi-line doc string.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer
    elementum, diam vitae tincidunt pulvinar, nunc tortor volutpat dui,
    vitae facilisis odio ligula a tortor. Donec ullamcorper odio eget ipsum tincidunt,
    vel mollis eros pellentesque.

    Args:
      name: The name of the test rule.
      config: Config to use for my macro
      deps: List of my macro's dependencies
      **kwargs: Other attributes to include

    Returns:
      An empty list.
    """
    _ignore = [name, config, deps, kwargs]
    return []

def macro_with_args(name, *args):
    """My args macro is OK.

    Args:
      name: The name of the test rule.
      *args: Other arguments to include

    Returns:
      An empty list.
    """
    _ignore = [name, args]
    return []

def macro_with_both(name, number = 3, *args, **kwargs):
    """Oh wow this macro has both.

    Not much else to say.

    Args:
      name: The name of the test rule.
      number: Some number used for important things
      *args: Other arguments to include
      **kwargs: Other attributes to include

    Returns:
      An empty list.
    """
    _ignore = [name, number, args, kwargs]
    return []
