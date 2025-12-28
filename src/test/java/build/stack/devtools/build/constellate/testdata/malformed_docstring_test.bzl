"""Test file with one malformed docstring and one good docstring."""

def good_function(param1):
    """This is a properly formatted function.

    Args:
        param1: A parameter

    Returns:
        The result
    """
    return param1

def bad_function(param1):
    """This is a poorly formatted function.

    Returns:
        The result

    Args:
        param1: This violates the rule that Args should come before Returns
    """
    return param1

def another_good_function(param2):
    """Another properly formatted function.

    Args:
        param2: Another parameter
    """
    pass
