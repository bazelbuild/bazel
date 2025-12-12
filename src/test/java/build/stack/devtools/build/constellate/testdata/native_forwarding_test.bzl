"""Test for detecting forwarding to native rules."""

def my_java_library(**kwargs):
    """Wrapper that forwards all kwargs to native.java_library.

    This should be detected as a RuleMacro that forwards to library_rules.java_library.

    Args:
        **kwargs: Arguments to forward to native.java_library
    """
    native.java_library(**kwargs)

def my_java_binary(name, **kwargs):
    """Wrapper that forwards name and kwargs to native.java_binary.

    This should be detected as a RuleMacro that forwards to binary_rules.java_binary.

    Args:
        name: Target name
        **kwargs: Additional arguments
    """
    native.java_binary(
        name = name,
        **kwargs
    )

def my_java_test(name, srcs):
    """Wrapper that explicitly forwards name to native.java_test.

    This should be detected as a RuleMacro that forwards name to test_rules.java_test.

    Args:
        name: Target name
        srcs: Source files
    """
    native.java_test(
        name = name,
        srcs = srcs,
    )

def mixed_native_wrapper(name, **kwargs):
    """Wrapper that calls both native.java_library and native.java_binary.

    This should detect forwarding to both native rules.

    Args:
        name: Base target name
        **kwargs: Additional arguments
    """
    native.java_library(
        name = name + "_lib",
        **kwargs
    )
    native.java_binary(
        name = name + "_bin",
        **kwargs
    )

def non_forwarding_helper(value):
    """Helper function that doesn't call any rules.

    This should NOT be detected as a RuleMacro.

    Args:
        value: Some value

    Returns:
        Processed value
    """
    return value.strip()

def indirect_native_call(name, rule_type):
    """Function that conditionally calls native rules.

    This demonstrates more complex patterns.

    Args:
        name: Target name
        rule_type: Type of rule to create
    """
    if rule_type == "java_library":
        native.java_library(name = name)
    elif rule_type == "java_binary":
        native.java_binary(name = name)
