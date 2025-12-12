"""Test for direct native rule forwarding without load statements."""

def java_binary(**attrs):
    """Bazel java_binary rule.

    https://docs.bazel.build/versions/master/be/java.html#java_binary

    Args:
      **attrs: Rule attributes
    """
    native.java_binary(**attrs)
