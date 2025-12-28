"""Test for transitive forwarding to native rules via loaded symbols.

This tests the pattern where:
1. User function forwards to a loaded symbol
2. That loaded symbol forwards to a native rule
3. Constellate should detect this and create a RuleMacro
"""

load("//src/test/java/build/stack/devtools/build/constellate/testdata:compatibility_proxy.bzl", _java_binary = "java_binary")

def java_binary(**attrs):
    """Bazel java_binary rule.

    https://docs.bazel.build/versions/master/be/java.html#java_binary

    Args:
      **attrs: Rule attributes
    """
    _java_binary(**attrs)
